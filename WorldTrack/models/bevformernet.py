import torch
import torch.nn as nn

import utils.geom
import utils.vox
import utils.basic

from models.encoder import Encoder_res101, Encoder_res50, Encoder_eff, Encoder_swin_t, Encoder_res18
from models.decoder import Decoder
from models.ops.ms_deform_attn import MSDeformAttn, MSDeformAttn3D


class VanillaSelfAttention(nn.Module):
    def __init__(self, dim=128, dropout=0.5):
        super(VanillaSelfAttention, self).__init__()
        self.dim = dim
        self.dropout = nn.Dropout(dropout)
        self.deformable_attention = MSDeformAttn(d_model=dim, n_levels=1, n_heads=4, n_points=8)
        self.output_proj = nn.Linear(dim, dim)

    def forward(self, query, Y, X, query_pos=None):
        """
        query: (B, Y*X, C)
        query_pos: (B, Y*X, C)
        """
        inp_residual = query.clone()

        if query_pos is not None:
            query = query + query_pos

        B, N, C = query.shape
        # Y, X = 200, 200
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, Y - 0.5, Y, dtype=torch.float, device=query.device),
            torch.linspace(0.5, X - 0.5, X, dtype=torch.float, device=query.device),
            indexing='ij'
        )
        ref_y = ref_y.reshape(-1)[None] / Y
        ref_x = ref_x.reshape(-1)[None] / X
        reference_points = torch.stack((ref_y, ref_x), -1)
        reference_points = reference_points.repeat(B, 1, 1).unsqueeze(2)  # (B, Y*X, 1, 2)

        input_spatial_shapes = query.new_zeros([1, 2]).long()
        input_spatial_shapes[0, 0] = Y
        input_spatial_shapes[0, 1] = X
        input_level_start_index = query.new_zeros([1, ]).long()
        queries = self.deformable_attention(query=query,
                                            reference_points=reference_points,
                                            input_flatten=query.clone(),
                                            input_spatial_shapes=input_spatial_shapes,
                                            input_level_start_index=input_level_start_index)

        queries = self.output_proj(queries)

        return self.dropout(queries) + inp_residual


class SpatialCrossAttention(nn.Module):
    # From https://github.com/zhiqi-li/BEVFormer
    def __init__(self, dim=128, dropout=0.5):
        super(SpatialCrossAttention, self).__init__()
        self.dim = dim
        self.dropout = nn.Dropout(dropout)
        self.deformable_attention = MSDeformAttn3D(embed_dims=dim, num_heads=4, num_levels=1, num_points=8)
        self.output_proj = nn.Linear(dim, dim)

    def forward(self, query, key, value, query_pos=None, reference_points_cam=None, spatial_shapes=None, bev_mask=None):
        """
        query: bev_queries (B, Y*X, C)
        key: bev_keys (S, Hf*Wf, B, C)
        value: bev_keys (S, Hf*Wf, B, C)
        query_pos: bev_queries_pos (B, Y*X, C)
        reference_points_cam: reference_points_cam (S, B, Y*X, Z, 2) normalized
        spatial_shapes: spatial_shapes (1,2) [[Hf,Wf]]
        bev_mask: bev_mask (S. B, Y*X, Z)
        """
        inp_residual = query
        slots = torch.zeros_like(query)  # (B, Y*X, C)

        if query_pos is not None:
            query = query + query_pos

        B, N, C = query.shape  # N=Y*X
        S, M, _, _ = key.shape  # M=Hf*Wf
        D = reference_points_cam.size(3)  # Z

        """
        Traverse the S cameras, 
        take the index of the valid coordinates of the BEV projection to each feature map
        find out which cam having most valid BEV projection coords and get the max_len
        """
        # for i, mask_per_img in enumerate(bev_mask):
        #     # if once valid through Z-axis, query it
        #     index_query_per_img = mask_per_img.sum(-1)
        #     indexes.append(index_query_per_img)
        max_len = bev_mask.sum(dim=-1).gt(0).sum(-1).max()

        # for each batch and cam reconstruct the query and reference_points
        queries_rebatch = query.new_zeros([B, S, max_len, self.dim])
        reference_points_rebatch = reference_points_cam.new_zeros([B, S, max_len, D, 2])

        for j in range(B):
            for i, reference_points_per_img in enumerate(reference_points_cam):
                index_query_per_img = bev_mask[i, j].sum(-1).nonzero().squeeze(-1)
                queries_rebatch[j, i, :len(index_query_per_img)] = query[j, index_query_per_img]
                reference_points_rebatch[j, i, :len(index_query_per_img)] = reference_points_per_img[
                    j, index_query_per_img]

        # take feature map as key and value of attention module
        key = key.permute(2, 0, 1, 3).reshape(B * S, M, C)
        value = value.permute(2, 0, 1, 3).reshape(B * S, M, C)

        level_start_index = query.new_zeros([1, ]).long()
        reference_points_rebatch = reference_points_rebatch.view(B * S, max_len, D, 2)
        # reference_points_rebatch = torch.mean(reference_points_rebatch, dim=-2, keepdim=True)
        """
        MSDeformAttn3D 
        num_query: max_len   num_key: Hf*Wf
        query (Tensor): Query of Transformer (bs, num_query, embed_dims)
        key (Tensor): The key tensor  (bs, num_key,  embed_dims)
        value (Tensor): The value tensor  (bs, num_key,  embed_dims)
        reference_points (Tensor):  The normalized reference points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0), bottom-right (1, 1)
        spatial_shapes (Tensor): Spatial shape of features in different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
        level_start_index (Tensor): The start index of each level. A tensor has shape (num_levels, )
                and can be represented as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...]
        """
        queries = self.deformable_attention(
            query=queries_rebatch.view(B * S, max_len, self.dim),
            key=key,
            value=value,
            reference_points=reference_points_rebatch,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index
        ).view(B, S, max_len, self.dim)

        for j in range(B):
            for i in range(S):
                # slots (B, Y*X, C)
                index_query_per_img = bev_mask[i, j].sum(-1).nonzero().squeeze(-1)
                slots[j, index_query_per_img] += queries[j, i, :len(index_query_per_img)]

        count = bev_mask.sum(-1) > 0
        count = count.permute(1, 2, 0).sum(-1)
        count = torch.clamp(count, min=1.0)
        slots = slots / count[..., None]
        slots = self.output_proj(slots)

        return self.dropout(slots) + inp_residual


# no radar/lidar integration
class Bevformernet(nn.Module):
    def __init__(self, Y, Z, X,
                 rand_flip=False,
                 latent_dim=128,
                 feat2d_dim=128,
                 num_classes=None,
                 z_sign=1,
                 encoder_type='swin_t'):
        super(Bevformernet, self).__init__()
        assert (encoder_type in ['res101', 'res50', 'res18', 'effb0', 'effb4', 'swin_t'])

        self.Y, self.Z, self.X = Y, Z, X
        self.rand_flip = rand_flip
        self.latent_dim = latent_dim
        self.encoder_type = encoder_type
        self.use_radar = False
        self.use_lidar = False

        self.mean = torch.as_tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1).float().cuda()
        self.std = torch.as_tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1).float().cuda()
        self.z_sign = z_sign

        # Encoder
        self.feat2d_dim = feat2d_dim
        if encoder_type == 'res101':
            self.encoder = Encoder_res101(feat2d_dim)
        elif encoder_type == 'res50':
            self.encoder = Encoder_res50(feat2d_dim)
        elif encoder_type == 'res18':
            self.encoder = Encoder_res18(feat2d_dim)
        elif encoder_type == 'effb0':
            self.encoder = Encoder_eff(feat2d_dim, version='b0')
        elif encoder_type == 'swin_t':
            self.encoder = Encoder_swin_t(feat2d_dim)
        else:
            # effb4
            self.encoder = Encoder_eff(feat2d_dim, version='b4')

        # BEVFormer self & cross attention layers
        self.bev_keys = nn.Linear(feat2d_dim, latent_dim)
        self.bev_queries = nn.Parameter(0.1 * torch.randn(latent_dim, Y, X))  # C, Y, X
        self.bev_queries_pos = nn.Parameter(0.1 * torch.randn(latent_dim, Y, X))  # C, Y, X
        num_layers = 6
        self.num_layers = num_layers
        self.self_attn_layers = nn.ModuleList([
            VanillaSelfAttention(dim=latent_dim) for _ in range(num_layers)
        ])  # deformable self attention
        self.norm1_layers = nn.ModuleList([
            nn.LayerNorm(latent_dim) for _ in range(num_layers)
        ])
        self.cross_attn_layers = nn.ModuleList([
            SpatialCrossAttention(dim=latent_dim) for _ in range(num_layers)
        ])
        self.norm2_layers = nn.ModuleList([
            nn.LayerNorm(latent_dim) for _ in range(num_layers)
        ])
        ffn_dim = 512
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(latent_dim, ffn_dim), nn.ReLU(), nn.Linear(ffn_dim, latent_dim)) for _ in
            range(num_layers)
        ])
        self.norm3_layers = nn.ModuleList([
            nn.LayerNorm(latent_dim) for _ in range(num_layers)
        ])

        self.bev_temporal = nn.Sequential(
            nn.Conv2d(latent_dim * 2, latent_dim, kernel_size=3, padding=1),
            nn.InstanceNorm2d(latent_dim), nn.ReLU(),
            nn.Conv2d(latent_dim, latent_dim, kernel_size=1),
        )

        # Decoder
        self.decoder = Decoder(
            in_channels=latent_dim,
            n_classes=num_classes,
            feat2d=feat2d_dim,
        )

        # Weights
        self.center_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.offset_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.tracking_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.size_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.rot_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)

    def forward(self, rgb_cams, pix_T_cams, cams_T_global, vox_util, ref_T_global, prev_bev=None):
        """
        B = batch size, S = number of cameras, C = 3, H = img height, W = img width
        rgb_cams: (B,S,C,H,W)
        pix_T_cams: (B,S,4,4)
        cams_T_global: (B,S,4,4)
        vox_util: vox util object
        ref_T_global: (B,4,4)
        """
        B, S, C, H, W = rgb_cams.shape
        B0 = B * S
        assert (C == 3)
        # reshape tensors
        __p = lambda x: utils.basic.pack_seqdim(x, B)
        __u = lambda x: utils.basic.unpack_seqdim(x, B)
        rgb_cams_ = __p(rgb_cams)  # B*S,3,H,W
        pix_T_cams_ = __p(pix_T_cams)  # B*S,4,4
        cams_T_global_ = __p(cams_T_global)  # B*S,4,4
        global_T_cams_ = torch.inverse(cams_T_global_)  # B*S,4,4
        ref_T_cams = torch.matmul(ref_T_global.repeat(S, 1, 1), global_T_cams_)  # B*S,4,4
        cams_T_ref_ = torch.inverse(ref_T_cams)  # B*S,4,4

        # rgb encoder
        device = rgb_cams_.device
        rgb_cams_ = (rgb_cams_ - self.mean.to(device)) / self.std.to(device)
        feat_cams_ = self.encoder(rgb_cams_)  # B*S,128,H/8,W/8
        _, C, Hf, Wf = feat_cams_.shape
        feat_cams = __u(feat_cams_)  # B,S,C,Hf,Wf
        Y, Z, X = self.Y, self.Z, self.X

        # compute the image locations (no flipping for now)
        xyz_mem_ = utils.basic.gridcloud3d(B0, Y, Z, X, norm=False, device=rgb_cams.device)  # B0, Z*Y*X, 3
        xyz_ref_ = vox_util.Mem2Ref(xyz_mem_, Y, Z, X, assert_cube=False)
        xyz_cams_ = utils.geom.apply_4x4(cams_T_ref_, xyz_ref_)
        xy_cams_ = utils.geom.camera2pixels(xyz_cams_, pix_T_cams_)  # B0, N, 2
        # bev coords project to pixel level and normalized  S,B,Y*X,Z,2
        reference_points_cam = xy_cams_.reshape(B, S, Y, Z, X, 2).permute(1, 0, 2, 4, 3, 5).reshape(S, B, Y * X, Z, 2)
        reference_points_cam[..., 0:1] = reference_points_cam[..., 0:1] / float(W)
        reference_points_cam[..., 1:2] = reference_points_cam[..., 1:2] / float(H)
        cam_x = xyz_cams_[..., 2].reshape(B, S, Y, Z, X, 1).permute(1, 0, 2, 4, 3, 5).reshape(S, B, Y * X, Z, 1)
        bev_mask = ((reference_points_cam[..., 1:2] >= 0.0)
                    & (reference_points_cam[..., 1:2] <= 1.0)
                    & (reference_points_cam[..., 0:1] <= 1.0)
                    & (reference_points_cam[..., 0:1] >= 0.0)
                    & (self.z_sign * cam_x >= 0.0)
                    ).squeeze(-1)  # S,B,Y*X,Z valid point or not

        # self-attention prepare
        bev_queries = self.bev_queries.clone().unsqueeze(0).repeat(B, 1, 1, 1).reshape(B, self.latent_dim, -1) \
            .permute(0, 2, 1)  # B, Y*X, C
        bev_queries_pos = self.bev_queries_pos.clone().unsqueeze(0).repeat(B, 1, 1, 1) \
            .reshape(B, self.latent_dim, -1).permute(0, 2, 1)  # B, Y*X, C

        # cross-attention prepare
        bev_keys = feat_cams.reshape(B, S, C, Hf * Wf).permute(1, 3, 0, 2)  # S, Hf*Wf, B, C
        bev_keys = self.bev_keys(bev_keys)
        spatial_shapes = bev_queries.new_zeros([1, 2]).long()
        spatial_shapes[0, 0] = Hf
        spatial_shapes[0, 1] = Wf

        for i in range(self.num_layers):
            # self attention within the features (B, Y*X, C)
            bev_queries = self.self_attn_layers[i](bev_queries, self.Y, self.X, bev_queries_pos)

            # normalize (B, Y*X, C)
            bev_queries = self.norm1_layers[i](bev_queries)

            # cross attention into the images
            bev_queries = self.cross_attn_layers[i](bev_queries, bev_keys, bev_keys,
                                                    query_pos=bev_queries_pos,
                                                    reference_points_cam=reference_points_cam,
                                                    spatial_shapes=spatial_shapes,
                                                    bev_mask=bev_mask)

            # normalize (B, N, C)
            bev_queries = self.norm2_layers[i](bev_queries)

            # feedforward layer (B, N, C)
            bev_queries = bev_queries + self.ffn_layers[i](bev_queries)

            # normalize (B, N, C)
            bev_queries = self.norm3_layers[i](bev_queries)

        feat_bev = bev_queries.permute(0, 2, 1).reshape(B, self.latent_dim, self.Y, self.X)

        if prev_bev is None:
            prev_bev = feat_bev
        feat_bev = torch.cat([feat_bev, prev_bev], dim=1)
        feat_bev = self.bev_temporal(feat_bev)

        # bev decoder
        out_dict = self.decoder(feat_bev, feat_cams_)

        return out_dict
