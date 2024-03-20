import numpy as np
import torch
import torch.nn as nn

import utils.basic
import utils.geom
import utils.vox
from models.decoder import Decoder
from models.encoder import Encoder_res101, Encoder_res50, Encoder_eff, Encoder_swin_t, Encoder_res18


class Liftnet(nn.Module):
    def __init__(self, Y, Z, X,
                 DMAX,
                 D,
                 DMIN=2.0,
                 num_classes=None,
                 num_cameras=None,
                 do_rgbcompress=True,
                 rand_flip=False,
                 latent_dim=256,
                 feat2d_dim=96,
                 encoder_type='swin_t',
                 z_sign=1,
                 ):
        super(Liftnet, self).__init__()
        assert (encoder_type in ['res101', 'res50', 'res18', 'effb0', 'effb4', 'swin_t'])

        self.Y, self.Z, self.X = Y, Z, X
        self.DMAX, self.DMIN = DMAX, DMIN
        self.D = D
        self.do_rgbcompress = do_rgbcompress
        self.rand_flip = rand_flip
        self.latent_dim = latent_dim
        self.encoder_type = encoder_type
        self.num_cameras = num_cameras
        self.z_sign = z_sign

        self.mean = torch.as_tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1).float().cuda()
        self.std = torch.as_tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1).float().cuda()

        # Encoder
        self.feat2d_dim = feat2d_dim
        if encoder_type == 'res101':
            self.encoder = Encoder_res101(feat2d_dim + self.D)
        elif encoder_type == 'res50':
            self.encoder = Encoder_res50(feat2d_dim + self.D)
        elif encoder_type == 'res18':
            self.encoder = Encoder_res18(feat2d_dim + self.D)
        elif encoder_type == 'effb0':
            self.encoder = Encoder_eff(feat2d_dim + self.D, version='b0')
        elif encoder_type == 'swin_t':
            self.encoder = Encoder_swin_t(feat2d_dim + self.D)
        else:
            # effb4
            self.encoder = Encoder_eff(feat2d_dim + self.D, version='b4')

        # BEV compressor
        if self.num_cameras is not None:
            self.cam_compressor = nn.Sequential(
                nn.Conv3d(feat2d_dim * self.num_cameras, feat2d_dim, kernel_size=3, padding=1, stride=1),
                nn.InstanceNorm3d(feat2d_dim), nn.ReLU(),
                nn.Conv3d(feat2d_dim, feat2d_dim, kernel_size=1),
            )

        self.bev_compressor = nn.Sequential(
            nn.Conv2d(self.feat2d_dim * self.Z, latent_dim, kernel_size=3, padding=1),
            nn.InstanceNorm2d(latent_dim), nn.ReLU(),
            nn.Conv2d(latent_dim, latent_dim, kernel_size=1),
        )

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

    def create_frustum(self):
        # make grid in image plane
        ogfH, ogfW = self.data_aug_conf['final_dim']
        fH, fW = ogfH // self.downsample, ogfW // self.downsample
        ds = torch.arange(*self.grid_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = ds.shape
        xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)

        # D x H x W x 3
        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, rots, trans, intrins, post_rots, post_trans):
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """
        B, N, _ = trans.shape

        # undo post-transformation
        # B x N x D x H x W x 3
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))

        # cam_to_ego
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                            points[:, :, :, :, :, 2:3]
                            ), 5)
        combine = rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)

        return points

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
        assert (C == 3)
        # reshape tensors
        __p = lambda x: utils.basic.pack_seqdim(x, B)
        __u = lambda x: utils.basic.unpack_seqdim(x, B)
        rgb_cams_ = __p(rgb_cams)
        pix_T_cams_ = __p(pix_T_cams)
        cams_T_global_ = __p(cams_T_global)  # B*S,4,4
        global_T_cams_ = torch.inverse(cams_T_global_)  # B*S,4,4
        ref_T_cams = torch.matmul(ref_T_global.repeat(S, 1, 1), global_T_cams_)  # B*S,4,4
        cams_T_ref_ = torch.inverse(ref_T_cams)  # B*S,4,4

        # rgb encoder
        device = rgb_cams_.device
        rgb_cams_ = (rgb_cams_ - self.mean.to(device)) / self.std.to(device)
        if self.rand_flip:
            B0, _, _, _ = rgb_cams_.shape
            self.rgb_flip_index = np.random.choice([0, 1], B0).astype(bool)
            rgb_cams_[self.rgb_flip_index] = torch.flip(rgb_cams_[self.rgb_flip_index], [-1])
        feat_cams_ = self.encoder(rgb_cams_)  # S,(D+C),H/8,W/8
        if self.rand_flip:
            feat_cams_[self.rgb_flip_index] = torch.flip(feat_cams_[self.rgb_flip_index], [-1])
        _, CD, Hf, Wf = feat_cams_.shape
        # assert (CD == (self.D + self.feat2d_dim))

        sy = Hf / float(H)
        sx = Wf / float(W)
        Y, Z, X = self.Y, self.Z, self.X
        featpix_T_cams_ = utils.geom.scale_intrinsics(pix_T_cams_, sx, sy)

        depth_cams_out = feat_cams_[:, :self.D].unsqueeze(1)  # BS,1,D(discrete depth),Hf,Wf
        feat_cams_ = feat_cams_[:, self.D:].unsqueeze(2)  # BS,C(latent dim),1,Hf,Wf
        # depth_camXs_ = (depth_camXs_out / 0.07)
        depth_cams_ = depth_cams_out.softmax(dim=2)  # BS,1,D(depth distribution),Hf,Wf
        feat_tileXs_ = feat_cams_ * depth_cams_  # BS,C,D,Hf,Wf

        # unproject image feature to 3d grid
        feat_mems_ = vox_util.warp_tiled_to_mem(
            feat_tileXs_,
            utils.basic.matmul2(featpix_T_cams_, cams_T_ref_),  # featpix_T_ref
            cams_T_ref_, Y, Z, X, self.DMIN, self.DMAX+self.DMIN, z_sign=self.z_sign)
        feat_mems = __u(feat_mems_)

        if self.num_cameras is None:
            one_mems_ = vox_util.warp_tiled_to_mem(
                torch.ones_like(feat_tileXs_),
                utils.basic.matmul2(featpix_T_cams_, cams_T_ref_),
                cams_T_ref_, Y, Z, X, self.DMIN, self.DMAX+self.DMIN, z_sign=self.z_sign)
            one_mems = __u(one_mems_)
            one_mems = one_mems.clamp(min=1.0)

            feat_mem = utils.basic.reduce_masked_mean(feat_mems, one_mems, dim=1)  # B, C, Y, Z, X
        else:
            feat_mem = self.cam_compressor(feat_mems.flatten(1, 2))

        if self.rand_flip:
            self.bev_flip1_index = np.random.choice([0, 1], B).astype(bool)
            self.bev_flip2_index = np.random.choice([0, 1], B).astype(bool)
            feat_mem[self.bev_flip1_index] = torch.flip(feat_mem[self.bev_flip1_index], [-1])
            feat_mem[self.bev_flip2_index] = torch.flip(feat_mem[self.bev_flip2_index], [-3])

        # bev compressing
        bev_features = feat_mem.permute(0, 1, 3, 2, 4).flatten(1, 2)
        bev_features = self.bev_compressor(bev_features)

        if prev_bev is None:
            prev_bev = bev_features
        bev_features = torch.cat([bev_features, prev_bev], dim=1)
        bev_features = self.bev_temporal(bev_features)

        # bev decoder
        out_dict = self.decoder(bev_features, feat_cams_.squeeze(2),
                                (self.bev_flip1_index, self.bev_flip2_index) if self.rand_flip else None)
        out_dict['depth'] = depth_cams_out

        return out_dict
