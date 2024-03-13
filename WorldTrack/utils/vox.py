import numpy as np
import torch
import torch.nn.functional as F

import utils.geom
import utils.basic


class VoxelUtil:
    def __init__(self, Y, Z, X, scene_centroid, bounds, pad=None, assert_cube=False):
        self.XMIN, self.XMAX, self.YMIN, self.YMAX, self.ZMIN, self.ZMAX = bounds
        self.Y, self.Z, self.X = Y, Z, X

        x_centroid, y_centroid, z_centroid = scene_centroid[0]
        self.XMIN += x_centroid
        self.XMAX += x_centroid
        self.YMIN += y_centroid
        self.YMAX += y_centroid
        self.ZMIN += z_centroid
        self.ZMAX += z_centroid

        self.default_vox_size_X = (self.XMAX - self.XMIN) / float(X)
        self.default_vox_size_Y = (self.YMAX - self.YMIN) / float(Y)
        self.default_vox_size_Z = (self.ZMAX - self.ZMIN) / float(Z)

        if pad:
            Y_pad, Z_pad, X_pad = pad
            self.ZMIN -= self.default_vox_size_Z * Z_pad
            self.ZMAX += self.default_vox_size_Z * Z_pad
            self.YMIN -= self.default_vox_size_Y * Y_pad
            self.YMAX += self.default_vox_size_Y * Y_pad
            self.XMIN -= self.default_vox_size_X * X_pad
            self.XMAX += self.default_vox_size_X * X_pad

        if assert_cube:
            # we assume cube voxels
            if (not np.isclose(self.default_vox_size_X, self.default_vox_size_Z)) or (
                    not np.isclose(self.default_vox_size_X, self.default_vox_size_Y)):
                print('Y, Z, X', Y, Z, X)
                print('bounds for this iter:',
                      'X = %.2f to %.2f' % (self.XMIN, self.XMAX),
                      'Y = %.2f to %.2f' % (self.YMIN, self.YMAX),
                      'Z = %.2f to %.2f' % (self.ZMIN, self.ZMAX),
                      )
                print('self.default_vox_size_X', self.default_vox_size_X)
                print('self.default_vox_size_Y', self.default_vox_size_Y)
                print('self.default_vox_size_Z', self.default_vox_size_Z)
            assert (np.isclose(self.default_vox_size_X, self.default_vox_size_Z))
            assert (np.isclose(self.default_vox_size_X, self.default_vox_size_Y))

    def Ref2Mem(self, xyz, Y, Z, X, assert_cube=False):
        # xyz is B x N x 3, in ref coordinates
        # transforms ref coordinates into mem coordinates
        B, N, C = list(xyz.shape)
        device = xyz.device
        assert (C == 3)
        mem_T_ref = self.get_mem_T_ref(B, Y, Z, X, assert_cube=assert_cube, device=device)
        xyz = utils.geom.apply_4x4(mem_T_ref, xyz)
        return xyz

    def Mem2Ref(self, xyz_mem, Y, Z, X, assert_cube=False):
        # xyz is B x N x 3, in mem coordinates
        # transforms mem coordinates into ref coordinates
        B, N, C = list(xyz_mem.shape)
        ref_T_mem = self.get_ref_T_mem(B, Y, Z, X, assert_cube=assert_cube, device=xyz_mem.device)
        xyz_ref = utils.geom.apply_4x4(ref_T_mem, xyz_mem)
        return xyz_ref

    def get_mem_T_ref(self, B, Y, Z, X, assert_cube=False, device='cuda'):
        vox_size_X = (self.XMAX - self.XMIN) / float(X)
        vox_size_Y = (self.YMAX - self.YMIN) / float(Y)
        vox_size_Z = (self.ZMAX - self.ZMIN) / float(Z)

        if assert_cube:
            if (not np.isclose(vox_size_X, vox_size_Y)) or (not np.isclose(vox_size_X, vox_size_Z)):
                print('Z, Y, X', Z, Y, X)
                print('bounds for this iter:',
                      'X = %.2f to %.2f' % (self.XMIN, self.XMAX),
                      'Y = %.2f to %.2f' % (self.YMIN, self.YMAX),
                      'Z = %.2f to %.2f' % (self.ZMIN, self.ZMAX),
                      )
                print('vox_size_X', vox_size_X)
                print('vox_size_Y', vox_size_Y)
                print('vox_size_Z', vox_size_Z)
            assert (np.isclose(vox_size_X, vox_size_Y))
            assert (np.isclose(vox_size_X, vox_size_Z))

        # translation
        # (this makes the left edge of the leftmost voxel correspond to XMIN)
        center_T_ref = np.eye(4)
        center_T_ref[0, 3] = -self.XMIN - vox_size_X / 2.0
        center_T_ref[1, 3] = -self.YMIN - vox_size_Y / 2.0
        center_T_ref[2, 3] = -self.ZMIN - vox_size_Z / 2.0
        # center_T_ref[0, 3] = -self.XMIN
        # center_T_ref[1, 3] = -self.YMIN
        # center_T_ref[2, 3] = -self.ZMIN
        center_T_ref = torch.tensor(center_T_ref, device=device, dtype=vox_size_X.dtype).view(1, 4, 4).repeat([B, 1, 1])

        # scaling
        # (this makes the right edge of the rightmost voxel correspond to XMAX)
        mem_T_center = np.eye(4)
        mem_T_center[0, 0] = 1. / vox_size_X
        mem_T_center[1, 1] = 1. / vox_size_Y
        mem_T_center[2, 2] = 1. / vox_size_Z
        mem_T_center = torch.tensor(mem_T_center, device=device, dtype=vox_size_X.dtype).view(1, 4, 4).repeat([B, 1, 1])
        mem_T_ref = utils.basic.matmul2(mem_T_center, center_T_ref)

        return mem_T_ref

    def get_ref_T_mem(self, B, Y, Z, X, assert_cube=False, device='cuda'):
        mem_T_ref = self.get_mem_T_ref(B, Y, Z, X, assert_cube=assert_cube, device=device)
        # note safe_inverse is inapplicable here,
        # since the transform is nonrigid
        ref_T_mem = mem_T_ref.inverse()
        return ref_T_mem

    def get_inbounds(self, xyz, Y, Z, X, already_mem=False, padding=0.0, assert_cube=False):
        # xyz is B x N x 3
        # padding should be 0 unless you are trying to account for some later cropping
        if not already_mem:
            xyz = self.Ref2Mem(xyz, Y, Z, X, assert_cube=assert_cube)

        x = xyz[:, :, 0]
        y = xyz[:, :, 1]
        z = xyz[:, :, 2]

        x_valid = ((x - padding) > -0.5).byte() & ((x + padding) < float(X - 0.5)).byte()
        y_valid = ((y - padding) > -0.5).byte() & ((y + padding) < float(Y - 0.5)).byte()
        z_valid = ((z - padding) > -0.5).byte() & ((z + padding) < float(Z - 0.5)).byte()
        nonzero = (~(z == 0.0)).byte()

        inbounds = x_valid & y_valid & z_valid & nonzero
        return inbounds.bool()

    def voxelize_xyz(self, xyz_ref, Y, Z, X, already_mem=False, assert_cube=False, clean_eps=0):
        B, N, D = list(xyz_ref.shape)
        assert (D == 3)
        if already_mem:
            xyz_mem = xyz_ref
        else:
            xyz_mem = self.Ref2Mem(xyz_ref, Y, Z, X, assert_cube=assert_cube)
            xyz_zero = self.Ref2Mem(xyz_ref[:, 0:1] * 0, Y, Z, X, assert_cube=assert_cube)
        vox = self.get_occupancy(xyz_mem, Y, Z, X, clean_eps=clean_eps, xyz_zero=xyz_zero)
        return vox

    def voxelize_xyz_and_feats(self, xyz_ref, feats, Y, Z, X, already_mem=False, assert_cube=False, clean_eps=0):
        B, N, D = list(xyz_ref.shape)
        B2, N2, D2 = list(feats.shape)
        assert (D == 3)
        assert (B == B2)
        assert (N == N2)
        if already_mem:
            xyz_mem = xyz_ref
        else:
            xyz_mem = self.Ref2Mem(xyz_ref, Z, Y, X, assert_cube=assert_cube)
            xyz_zero = self.Ref2Mem(xyz_ref[:, 0:1] * 0, Y, Z, X, assert_cube=assert_cube)
        feats = self.get_feat_occupancy(xyz_mem, feats, Y, Z, X, clean_eps=clean_eps, xyz_zero=xyz_zero)
        return feats

    def get_occupancy(self, xyz, Y, Z, X, clean_eps=0, xyz_zero=None):
        # xyz is B x N x 3 and in mem coords
        # we want to fill a voxel tensor with 1's at these inds
        B, N, C = list(xyz.shape)
        assert (C == 3)

        # these papers say simple 1/0 occupancy is ok:
        #  http://openaccess.thecvf.com/content_cvpr_2018/papers/Yang_PIXOR_Real-Time_3d_CVPR_2018_paper.pdf
        #  http://openaccess.thecvf.com/content_cvpr_2018/papers/Luo_Fast_and_Furious_CVPR_2018_paper.pdf
        # cont fusion says they do 8-neighbor interp
        # voxelnet does occupancy but with a bit of randomness in terms of the reflectance value i think

        inbounds = self.get_inbounds(xyz, Y, Z, X, already_mem=True)
        x, y, z = xyz[:, :, 0], xyz[:, :, 1], xyz[:, :, 2]
        mask = torch.zeros_like(x)
        mask[inbounds] = 1.0

        if xyz_zero is not None:
            # only take points that are beyond a thresh of zero
            dist = torch.norm(xyz_zero - xyz, dim=2)
            mask[dist < 0.1] = 0

        if clean_eps > 0:
            # only take points that are already near centers
            xyz_round = torch.round(xyz)  # B, N, 3
            dist = torch.norm(xyz_round - xyz, dim=2)
            mask[dist > clean_eps] = 0

        # set the invalid guys to zero
        # we then need to zero out 0,0,0
        # (this method seems a bit clumsy)
        x = x * mask
        y = y * mask
        z = z * mask

        x = torch.round(x)
        y = torch.round(y)
        z = torch.round(z)
        x = torch.clamp(x, 0, X - 1).int()
        y = torch.clamp(y, 0, Y - 1).int()
        z = torch.clamp(z, 0, Z - 1).int()

        x = x.view(B * N)
        y = y.view(B * N)
        z = z.view(B * N)

        dim3 = X
        dim2 = X * Y
        dim1 = X * Y * Z

        base = torch.arange(0, B, dtype=torch.int32, device=xyz.device) * dim1
        base = torch.reshape(base, [B, 1]).repeat([1, N]).view(B * N)

        vox_inds = base + z * dim2 + y * dim3 + x
        voxels = torch.zeros(B * Z * Y * X, device=xyz.device).float()
        voxels[vox_inds.long()] = 1.0
        # zero out the singularity
        voxels[base.long()] = 0.0
        voxels = voxels.reshape(B, 1, Y, Z, X)
        # B x 1 x Z x Y x X
        return voxels

    def get_feat_occupancy(self, xyz, feat, Y, Z, X, clean_eps=0, xyz_zero=None):
        # xyz is B x N x 3 and in mem coords
        # feat is B x N x D
        # we want to fill a voxel tensor with 1's at these inds
        B, N, C = list(xyz.shape)
        B2, N2, D2 = list(feat.shape)
        assert (C == 3)
        assert (B == B2)
        assert (N == N2)

        # these papers say simple 1/0 occupancy is ok:
        #  http://openaccess.thecvf.com/content_cvpr_2018/papers/Yang_PIXOR_Real-Time_3d_CVPR_2018_paper.pdf
        #  http://openaccess.thecvf.com/content_cvpr_2018/papers/Luo_Fast_and_Furious_CVPR_2018_paper.pdf
        # cont fusion says they do 8-neighbor interp
        # voxelnet does occupancy but with a bit of randomness in terms of the reflectance value i think

        inbounds = self.get_inbounds(xyz, Y, Z, X, already_mem=True)
        x, y, z = xyz[:, :, 0], xyz[:, :, 1], xyz[:, :, 2]
        mask = torch.zeros_like(x)
        mask[inbounds] = 1.0

        if xyz_zero is not None:
            # only take points that are beyond a thresh of zero
            dist = torch.norm(xyz_zero - xyz, dim=2)
            mask[dist < 0.1] = 0

        if clean_eps > 0:
            # only take points that are already near centers
            xyz_round = torch.round(xyz)  # B, N, 3
            dist = torch.norm(xyz_round - xyz, dim=2)
            mask[dist > clean_eps] = 0

        # set the invalid guys to zero
        # we then need to zero out 0,0,0
        # (this method seems a bit clumsy)
        x = x * mask  # B, N
        y = y * mask
        z = z * mask
        feat = feat * mask.unsqueeze(-1)  # B, N, D

        x = torch.round(x)
        y = torch.round(y)
        z = torch.round(z)
        x = torch.clamp(x, 0, X - 1).int()
        y = torch.clamp(y, 0, Y - 1).int()
        z = torch.clamp(z, 0, Z - 1).int()

        # permute point orders
        perm = torch.randperm(N)
        x = x[:, perm]
        y = y[:, perm]
        z = z[:, perm]
        feat = feat[:, perm]

        x = x.view(B * N)
        y = y.view(B * N)
        z = z.view(B * N)
        feat = feat.view(B * N, -1)

        dim3 = X
        dim2 = X * Y
        dim1 = X * Y * Z

        base = torch.arange(0, B, dtype=torch.int32, device=xyz.device) * dim1
        base = torch.reshape(base, [B, 1]).repeat([1, N]).view(B * N)

        vox_inds = base + z * dim2 + y * dim3 + x
        feat_voxels = torch.zeros((B * Z * Y * X, D2), device=xyz.device).float()
        feat_voxels[vox_inds.long()] = feat
        # zero out the singularity
        feat_voxels[base.long()] = 0.0
        feat_voxels = feat_voxels.reshape(B, Y, Z, X, D2).permute(0, 4, 1, 2, 3)
        # B x C x Z x Y x X
        return feat_voxels

    def unproject_image_to_mem(self, rgb_camB, pixB_T_refA, camB_T_refA, Y, Z, X, assert_cube=False, xyz_refA=None,
                               z_sign=1, mode='bilinear'):
        """
        rgb_camB: B*S x 128 x H x W
        pixB_T_refA: B*S x 4 x 4
        camB_T_refA: B*S x 4 x 4
        rgb lives in B pixel coords we want everything in A ref coords
        this puts each C-dim pixel in the rgb_camB along a ray in the voxel grid
        """
        B, C, H, W = rgb_camB.shape

        if xyz_refA is None:
            xyz_memA = utils.basic.gridcloud3d(B, Y, Z, X, norm=False, device=pixB_T_refA.device)
            xyz_refA = self.Mem2Ref(xyz_memA, Y, Z, X, assert_cube=assert_cube)
        xyz_camB = utils.geom.apply_4x4(camB_T_refA, xyz_refA)
        xyz_pixB = utils.geom.apply_4x4(pixB_T_refA, xyz_refA)
        normalizer = torch.unsqueeze(xyz_pixB[:, :, 2], 2)
        EPS = 1e-6
        normalizer[normalizer <= 0] = normalizer[normalizer <= 0].clamp(max=-EPS)
        normalizer[normalizer >= 0] = normalizer[normalizer >= 0].clamp(min=EPS)
        z = xyz_camB[:, :, 2]
        xy_pixB = xyz_pixB[:, :, :2] / normalizer  # B,N,2

        # this is the (floating point) pixel coordinate of each voxel
        x, y = xy_pixB[:, :, 0], xy_pixB[:, :, 1]  # B,N

        x_valid = (x >= 0).bool() & (x / float(W) <= 1).bool()
        y_valid = (y >= 0).bool() & (y / float(H) <= 1).bool()
        z_valid = (z_sign * z >= 0).bool()
        valid_mem = (x_valid & y_valid & z_valid).reshape(B, 1, Y, Z, X).float()

        # native pytorch version
        y_pixB, x_pixB = utils.basic.normalize_grid2d(y, x, H, W)
        # since we want a 3d output, we need 5d tensors
        z_pixB = torch.zeros_like(x)
        xyz_pixB = torch.stack([x_pixB, y_pixB, z_pixB], axis=2)
        rgb_camB = rgb_camB.unsqueeze(2)  # B*S,128,1(D),H,W
        xyz_pixB = torch.reshape(xyz_pixB, [B, Y, Z, X, 3])  # B*S,200,8,200,3
        values = F.grid_sample(rgb_camB, xyz_pixB, align_corners=False, mode=mode)

        values = torch.reshape(values, (B, C, Y, Z, X))
        values = values * valid_mem
        return values

    def warp_tiled_to_mem(self, rgb_tileB, pixB_T_ref, camB_T_ref, Y, Z, X, DMIN, DMAX, assert_cube=False, z_sign=1):
        """
        B = batch size, S = number of cameras, C = latent dim, D = depth, H = img height, W = img width
        rgb_tileB: B*S,C,D,H/8,W/8
        pixB_T_ref: B*S,4,4
        camB_T_ref: B*S,4,4

        rgb_tileB lives in B pixel coords but it has been tiled across the Z dimension
        we want everything in A memory coords

        this resamples the so that each C-dim pixel in rgb_tilB
        is put into its correct place in the voxel grid

        mapping [0,D-1] pixel-level depth distribution to [DMIN,DMAX] in real world
        """

        B, C, D, H, W = list(rgb_tileB.shape)

        xyz_memA = utils.basic.gridcloud3d(B, Y, Z, X, norm=False, device=pixB_T_ref.device)

        xyz_ref = self.Mem2Ref(xyz_memA, Y, Z, X, assert_cube=assert_cube)

        xyz_camB = utils.geom.apply_4x4(camB_T_ref, xyz_ref)
        z_camB = xyz_camB[:, :, 2]

        # rgb_tileB has depth=DMIN in tile 0, and depth=DMAX in tile D-1
        z_tileB = (D - 1.0) * (z_camB - float(DMIN)) / float(DMAX - DMIN)

        xyz_pixB = utils.geom.apply_4x4(pixB_T_ref, xyz_ref)
        normalizer = torch.unsqueeze(xyz_pixB[:, :, 2], 2)
        EPS = 1e-4
        normalizer[normalizer <= 0] = normalizer[normalizer <= 0].clamp(max=-EPS)
        normalizer[normalizer >= 0] = normalizer[normalizer >= 0].clamp(min=EPS)
        xy_pixB = xyz_pixB[:, :, :2] / normalizer  # B,N,2

        # this is the (floating point) pixel coordinate of each voxel
        x, y = xy_pixB[:, :, 0], xy_pixB[:, :, 1]  # B,N

        x_valid = (x >= 0).bool() & (x / float(W) <= 1).bool()
        y_valid = (y >= 0).bool() & (y / float(H) <= 1).bool()
        z_valid = (z_sign * z_camB >= 0).bool()
        valid_mem = (x_valid & y_valid & z_valid).reshape(B, 1, Y, Z, X).float()

        z_tileB, y_pixB, x_pixB = utils.basic.normalize_grid3d(z_tileB, y, x, D, H, W)
        xyz_pixB = torch.stack([x_pixB, y_pixB, z_tileB], axis=2)
        xyz_pixB = torch.reshape(xyz_pixB, [B, Y, Z, X, 3])
        values = F.grid_sample(rgb_tileB, xyz_pixB, align_corners=False)

        values = torch.reshape(values, (B, C, Y, Z, X))
        values = values * valid_mem
        return values
