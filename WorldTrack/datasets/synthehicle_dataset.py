import glob
import json
import os.path as osp
from typing import Any
import numpy as np
import torch
import random
from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.transforms import functional as F

from utils import geom, vox, basic


class SynthehicleDataset(VisionDataset):
    def __init__(
            self,
            root: str,
            split: str = 'train',
            resolution: tuple = (200, 1, 200),  # Y Z X
            bounds: tuple = (-75, 75, -75, 75, -1, 5),  # X Y Z
            num_train_cams: int = 5,
            max_objects: int = 60,
            final_dim: tuple = (432, 768),
            resize_lim: list = (0.8, 1.2),
            is_train: bool = False,
    ) -> None:
        super().__init__(root)
        self.root = root
        self.split = split
        self.is_train = is_train or split == 'train'
        self.resolution = resolution
        self.bounds = bounds
        self.final_dim = final_dim
        self.data_aug_conf = {'final_dim': final_dim, 'resize_lim': resize_lim}
        self.max_objects = max_objects
        self.num_train_cams = num_train_cams
        self.img_downsample = 4
        self.kernel_size = 1.5
        self.num_classes = 3

        self.Y, self.Z, self.X = self.resolution
        self.D = 100
        self.scene_centroid = torch.tensor((0., 0., 0.)).reshape([1, 3])
        self.vox_util = vox.VoxelUtil(
            self.Y, self.Z, self.X,
            scene_centroid=self.scene_centroid,
            bounds=self.bounds,
            assert_cube=False)

        self.samples = []
        self.cameras = {}
        self.scene = {}
        self.setup()

    def setup(self):
        if self.split != 'test':
            town_paths = glob.glob(osp.join(self.root, 'train', 'Town*-O-*'))
        else:
            town_paths = glob.glob(osp.join(self.root, self.split, 'Town*-O-*'))

        for town_num, town_path in enumerate(sorted(town_paths)):
            town_name = osp.basename(town_path).split('-')[0]
            camera_file = osp.join(town_path, 'camera_name.txt')
            camera_names: list = np.loadtxt(camera_file, dtype=str).tolist()

            # Add samples
            frames = glob.glob(osp.join(town_path, camera_names[0], 'out_rgb', '*.jpg'))
            samples = [(town_num, town_name, town_path, osp.basename(f)[:-4]) for f in sorted(frames)]
            if self.split == 'train':
                samples = samples[:int(len(samples)*0.7)]
            elif self.split == 'val':
                samples = samples[int(len(samples)*0.7):]
            self.samples.extend(samples)

            # Fetch camera calibration
            if town_name in self.cameras:
                continue

            self.cameras[town_name] = {}
            for index, camera_name in enumerate(camera_names):
                cam_id = int(camera_name[1:])
                calibration_path = osp.join(self.root, town_name, 'camera_info', f'camera_{cam_id}.txt')
                with open(calibration_path) as f:
                    calibration_dict = json.load(f)

                intrinsic = torch.tensor(calibration_dict['intrinsic_matrix'])  # 3,3
                intrinsic = geom.merge_intrinsics(*geom.split_intrinsics(intrinsic.unsqueeze(0))).squeeze()  # 4,4
                extrinsic = torch.tensor(calibration_dict['extrinsic_matrix'])  # w2c
                t = extrinsic[:3, 3:]  # 3,1
                r = extrinsic[:3, :3]  # 3,3
                # Convert UE's coordinate system 'back right up' to 'right up back'
                # ^ z                     ^ y
                # | . x          to:      | . z
                # |/                      |/
                # +----> y                +-----> x
                t = torch.tensor([t[1], t[2], t[0]])
                change = torch.tensor([
                    [0, 1, 0],
                    [0, 0, 1],
                    [1, 0, 0]
                ])
                r = torch.tensor(np.dot(change, r))

                self.cameras[town_name][camera_name] = {
                    'id': index,
                    'name': camera_name,
                    'rot': r,
                    'tran': t,
                    'intrin': intrinsic
                }

            # Find scene center
            rots = torch.stack([c['rot'] for c in self.cameras[town_name].values()])
            trans = torch.stack([c['tran'] for c in self.cameras[town_name].values()])
            cams_T_global = geom.merge_rt(rots, trans.squeeze(-1))  # S 4 4
            global_T_cams = torch.inverse(cams_T_global)  # S 4 4
            cx, cy = global_T_cams[:, :2, 3].mean(dim=0)
            self.scene[town_name] = {'scene_center': torch.tensor([cx, cy, 0.0])}

    def get_image_data(self, index, cameras, bbox_cam):
        imgs, intrins, rots, trans = [], [], [], []
        centers, offsets, sizes, pids, valids = [], [], [], [], []
        _, town_name, town_path, frame = self.samples[index]
        for camera in cameras.values():
            img_path = osp.join(town_path, camera['name'], 'out_rgb', f'{frame}.jpg')
            img = Image.open(img_path)
            img = img.transpose(Image.FLIP_TOP_BOTTOM)

            W, H = img.size  # 1920, 1080

            resize_dims, crop = self.sample_augmentation()
            sx = resize_dims[0] / float(W)
            sy = resize_dims[1] / float(H)

            intrin = torch.Tensor(camera['intrin'])
            intrin = geom.scale_intrinsics(intrin.unsqueeze(0), sx, sy).squeeze(0)

            fx, fy, x0, y0 = geom.split_intrinsics(intrin.unsqueeze(0))

            new_x0 = x0 - crop[0]
            new_y0 = y0 - crop[1]

            pix_T_cam = geom.merge_intrinsics(fx, fy, new_x0, new_y0)
            intrin = pix_T_cam.squeeze(0)  # 4,4
            img = basic.img_transform(img, resize_dims, crop)
            imgs.append(F.to_tensor(img))

            rot = camera['rot']
            tran = camera['tran']
            intrins.append(intrin)
            rots.append(rot)
            trans.append(tran)

            bbox = bbox_cam[camera['name']]
            center_img, offset_img, size_img, pid_img, valid_img = self.get_img_gt(bbox, sx, sy, crop)

            centers.append(center_img)
            offsets.append(offset_img)
            sizes.append(size_img)
            pids.append(pid_img)
            valids.append(valid_img)

        return (torch.stack(imgs), torch.stack(rots), torch.stack(trans), torch.stack(intrins), torch.stack(centers),
                torch.stack(offsets), torch.stack(sizes), torch.stack(pids), torch.stack(valids))

    def sample_augmentation(self):
        fH, fW = self.data_aug_conf['final_dim']
        if self.is_train:
            resize = np.random.uniform(*self.data_aug_conf['resize_lim'])
            resize_dims = (int(fW * resize), int(fH * resize))
            newW, newH = resize_dims

            # center it
            crop_h = int((newH - fH) / 2)
            crop_w = int((newW - fW) / 2)

            crop_offset = 0.5 * int(self.data_aug_conf['resize_lim'][0] * self.data_aug_conf['final_dim'][0])
            crop_w = crop_w + int(np.random.uniform(-crop_offset, crop_offset))
            crop_h = crop_h + int(np.random.uniform(-crop_offset, crop_offset))

            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        else:  # validation/test
            # do a perfect resize
            resize_dims = (fW, fH)
            crop_h = 0
            crop_w = 0
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        return resize_dims, crop

    def get_bbox_global(self, index):
        _, town_name, town_path, frame = self.samples[index]
        cameras = self.cameras[town_name]
        labels = []
        bbox_global = []
        bbox_ids = set()
        bbox_cam = {}
        for _, cam in cameras.items():
            bbox_path = osp.join(town_path, cam['name'], 'out_bbox', f'{frame}.txt')
            with open(bbox_path) as f:
                raw_data = json.load(f)
            bbox_cam[cam['name']] = raw_data
            for v_id, v_class, box in zip(raw_data['vehicle_id'], raw_data['vehicle_class'], raw_data['world_coords']):
                if v_class not in (0, 1, 2):  # only passenger_cars, trucks, and motorbikes
                    continue
                if v_id in bbox_ids:
                    continue
                labels.append({
                    'track_id': v_id,
                    'class': v_class,
                    'box': box,
                })
                bbox_ids.add(v_id)
                bbox_global.append(torch.tensor(box))

        if bbox_global:
            bbox_global = torch.stack(bbox_global)
        else:
            bbox_global = torch.zeros((0, 4, 8))

        return bbox_global, labels, bbox_cam, town_path

    def get_lwh_and_yaw(self, corners):
        """
        corners: N,8,3
        """
        z_max = torch.mean(torch.topk(corners[:, :, 2], 4, dim=1, largest=True).values, dim=1)
        z_min = torch.mean(torch.topk(corners[:, :, 2], 4, dim=1, largest=False).values, dim=1)
        height = z_max - z_min  # (N,)
        corners = corners[:, :4, :2]  # N,4,2
        x1, y1 = corners[:, 0, 0], corners[:, 0, 1]  # (N,)
        x2, y2 = corners[:, 1, 0], corners[:, 1, 1]
        x3, y3 = corners[:, 2, 0], corners[:, 2, 1]
        dx1, dy1 = x2 - x1, y2 - y1
        dx2, dy2 = x3 - x2, y3 - y2

        length1 = torch.sqrt(torch.square(dx1) + torch.square(dy1))  # (N,)
        length2 = torch.sqrt(torch.square(dx2) + torch.square(dy2))

        length = torch.max(length1, length2)
        width = torch.min(length1, length2)
        lwh = torch.stack((length, width, height)).transpose(0, 1)

        yaw = dy1 / (dx1 + 1e-6)
        yaw[length1 <= length2] = (dy2 / (dx2 + 1e-6))[length1 <= length2]

        # yaw = torch.atan2(dy1, dx1)
        yaw = torch.arctan(yaw)
        return lwh.unsqueeze(0), yaw.unsqueeze(0)

    def get_bev_get(self, corners_ref, labels, corners_prev, labels_prev):
        """
        corners_ref: 1,N,8,3
        seg_bev: 1,Y,X
        """
        clist_ref = torch.mean(corners_ref, dim=2, keepdim=False)  # 1,N,3
        mem_pts = self.vox_util.Ref2Mem(clist_ref, self.Y, self.Z, self.X).squeeze(0)

        clist_prev = torch.mean(corners_prev, dim=2, keepdim=False)  # 1,N,3
        mem_pts_prev = self.vox_util.Ref2Mem(clist_prev, self.Y, self.Z, self.X).squeeze(0)
        prev_loc = {}
        for pt, label in zip(mem_pts_prev, labels_prev):
            prev_loc[label['track_id']] = pt

        # get size and yaw of each bbox
        lwhs, yaws = self.get_lwh_and_yaw(corners_ref[0])  # 1,N

        center = torch.zeros((self.num_classes, self.Y, self.X), dtype=torch.float32)
        valid_mask = torch.zeros((1, self.Y, self.X), dtype=torch.bool)
        offset = torch.zeros((4, self.Y, self.X), dtype=torch.float32)
        ids = torch.zeros((1, self.Y, self.X), dtype=torch.long)
        size = torch.zeros((3, self.Y, self.X), dtype=torch.float32)
        rotbin = torch.zeros((2, self.Y, self.X), dtype=torch.long)
        rotres = torch.zeros((2, self.Y, self.X), dtype=torch.float32)

        for center_pt, lwh, yaw, label in zip(mem_pts, lwhs[0], yaws[0], labels):
            ct_int = center_pt.int()
            if ct_int[1] >= self.Y or ct_int[0] >= self.X or ct_int[1] < 0 or ct_int[0] < 0:
                continue
            basic.draw_umich_gaussian(center[label['class']], ct_int, self.kernel_size)
            valid_mask[:, ct_int[1], ct_int[0]] = 1
            offset[:2, ct_int[1], ct_int[0]] = (center_pt - ct_int)[:2]
            if label['track_id'] in prev_loc:
                offset[2:, ct_int[1], ct_int[0]] = (center_pt - prev_loc[label['track_id']])[:2]

            ids[:, ct_int[1], ct_int[0]] = int(label['track_id'])
            size[:, ct_int[1], ct_int[0]] = lwh
            alpha = yaw
            if alpha < np.pi / 6. or alpha > 5 * np.pi / 6.:
                rotbin[0, ct_int[1], ct_int[0]] = 1
                rotres[0, ct_int[1], ct_int[0]] = alpha - (-0.5 * np.pi)
            if alpha > -np.pi / 6. or alpha < -5 * np.pi / 6.:
                rotbin[1, ct_int[1], ct_int[0]] = 1
                rotres[1, ct_int[1], ct_int[0]] = alpha - (0.5 * np.pi)

        return center, valid_mask, ids, offset, size, rotbin, rotres

    def get_img_gt(self, bbox, sx, sy, crop):
        H = int(self.data_aug_conf['final_dim'][0] / self.img_downsample)
        W = int(self.data_aug_conf['final_dim'][1] / self.img_downsample)
        center = torch.zeros((3, H, W), dtype=torch.float32)
        offset = torch.zeros((2, H, W), dtype=torch.float32)
        size = torch.zeros((2, H, W), dtype=torch.float32)
        valid_mask = torch.zeros((1, H, W), dtype=torch.bool)
        person_ids = torch.zeros((1, H, W), dtype=torch.long)

        valid = torch.tensor(bbox['vehicle_class']) < 3
        classes = torch.tensor(bbox['vehicle_class'])[valid]
        img_pids = torch.tensor(bbox['vehicle_id'])[valid]
        img_pts = torch.tensor(bbox['bboxes'])[valid]

        if len(img_pts) == 0:
            return center, offset, size, person_ids, valid_mask

        for (pts, pid, idx) in zip(img_pts, img_pids, classes):
            (x1, y1), (x2, y2) = pts
            y1, y2 = 1080 - y2, 1080 - y1

            x1 = (x1 * sx - crop[0]) / self.img_downsample
            y1 = (y1 * sy - crop[1]) / self.img_downsample
            x2 = (x2 * sx - crop[0]) / self.img_downsample
            y2 = (y2 * sy - crop[1]) / self.img_downsample

            ct = torch.tensor([(x1 + x2) / 2, (y1 + y2) / 2])
            ct_int = ct.int()

            if ct_int[0] < 0 or ct_int[0] >= W or ct_int[1] < 0 or ct_int[1] >= H:
                continue

            basic.draw_umich_gaussian(center[idx], ct_int.int(), self.kernel_size)
            valid_mask[:, ct_int[1], ct_int[0]] = 1
            offset[:, ct_int[1], ct_int[0]] = ct - ct_int
            size[:, ct_int[1], ct_int[0]] = torch.tensor([x2 - x1, y2 - y1])
            person_ids[:, ct_int[1], ct_int[0]] = pid

        return center, offset, size, person_ids, valid_mask

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index) -> Any:
        town_num, town_name, town_path, frame = self.samples[index]
        cameras = self.cameras[town_name]
        scene_center = self.scene[town_name]['scene_center']

        ref_T_global = geom.merge_rt(torch.eye(3).unsqueeze(0), -scene_center.unsqueeze(0)).squeeze()
        if self.is_train:
            scene_center[:2] += torch.zeros(2, dtype=torch.float32).uniform_(-0.25, 0.25)
            ref_T_global = geom.merge_rt(torch.eye(3).unsqueeze(0), -scene_center.unsqueeze(0)).squeeze()

            while len(cameras) < self.num_train_cams:
                cameras[f'C{len(cameras) + 1:02d}'] = random.choice(list(cameras.values()))
            cameras = dict(random.sample(cameras.items(), self.num_train_cams))

        # label
        corners_global, labels, bbox_cam, town_name = self.get_bbox_global(index)  # N_,4,8
        corners_prev, labels_prev, _, town_prev = self.get_bbox_global(max(0, index - 1))  # N_,4,8
        if town_prev != town_name:
            corners_prev, labels_prev = corners_global, labels

        # image
        imgs, rots, trans, intrins, centers_img, offsets_img, sizes_img, tids_img, valids_img \
            = self.get_image_data(index, cameras, bbox_cam)
        extrinsic = geom.merge_rt(rots, trans.squeeze(-1))  # S 4 4

        # bev
        corners_ref = torch.matmul(ref_T_global, corners_global).transpose(1, 2)[:, :, :3].unsqueeze(0)
        corners_prev = torch.matmul(ref_T_global, corners_prev).transpose(1, 2)[:, :, :3].unsqueeze(0)
        center_bev, valid_bev, ids_bev, offset_bev, size_bev, rotbin_bev, rotres_bev \
            = self.get_bev_get(corners_ref, labels, corners_prev, labels_prev)

        center_gt = torch.mean(corners_ref, dim=2, keepdim=False)
        ref_gt = torch.zeros((self.max_objects, 3))
        ref_gt[:corners_ref.shape[1], :2] = center_gt[0, :, :2]
        ref_gt[:corners_ref.shape[1], 2] = torch.tensor([l['track_id'] for l in labels])

        item = {
            'img': imgs,  # S,3,288,512
            'intrinsic': intrins,  # S,4,4
            'extrinsic': extrinsic,  # S,4,4
            'scene_center': scene_center,  # 3,
            'ref_T_global': ref_T_global,  # 4,4
            'num_cameras': len(cameras),
            'town_path': town_path,
            'sequence_num': int(town_num),
            'frame': int(frame),
            'grid_gt': ref_gt  # N,4,2
        }
        target = {
            'valid_bev': valid_bev,  # 1,Y,X
            'center_bev': center_bev,  # 1,Y,X
            'offset_bev': offset_bev,  # 2,Y,X
            'size_bev': size_bev,  # 3,Y,X
            'rotbin_bev': rotbin_bev,  # 2,Y,X
            'rotres_bev': rotres_bev,  # 2,Y,Xv
            # img
            'center_img': centers_img,  # S,1,H/8,W/8
            'offset_img': offsets_img,  # S,2,H/8,W/8
            'size_img': sizes_img,  # S,2,H/8,W/8
            'valid_img': valids_img,  # S,1,H/8,W/8
            'pid_img': tids_img,  # S,1,H/8,W/8
        }

        return item, target


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    data = SynthehicleDataset('/usr/home/tee/Developer/datasets/synthehicle')
    print(len(data))
    loader = DataLoader(data, batch_size=16, shuffle=True)
    _, items = next(enumerate(loader))
