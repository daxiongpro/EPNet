import os
import numpy as np
import torch
import torch.utils.data as torch_data
import datasets.calibration as calibration
import datasets.kitti_utils as kitti_utils
from PIL import Image
import cv2

from config import cfg


class KittiDataset(torch_data.Dataset):
    def __init__(self, root_dir, split='train', classes='Car'):
        self.split = split

        self.trainset_dir = os.path.join(root_dir, 'KITTI', 'object', 'training')
        split_dir = os.path.join(root_dir, 'KITTI', 'ImageSets', split + '.txt')
        self.image_idx_list = [x.strip() for x in open(split_dir).readlines()]
        self.num_sample = self.image_idx_list.__len__()
        self.image_dir = os.path.join(self.trainset_dir, 'image_2')
        self.lidar_dir = os.path.join(self.trainset_dir, 'velodyne')
        self.calib_dir = os.path.join(self.trainset_dir, 'calib')
        self.label_dir = os.path.join(self.trainset_dir, 'label_2')
        self.plane_dir = os.path.join(self.trainset_dir, 'planes')
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        # Don't need to permute while using grid_sample
        self.image_hw_with_padding_np = np.array([1280., 384.])  # 用途？

        if classes == 'Car':
            self.classes = ('Background', 'Car')
            aug_scene_root_dir = os.path.join(root_dir, 'KITTI', 'aug_scene')
        elif classes == 'People':
            self.classes = ('Background', 'Pedestrian', 'Cyclist')
        elif classes == 'Pedestrian':
            self.classes = ('Background', 'Pedestrian')
            aug_scene_root_dir = os.path.join(root_dir, 'KITTI', 'aug_scene_ped')
        elif classes == 'Cyclist':
            self.classes = ('Background', 'Cyclist')
            aug_scene_root_dir = os.path.join(root_dir, 'KITTI', 'aug_scene_cyclist')
        else:
            assert False, "Invalid classes: %s" % classes

    def __len__(self):
        return len(self.image_idx_list)

    def __getitem__(self, index) -> dict:
        """获取单个样本
        @param index: (int)
        @return: sample_info (dict):
        dict_keys([
        'sample_id',
        'random_select',
        'img',
        'pts_origin_xy',
        'aug_method',
        'pts_input',
        'pts_rect',
        'pts_features',
        'rpn_cls_label',
        'rpn_reg_label',
        'gt_boxes3d'
        ])
        """

        sample_id = int(self.image_idx_list[index])
        img = self.get_image_rgb_with_normal(sample_id)
        pts_lidar = self.get_lidar(sample_id)  # (N, xyz_intensity) = (113110, 4)

        gt_obj_list = self.filtrate_objects(self.get_label(sample_id))
        gt_boxes3d = kitti_utils.objs_to_boxes3d(gt_obj_list)

        # get valid point (projected points should be in image)
        calib = self.get_calib(sample_id)
        pts_rect = calib.lidar_to_rect(pts_lidar[:, 0:3])  # 点云在相机坐标系下的坐标
        pts_intensity = pts_lidar[:, 3]  # lidar 强度
        pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)  # 点云投影到深度图，深度图的深度
        img_shape = self.get_image_shape(sample_id)
        pts_valid_flag = self.get_valid_flag(pts_rect, pts_img, pts_rect_depth,
                                             img_shape)  # np.ndarrya(True, False, False, True...)
        pts_intensity = pts_intensity[pts_valid_flag]
        pts_origin_xy = pts_img[pts_valid_flag]

        pts_rect = pts_rect[pts_valid_flag][:, 0:3]  # 少见写法，可以借鉴
        """
        >>> a = np.array([1, 2, 3])
        >>> b = np.array([ True, False,  True])
        >>> a[b]
        array([1, 3])
        """

        pts_intensity = pts_intensity[pts_valid_flag]
        pts_origin_xy = pts_img[pts_valid_flag]  # 点云在img上的坐标

        sample_info = {
            'sample_id': sample_id,
            'img': img,
            'pts_origin_xy': pts_origin_xy,
            'pts_input': pts_lidar,  # xyz_intensity坐标、
            'pts_rect': pts_rect,  # 点云在相机坐标系下坐标
            'pts_features': None,
            'cls_label': rpn_cls_label,
            'reg_label': rpn_reg_label,
            'gt_boxes3d': gt_boxes3d
        }
        return sample_info

    @staticmethod
    def get_valid_flag(pts_rect, pts_img, pts_rect_depth, img_shape):
        """
        Valid point should be in the image (and in the PC_AREA_SCOPE)
        :param pts_rect:
        :param pts_img:
        :param pts_rect_depth:
        :param img_shape:
        :return:
        """
        val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

        if cfg.PC_REDUCE_BY_RANGE:
            x_range, y_range, z_range = cfg.PC_AREA_SCOPE
            pts_x, pts_y, pts_z = pts_rect[:, 0], pts_rect[:, 1], pts_rect[:, 2]
            range_flag = (pts_x >= x_range[0]) & (pts_x <= x_range[1]) \
                         & (pts_y >= y_range[0]) & (pts_y <= y_range[1]) \
                         & (pts_z >= z_range[0]) & (pts_z <= z_range[1])
            pts_valid_flag = pts_valid_flag & range_flag
        return pts_valid_flag

    @staticmethod
    def check_pc_range(xyz):
        """查看xyz是否在指定范围内，z必须在[0, 70.4]范围内
        :param xyz: [x, y, z]
        :return: bool
        """
        x_range, y_range, z_range = cfg.PC_AREA_SCOPE
        if (x_range[0] <= xyz[0] <= x_range[1]) and (y_range[0] <= xyz[1] <= y_range[1]) and \
                (z_range[0] <= xyz[2] <= z_range[1]):
            return True
        return False

    def filtrate_objects(self, obj_list):
        """忽略掉一些不在范围内或不需要检测的object
        Discard objects which are not in self.classes (or its similar classes)
        :param obj_list: list
        :return: list
        """
        type_whitelist = self.classes

        type_whitelist = list(self.classes)
        if 'Car' in self.classes:
            type_whitelist.append('Van')
        if 'Pedestrian' in self.classes:  # or 'Cyclist' in self.classes:
            type_whitelist.append('Person_sitting')

        valid_obj_list = []
        for obj in obj_list:
            if obj.cls_type not in type_whitelist:
                continue
            if cfg.PC_REDUCE_BY_RANGE and (self.check_pc_range(obj.pos) is False):
                continue
            valid_obj_list.append(obj)
        return valid_obj_list

    def get_image(self, idx):
        # cv2.setNumThreads(0)  # for solving deadlock when switching epoch
        img_file = os.path.join(self.image_dir, '%06d.png' % idx)

        return cv2.imread(img_file)  # (H, W, 3) BGR mode

    def get_image_rgb_with_normal(self, idx):
        """
        return img with normalization in rgb mode
        :param idx:
        :return: imback(H,W,3)
        """
        img_file = os.path.join(self.image_dir, '%06d.png' % idx)
        assert os.path.exists(img_file)
        im = Image.open(img_file).convert('RGB')
        im = np.array(im).astype(np.float)
        im = im / 255.0
        im -= self.mean
        im /= self.std
        # print(im.shape)
        # ~[-2,2]
        # im = im[:, :, ::-1]
        # make same size padding with 0
        imback = np.zeros([384, 1280, 3], dtype=np.float)
        imback[:im.shape[0], :im.shape[1], :] = im

        return imback  # (H,W,3) RGB mode

    def get_image_shape_with_padding(self, idx=0):
        return 384, 1280, 3

    def get_image_shape(self, idx):
        img_file = os.path.join(self.image_dir, '%06d.png' % idx)
        assert os.path.exists(img_file)
        im = Image.open(img_file)
        width, height = im.size
        return height, width, 3

    def get_lidar(self, idx):
        lidar_file = os.path.join(self.lidar_dir, '%06d.bin' % idx)
        assert os.path.exists(lidar_file)
        return np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)

    def get_calib(self, idx):
        calib_file = os.path.join(self.calib_dir, '%06d.txt' % idx)
        assert os.path.exists(calib_file)
        return calibration.Calibration(calib_file)

    def get_label(self, idx):
        label_file = os.path.join(self.label_dir, '%06d.txt' % idx)
        assert os.path.exists(label_file)
        return kitti_utils.get_objects_from_label(label_file)

    def get_road_plane(self, idx):
        plane_file = os.path.join(self.plane_dir, '%06d.txt' % idx)
        with open(plane_file, 'r') as f:
            lines = f.readlines()
        lines = [float(i) for i in lines[3].split()]
        plane = np.asarray(lines)

        # Ensure normal is always facing up, this is in the rectified camera coordinate
        if plane[1] > 0:
            plane = -plane

        norm = np.linalg.norm(plane[0:3])
        plane = plane / norm
        return plane


if __name__ == '__main__':
    root_dir = r'D:\code\EPNet\data'
    dataset = KittiDataset(root_dir=root_dir, split="train")
    print(dataset[1])
