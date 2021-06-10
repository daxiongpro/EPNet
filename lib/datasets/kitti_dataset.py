import os
import numpy as np
import torch
import torch.utils.data as torch_data
import lib.utils.calibration as calibration
import lib.utils.kitti_utils as kitti_utils
from PIL import Image
import cv2


class KittiDataset(torch_data.Dataset):
    def __init__(self, root_dir, split='train'):
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

    def __len__(self):
        return len(self.image_idx_list)

    def __getitem__(self, index) -> dict:
        """
        @param index: int
        @return: sample_info -> dict
        """
        sample_id = int(self.image_idx_list[index])
        if sample_id < 10000:
            calib = self.get_calib(sample_id)
            img = self.get_image_rgb_with_normal(sample_id)
            img_shape = self.get_image_shape(sample_id)
            pts_lidar = self.get_lidar(sample_id)

            # get valid point (projected points should be in image)
            pts_rect = calib.lidar_to_rect(pts_lidar[:, 0:3])  # 用途？
            pts_intensity = pts_lidar[:, 3]  # 用途？

        pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
        pts_valid_flag = self.get_valid_flag(pts_rect, pts_img, pts_rect_depth, img_shape)

        pts_rect = pts_rect[pts_valid_flag][:, 0:3]

        pts_intensity = pts_intensity[pts_valid_flag]
        pts_origin_xy = pts_img[pts_valid_flag]
        # generate inputs


        ret_pts_features = np.concatenate(pts_features, axis=1) if pts_features.__len__() > 1 else pts_features[0]

        sample_info = {'sample_id': sample_id, 'random_select': self.random_select, 'img': img,
                       'pts_origin_xy': pts_origin_xy}

        if self.mode == 'TEST':
            if cfg.RPN.USE_INTENSITY:
                pts_input = np.concatenate((ret_pts_rect, ret_pts_features), axis=1)  # (N, C)
            else:
                pts_input = ret_pts_rect
            sample_info['pts_input'] = pts_input
            sample_info['pts_rect'] = ret_pts_rect
            sample_info['pts_features'] = ret_pts_features

            return sample_info

        gt_obj_list = self.filtrate_objects(self.get_label(sample_id))
        # if cfg.GT_AUG_ENABLED and self.mode == 'TRAIN' and gt_aug_flag:
        #     gt_obj_list.extend(extra_gt_obj_list)
        gt_boxes3d = kitti_utils.objs_to_boxes3d(gt_obj_list)

        gt_alpha = np.zeros((gt_obj_list.__len__()), dtype=np.float32)
        for k, obj in enumerate(gt_obj_list):
            gt_alpha[k] = obj.alpha

        # data augmentation
        aug_pts_rect = ret_pts_rect.copy()
        aug_gt_boxes3d = gt_boxes3d.copy()
        if cfg.AUG_DATA and self.mode == 'TRAIN':
            #
            aug_pts_rect, aug_gt_boxes3d, aug_method = self.data_augmentation(aug_pts_rect, aug_gt_boxes3d, gt_alpha,
                                                                              sample_id)
            sample_info['aug_method'] = aug_method

        # prepare input
        if cfg.RPN.USE_INTENSITY:
            pts_input = np.concatenate((aug_pts_rect, ret_pts_features), axis=1)  # (N, C)
        else:
            pts_input = aug_pts_rect

        if cfg.RPN.FIXED:
            sample_info['pts_input'] = pts_input
            sample_info['pts_rect'] = aug_pts_rect
            #
            sample_info['pts_features'] = ret_pts_features
            sample_info['gt_boxes3d'] = aug_gt_boxes3d
            return sample_info

        # generate training labels
        rpn_cls_label, rpn_reg_label = self.generate_rpn_training_labels(aug_pts_rect, aug_gt_boxes3d)
        sample_info['pts_input'] = pts_input
        sample_info['pts_rect'] = aug_pts_rect
        sample_info['pts_features'] = ret_pts_features
        sample_info['rpn_cls_label'] = rpn_cls_label
        sample_info['rpn_reg_label'] = rpn_reg_label
        sample_info['gt_boxes3d'] = aug_gt_boxes3d
        return sample_info

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
    a = torch.ones(3)
    a.cuda()
    root_dir = "../../data/"
    dataset = KittiDataset(root_dir=root_dir, split="train")
    print(dataset[1])
