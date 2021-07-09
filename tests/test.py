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
       pass

if __name__ == '__main__':
    root_dir = r'D:\code\EPNet\data'
    dataset = KittiDataset(root_dir, split='train', classes='Car')