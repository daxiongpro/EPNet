import logging
import numpy as np
import os
import pickle
import torch
from lib.datasets.kitti_dataset import KittiDataset
import lib.datasets.kitti_utils as kitti_utils
from lib.config import cfg
from torch.nn.functional import grid_sample


def interpolate_img_by_xy(img, xy, normal_shape):
    """
    :param img:(H,W,c)
    :param xy:(N,2) (x,y)->(w,h)
    :param normal_shape:(2),H_size and W_size
    :return:interpolated features (N,3)
    """
    # (B,3,H,W)
    channel = img.shape[-1]
    img = torch.from_numpy(img).unsqueeze(0).permute(0, 3, 1, 2)
    # print(xy.min(),xy.max())
    xy = xy * 2 / (normal_shape - 1.) - 1.
    # print(xy.min(), xy.max())

    # print(xy)
    xy = torch.from_numpy(xy).view(1, 1, -1, 2)
    # xy=torch.cat([xy[:,:,:,1:2],xy[:,:,:,0:1]],dim = 3)
    # (1,3,1,N)
    ret_img = grid_sample(img, xy, padding_mode='zeros', mode='bilinear')
    # (N,3)
    ret_img = ret_img.view(channel, -1).permute(1, 0).numpy()

    return ret_img


class KittiSSDDataset(KittiDataset):
    def __init__(self,
                 root_dir,
                 npoints=16384,
                 split='train',
                 classes='Car',
                 mode='TRAIN',
                 random_select=True,
                 logger=None,
                 rcnn_training_roi_dir=None,
                 rcnn_training_feature_dir=None,
                 rcnn_eval_roi_dir=None,
                 rcnn_eval_feature_dir=None,
                 gt_database_dir=None):
        super().__init__(root_dir=root_dir, split=split)
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

        self.num_class = self.classes.__len__()

        self.npoints = npoints
        self.sample_id_list = []
        self.random_select = random_select
        self.logger = logger

        if split == 'train_aug':
            self.aug_label_dir = os.path.join(aug_scene_root_dir, 'training', 'aug_label')
            self.aug_pts_dir = os.path.join(aug_scene_root_dir, 'training', 'rectified_data')
        else:
            self.aug_label_dir = os.path.join(aug_scene_root_dir, 'training', 'aug_label')
            self.aug_pts_dir = os.path.join(aug_scene_root_dir, 'training', 'rectified_data')

        # for rcnn training
        self.rcnn_training_bbox_list = []
        self.rpn_feature_list = {}
        self.pos_bbox_list = []
        self.neg_bbox_list = []
        self.far_neg_bbox_list = []
        self.rcnn_eval_roi_dir = rcnn_eval_roi_dir
        self.rcnn_eval_feature_dir = rcnn_eval_feature_dir
        self.rcnn_training_roi_dir = rcnn_training_roi_dir
        self.rcnn_training_feature_dir = rcnn_training_feature_dir

        self.gt_database = None

        if not self.random_select:
            self.logger.warning('random select is False')

        assert mode in ['TRAIN', 'EVAL', 'TEST'], 'Invalid mode: %s' % mode
        self.mode = mode

        if cfg.RPN.ENABLED:
            if gt_database_dir is not None:
                self.gt_database = pickle.load(open(gt_database_dir, 'rb'))

                if cfg.GT_AUG_HARD_RATIO > 0:
                    easy_list, hard_list = [], []
                    for k in range(self.gt_database.__len__()):
                        obj = self.gt_database[k]
                        if obj['points'].shape[0] > 100:
                            easy_list.append(obj)
                        else:
                            hard_list.append(obj)
                    self.gt_database = [easy_list, hard_list]
                    logger.info('Loading gt_database(easy(pt_num>100): %d, hard(pt_num<=100): %d) from %s'
                                % (len(easy_list), len(hard_list), gt_database_dir))
                else:
                    logger.info('Loading gt_database(%d) from %s' % (len(self.gt_database), gt_database_dir))

            if mode == 'TRAIN':
                self.preprocess_rpn_training_data()
            else:
                self.sample_id_list = [int(sample_id) for sample_id in self.image_idx_list]
                self.logger.info('Load testing samples from %s' % self.imageset_dir)
                self.logger.info('Done: total test samples %d' % len(self.sample_id_list))
        elif cfg.RCNN.ENABLED:
            self.sample_id_list = [int(sample_id) for sample_id in self.image_idx_list]
            self.logger.info('Load testing samples from %s' % self.imageset_dir)
            self.logger.info('Done: total test samples %d' % len(self.sample_id_list))
            # for idx in range(0, self.num_sample):
            #     sample_id = int(self.image_idx_list[idx])
            #     obj_list = self.filtrate_objects(self.get_label(sample_id))
            #     if len(obj_list) == 0:
            #         # logger.info('No gt classes: %06d' % sample_id)
            #         continue
            #     self.sample_id_list.append(sample_id)
            #
            # print('Done: filter %s results for rcnn training: %d / %d\n' %
            #       (self.mode, len(self.sample_id_list), len(self.image_idx_list)))

    def preprocess_rpn_training_data(self):
        """丢弃当前没有类的样本，这些类不会用于培训。
        Discard samples which don't have current classes, which will not be used for training.
        Valid sample_id is stored in self.sample_id_list
        """
        self.logger.info('Loading %s samples from %s ...' % (self.mode, self.label_dir))
        for idx in range(0, self.num_sample):
            sample_id = int(self.image_idx_list[idx])
            obj_list = self.filtrate_objects(self.get_label(sample_id))
            # if cfg.LI_FUSION.ENABLED:  #####
            if len(obj_list) == 0:
                # self.logger.info('No gt classes: %06d' % sample_id)
                continue
            self.sample_id_list.append(sample_id)

        self.logger.info('Done: filter %s results: %d / %d\n' % (self.mode, len(self.sample_id_list),
                                                                 len(self.image_idx_list)))

    def get_label(self, idx):
        if idx < 10000:
            label_file = os.path.join(self.label_dir, '%06d.txt' % idx)
        else:
            label_file = os.path.join(self.aug_label_dir, '%06d.txt' % idx)

        assert os.path.exists(label_file)
        return kitti_utils.get_objects_from_label(label_file)

    def get_image(self, idx):
        return super().get_image(idx % 10000)

    def get_image_shape(self, idx):
        return super().get_image_shape(idx % 10000)

    def get_calib(self, idx):
        return super().get_calib(idx % 10000)

    def get_road_plane(self, idx):
        return super().get_road_plane(idx % 10000)

    @staticmethod
    def get_rpn_features(rpn_feature_dir, idx):
        rpn_feature_file = os.path.join(rpn_feature_dir, '%06d.npy' % idx)
        rpn_xyz_file = os.path.join(rpn_feature_dir, '%06d_xyz.npy' % idx)
        rpn_intensity_file = os.path.join(rpn_feature_dir, '%06d_intensity.npy' % idx)
        if cfg.RCNN.USE_SEG_SCORE:
            rpn_seg_file = os.path.join(rpn_feature_dir, '%06d_rawscore.npy' % idx)
            rpn_seg_score = np.load(rpn_seg_file).reshape(-1)
            rpn_seg_score = torch.sigmoid(torch.from_numpy(rpn_seg_score)).numpy()
        else:
            rpn_seg_file = os.path.join(rpn_feature_dir, '%06d_seg.npy' % idx)
            rpn_seg_score = np.load(rpn_seg_file).reshape(-1)
        return np.load(rpn_xyz_file), np.load(rpn_feature_file), np.load(rpn_intensity_file).reshape(-1), rpn_seg_score

    def filtrate_objects(self, obj_list):
        """忽略掉一些不在范围内或不需要检测的object
        Discard objects which are not in self.classes (or its similar classes)
        :param obj_list: list
        :return: list
        """
        type_whitelist = self.classes
        if self.mode == 'TRAIN' and cfg.INCLUDE_SIMILAR_TYPE:
            type_whitelist = list(self.classes)
            if 'Car' in self.classes:
                type_whitelist.append('Van')
            if 'Pedestrian' in self.classes:  # or 'Cyclist' in self.classes:
                type_whitelist.append('Person_sitting')

        valid_obj_list = []
        for obj in obj_list:
            if obj.cls_type not in type_whitelist:
                continue
            if self.mode == 'TRAIN' and cfg.PC_REDUCE_BY_RANGE and (self.check_pc_range(obj.pos) is False):
                continue
            valid_obj_list.append(obj)
        return valid_obj_list

    @staticmethod
    def filtrate_dc_objects(obj_list):
        valid_obj_list = []
        for obj in obj_list:
            if obj.cls_type in ['DontCare']:
                continue
            valid_obj_list.append(obj)

        return valid_obj_list

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

    @staticmethod
    def get_valid_flag(pts_rect, pts_img, pts_rect_depth, img_shape):
        """
        返回是否为有效点
        Valid point should be in the image (and in the PC_AREA_SCOPE)
        :param pts_rect:点在相机坐标系下的坐标
        :param pts_img:点在图像坐标系下的坐标
        :param pts_rect_depth:点在相机坐标系下的深度
        :param img_shape:
        :return:[True, False, False, True, ...]
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

    def __len__(self):
        if cfg.RPN.ENABLED:
            return len(self.sample_id_list)
        elif cfg.RCNN.ENABLED:
            if self.mode == 'TRAIN':
                return len(self.sample_id_list)
            else:
                return len(self.image_idx_list)
        else:
            raise NotImplementedError

    def __getitem__(self, index):
        if cfg.LI_FUSION.ENABLED:
            return self.get_rpn_with_li_fusion(index)

        if cfg.RPN.ENABLED:
            return self.get_rpn_sample(index)
        elif cfg.RCNN.ENABLED:
            if self.mode == 'TRAIN':
                if cfg.RCNN.ROI_SAMPLE_JIT:
                    return self.get_rcnn_sample_jit(index)
                else:
                    pass
                    # return self.get_rcnn_training_sample_batch(index)
            else:
                pass
                # return self.get_proposal_from_file(index)
        else:
            raise NotImplementedError

    def get_rpn_with_li_fusion(self, index):
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
        sample_id = int(self.sample_id_list[index])
        if sample_id < 10000:
            calib = self.get_calib(sample_id)
            img = self.get_image_rgb_with_normal(sample_id)
            img_shape = self.get_image_shape(sample_id)
            pts_lidar = self.get_lidar(sample_id)

            # get valid point (projected points should be in image)
            pts_rect = calib.lidar_to_rect(pts_lidar[:, 0:3])  # 点云在相机坐标系下的坐标
            pts_intensity = pts_lidar[:, 3]

        else:
            assert False, print('unable to use aug data with img align')
            calib = self.get_calib(sample_id % 10000)
            # img = self.get_image_by_python(sample_id % 10000)
            img_shape = self.get_image_shape(sample_id % 10000)

            pts_file = os.path.join(self.aug_pts_dir, '%06d.bin' % sample_id)
            assert os.path.exists(pts_file), '%s' % pts_file
            aug_pts = np.fromfile(pts_file, dtype=np.float32).reshape(-1, 4)
            pts_rect, pts_intensity = aug_pts[:, 0:3], aug_pts[:, 3]

        pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)  # 点云投影到深度图，深度图的深度
        pts_valid_flag = self.get_valid_flag(pts_rect, pts_img, pts_rect_depth, img_shape)

        pts_rect = pts_rect[pts_valid_flag][:, 0:3]  # 少见写法，可以借鉴
        """
        >>> a = np.array([1, 2, 3])
        >>> b = np.array([ True, False,  True])
        >>> a[b]
        array([1, 3])
        """
        pts_intensity = pts_intensity[pts_valid_flag]
        pts_origin_xy = pts_img[pts_valid_flag]  # 点云在img上的坐标

        # TODO use GT_AUG_ENABLE, failed to align img2d to random box from other scene
        # if cfg.GT_AUG_ENABLED and self.mode == 'TRAIN':
        #     # all labels for checking overlapping
        #     all_gt_obj_list = self.filtrate_dc_objects(self.get_label(sample_id))
        #     all_gt_boxes3d = kitti_utils.objs_to_boxes3d(all_gt_obj_list)
        #
        #     gt_aug_flag = False
        #     if np.random.rand() < cfg.GT_AUG_APPLY_PROB:
        #         # augment one scene
        #         gt_aug_flag, pts_rect, pts_intensity, extra_gt_boxes3d, extra_gt_obj_list = \
        #             self.apply_gt_aug_to_one_scene(sample_id, pts_rect, pts_intensity, all_gt_boxes3d)

        # generate inputs
        if self.mode == 'TRAIN' or self.random_select:
            # make sure len(pts_rect) ==self.npoints
            if self.npoints < len(pts_rect):
                pts_depth = pts_rect[:, 2]
                pts_near_flag = pts_depth < 40.0
                far_idxs_choice = np.where(pts_near_flag == 0)[0]
                near_idxs = np.where(pts_near_flag == 1)[0]
                near_idxs_choice = np.random.choice(near_idxs, self.npoints - len(far_idxs_choice), replace=False)

                choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0) \
                    if len(far_idxs_choice) > 0 else near_idxs_choice
                np.random.shuffle(choice)
            else:
                choice = np.arange(0, len(pts_rect), dtype=np.int32)
                if self.npoints > len(pts_rect):
                    extra_choice = np.random.choice(choice, self.npoints - len(pts_rect), replace=False)
                    choice = np.concatenate((choice, extra_choice), axis=0)
                np.random.shuffle(choice)

            ret_pts_rect = pts_rect[choice, :]
            ret_pts_intensity = pts_intensity[choice] - 0.5  # translate intensity to [-0.5, 0.5]
            ret_pts_origin_xy = pts_origin_xy[choice, :]
        else:
            ret_pts_rect = pts_rect
            ret_pts_intensity = pts_intensity - 0.5
            ret_pts_origin_xy = pts_origin_xy[choice, :]

        pts_features = [ret_pts_intensity.reshape(-1, 1)]
        ret_pts_features = np.concatenate(pts_features, axis=1) if pts_features.__len__() > 1 else pts_features[0]

        sample_info = {'sample_id': sample_id, 'random_select': self.random_select, 'img': img,
                       'pts_origin_xy': ret_pts_origin_xy}

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
        sample_info['pts_input'] = pts_input  # xyz_intensity坐标
        sample_info['pts_rect'] = aug_pts_rect  # 点云在相机坐标系下坐标 pts_rect: (N, 3)
        sample_info['pts_features'] = ret_pts_features
        sample_info['rpn_cls_label'] = rpn_cls_label
        sample_info['rpn_reg_label'] = rpn_reg_label
        sample_info['gt_boxes3d'] = aug_gt_boxes3d
        return sample_info

    def get_rpn_sample(self, index):
        sample_id = int(self.sample_id_list[index])
        if sample_id < 10000:
            calib = self.get_calib(sample_id)
            img = self.get_image_rgb_with_normal(sample_id)
            img_shape = self.get_image_shape(sample_id)
            pts_lidar = self.get_lidar(sample_id)

            # get valid point (projected points should be in image)
            pts_rect = calib.lidar_to_rect(pts_lidar[:, 0:3])
            pts_intensity = pts_lidar[:, 3]
        else:
            calib = self.get_calib(sample_id % 10000)
            img = self.get_image_rgb_with_normal(sample_id % 10000)
            img_shape = self.get_image_shape(sample_id % 10000)

            pts_file = os.path.join(self.aug_pts_dir, '%06d.bin' % sample_id)
            assert os.path.exists(pts_file), '%s' % pts_file
            aug_pts = np.fromfile(pts_file, dtype=np.float32).reshape(-1, 4)
            pts_rect, pts_intensity = aug_pts[:, 0:3], aug_pts[:, 3]

        pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
        pts_valid_flag = self.get_valid_flag(pts_rect, pts_img, pts_rect_depth, img_shape)

        pts_rect = pts_rect[pts_valid_flag][:, 0:3]
        pts_intensity = pts_intensity[pts_valid_flag]
        pts_in_img = pts_img[pts_valid_flag]

        # 在这里加上RGB信息。
        shape = self.image_hw_with_padding_np
        pts_rgb = interpolate_img_by_xy(img, pts_in_img, shape)

        if cfg.GT_AUG_ENABLED and self.mode == 'TRAIN':
            # print(pts_intensity.shape, pts_rgb.shape)
            pts_features = np.concatenate([pts_intensity.reshape(-1, 1), pts_rgb], axis=1)
            # all labels for checking overlapping
            all_gt_obj_list = self.filtrate_dc_objects(self.get_label(sample_id))
            all_gt_boxes3d = kitti_utils.objs_to_boxes3d(all_gt_obj_list)

            gt_aug_flag = False
            if np.random.rand() < cfg.GT_AUG_APPLY_PROB:
                # augment one scene
                gt_aug_flag, pts_rect, pts_features, extra_gt_boxes3d, extra_gt_obj_list = \
                    self.apply_gt_aug_to_one_scene(sample_id, pts_rect, pts_features, all_gt_boxes3d)
            pts_intensity, pts_rgb = pts_features[:, 0], pts_features[:, 1:]
        # generate inputs
        if self.mode == 'TRAIN' or self.random_select:
            if self.npoints < len(pts_rect):
                pts_depth = pts_rect[:, 2]
                pts_near_flag = pts_depth < 40.0
                far_idxs_choice = np.where(pts_near_flag == 0)[0]
                near_idxs = np.where(pts_near_flag == 1)[0]
                near_idxs_choice = np.random.choice(near_idxs, self.npoints - len(far_idxs_choice), replace=False)

                choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0) \
                    if len(far_idxs_choice) > 0 else near_idxs_choice
                np.random.shuffle(choice)
            else:
                choice = np.arange(0, len(pts_rect), dtype=np.int32)
                if self.npoints > len(pts_rect):
                    extra_choice = np.random.choice(choice, self.npoints - len(pts_rect), replace=False)
                    choice = np.concatenate((choice, extra_choice), axis=0)
                np.random.shuffle(choice)

            ret_pts_rect = pts_rect[choice, :]
            ret_pts_intensity = pts_intensity[choice] - 0.5  # translate intensity to [-0.5, 0.5]
            ret_pts_rgb = pts_rgb[choice, :]

        else:
            ret_pts_rect = pts_rect
            ret_pts_intensity = pts_intensity - 0.5

            ret_pts_rgb = pts_rgb
        pts_features = []
        if cfg.RPN.USE_INTENSITY:
            pts_features.append(ret_pts_intensity.reshape(-1, 1))
        if cfg.RPN.USE_RGB:
            pts_features.append(ret_pts_rgb)
        ret_pts_features = None
        if pts_features.__len__() >= 1:
            ret_pts_features = np.concatenate(pts_features, axis=1)

        sample_info = {'sample_id': sample_id, 'random_select': self.random_select, 'rgb': ret_pts_rgb}

        if self.mode == 'TEST':
            if ret_pts_features != None:
                pts_input = np.concatenate((ret_pts_rect, ret_pts_features), axis=1)  # (N, C)
            else:
                pts_input = ret_pts_rect

            sample_info['pts_input'] = pts_input
            sample_info['pts_rect'] = ret_pts_rect
            sample_info['pts_features'] = ret_pts_features
            return sample_info

        gt_obj_list = self.filtrate_objects(self.get_label(sample_id))
        if cfg.GT_AUG_ENABLED and self.mode == 'TRAIN' and gt_aug_flag:
            gt_obj_list.extend(extra_gt_obj_list)
        gt_boxes3d = kitti_utils.objs_to_boxes3d(gt_obj_list)

        gt_alpha = np.zeros((gt_obj_list.__len__()), dtype=np.float32)
        for k, obj in enumerate(gt_obj_list):
            gt_alpha[k] = obj.alpha

        # data augmentation
        aug_pts_rect = ret_pts_rect.copy()
        aug_gt_boxes3d = gt_boxes3d.copy()
        if cfg.AUG_DATA and self.mode == 'TRAIN':
            aug_pts_rect, aug_gt_boxes3d, aug_method = self.data_augmentation(aug_pts_rect, aug_gt_boxes3d, gt_alpha,
                                                                              sample_id)
            sample_info['aug_method'] = aug_method

        # prepare input
        if cfg.RPN.USE_INTENSITY or cfg.RPN.USE_RGB:
            pts_input = np.concatenate((aug_pts_rect, ret_pts_features), axis=1)  # (N, C)
        else:
            pts_input = aug_pts_rect

        if cfg.RPN.FIXED:
            sample_info['pts_input'] = pts_input
            sample_info['pts_rect'] = aug_pts_rect
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

    @staticmethod
    def generate_rpn_training_labels(pts_rect, gt_boxes3d):
        """
        判断pts_rect中的点是否在gt_boxes3d内部，如果是，则pts_rect对应的cls_label赋为对应的标签
        :param pts_rect: 点云在img坐标系的坐标
        :param gt_boxes3d: img坐标系下的回归框
        :return:
        cls_label:(N, 1)
        reg_label:(N, 7)
        """
        cls_label = np.zeros((pts_rect.shape[0]), dtype=np.int32)
        reg_label = np.zeros((pts_rect.shape[0], 7), dtype=np.float32)  # dx, dy, dz, ry, h, w, l
        gt_corners = kitti_utils.boxes3d_to_corners3d(gt_boxes3d, rotate=True)
        extend_gt_boxes3d = kitti_utils.enlarge_box3d(gt_boxes3d, extra_width=0.2)
        extend_gt_corners = kitti_utils.boxes3d_to_corners3d(extend_gt_boxes3d, rotate=True)
        for k in range(gt_boxes3d.shape[0]):
            box_corners = gt_corners[k]
            fg_pt_flag = kitti_utils.in_hull(pts_rect, box_corners)
            fg_pts_rect = pts_rect[fg_pt_flag]
            cls_label[fg_pt_flag] = 1

            # enlarge the bbox3d, ignore nearby points
            extend_box_corners = extend_gt_corners[k]
            fg_enlarge_flag = kitti_utils.in_hull(pts_rect, extend_box_corners)
            ignore_flag = np.logical_xor(fg_pt_flag, fg_enlarge_flag)
            cls_label[ignore_flag] = -1

            # pixel offset of object center
            center3d = gt_boxes3d[k][0:3].copy()  # (x, y, z)
            center3d[1] -= gt_boxes3d[k][3] / 2
            reg_label[fg_pt_flag, 0:3] = center3d - fg_pts_rect  # Now y is the true center of 3d box 20180928

            # size and angle encoding
            reg_label[fg_pt_flag, 3] = gt_boxes3d[k][3]  # h
            reg_label[fg_pt_flag, 4] = gt_boxes3d[k][4]  # w
            reg_label[fg_pt_flag, 5] = gt_boxes3d[k][5]  # l
            reg_label[fg_pt_flag, 6] = gt_boxes3d[k][6]  # ry

        return cls_label, reg_label

    def rotate_box3d_along_y(self, box3d, rot_angle):
        old_x, old_z, ry = box3d[0], box3d[2], box3d[6]
        old_beta = np.arctan2(old_z, old_x)
        alpha = -np.sign(old_beta) * np.pi / 2 + old_beta + ry

        box3d = kitti_utils.rotate_pc_along_y(box3d.reshape(1, 7), rot_angle=rot_angle)[0]
        new_x, new_z = box3d[0], box3d[2]
        new_beta = np.arctan2(new_z, new_x)
        box3d[6] = np.sign(new_beta) * np.pi / 2 + alpha - new_beta

        return box3d

    def apply_gt_aug_to_one_scene(self, sample_id, pts_rect, pts_features_offline, all_gt_boxes3d):
        """
        :param pts_rect: (N, 3)
        :param all_gt_boxex3d: (M2, 7)
        :return:
        """
        assert self.gt_database is not None
        # extra_gt_num = np.random.randint(10, 15)
        # try_times = 50
        if cfg.GT_AUG_RAND_NUM:
            extra_gt_num = np.random.randint(10, cfg.GT_EXTRA_NUM)
        else:
            extra_gt_num = cfg.GT_EXTRA_NUM
        try_times = 100
        cnt = 0
        cur_gt_boxes3d = all_gt_boxes3d.copy()
        cur_gt_boxes3d[:, 4] += 0.5  # TODO: consider different objects
        cur_gt_boxes3d[:, 5] += 0.5  # enlarge new added box to avoid too nearby boxes
        cur_gt_corners = kitti_utils.boxes3d_to_corners3d(cur_gt_boxes3d)

        extra_gt_obj_list = []
        extra_gt_boxes3d_list = []
        new_pts_list, new_pts_intensity_list = [], []
        src_pts_flag = np.ones(pts_rect.shape[0], dtype=np.int32)

        road_plane = self.get_road_plane(sample_id)
        a, b, c, d = road_plane

        while try_times > 0:
            if cnt > extra_gt_num:
                break

            try_times -= 1
            if cfg.GT_AUG_HARD_RATIO > 0:
                p = np.random.rand()
                if p > cfg.GT_AUG_HARD_RATIO:
                    # use easy sample
                    rand_idx = np.random.randint(0, len(self.gt_database[0]))
                    new_gt_dict = self.gt_database[0][rand_idx]
                else:
                    # use hard sample
                    rand_idx = np.random.randint(0, len(self.gt_database[1]))
                    new_gt_dict = self.gt_database[1][rand_idx]
            else:
                rand_idx = np.random.randint(0, self.gt_database.__len__())
                new_gt_dict = self.gt_database[rand_idx]

            new_gt_box3d = new_gt_dict['gt_box3d'].copy()
            new_gt_points = new_gt_dict['points'].copy()
            new_gt_intensity = new_gt_dict['intensity'].copy()
            new_gt_rgb = new_gt_dict['rgb'].copy()
            new_gt_features = np.concatenate([new_gt_intensity.reshape(-1, 1), new_gt_rgb], axis=1)

            new_gt_obj = new_gt_dict['obj']
            center = new_gt_box3d[0:3]
            if cfg.PC_REDUCE_BY_RANGE and (self.check_pc_range(center) is False):
                continue

            if new_gt_points.__len__() < 5:  # too few points
                continue

            # put it on the road plane
            cur_height = (-d - a * center[0] - c * center[2]) / b
            move_height = new_gt_box3d[1] - cur_height
            new_gt_box3d[1] -= move_height
            new_gt_points[:, 1] -= move_height
            new_gt_obj.pos[1] -= move_height

            new_enlarged_box3d = new_gt_box3d.copy()
            new_enlarged_box3d[4] += 0.5
            new_enlarged_box3d[5] += 0.5  # enlarge new added box to avoid too nearby boxes

            cnt += 1
            new_corners = kitti_utils.boxes3d_to_corners3d(new_enlarged_box3d.reshape(1, 7))
            iou3d = kitti_utils.get_iou3d(new_corners, cur_gt_corners)
            valid_flag = iou3d.max() < 1e-8
            if not valid_flag:
                continue

            enlarged_box3d = new_gt_box3d.copy()
            enlarged_box3d[3] += 2  # remove the points above and below the object

            boxes_pts_mask_list = roipool3d_utils.pts_in_boxes3d_cpu(
                torch.from_numpy(pts_rect), torch.from_numpy(enlarged_box3d.reshape(1, 7)))
            pt_mask_flag = (boxes_pts_mask_list[0].numpy() == 1)
            src_pts_flag[pt_mask_flag] = 0  # remove the original points which are inside the new box

            new_pts_list.append(new_gt_points)
            new_pts_intensity_list.append(new_gt_features)
            cur_gt_boxes3d = np.concatenate((cur_gt_boxes3d, new_enlarged_box3d.reshape(1, 7)), axis=0)
            cur_gt_corners = np.concatenate((cur_gt_corners, new_corners), axis=0)
            extra_gt_boxes3d_list.append(new_gt_box3d.reshape(1, 7))
            extra_gt_obj_list.append(new_gt_obj)

        if new_pts_list.__len__() == 0:
            return False, pts_rect, pts_features_offline, None, None

        extra_gt_boxes3d = np.concatenate(extra_gt_boxes3d_list, axis=0)
        # remove original points and add new points
        pts_rect = pts_rect[src_pts_flag == 1]
        pts_features_offline = pts_features_offline[src_pts_flag == 1]
        new_pts_rect = np.concatenate(new_pts_list, axis=0)
        new_pts_intensity = np.concatenate(new_pts_intensity_list, axis=0)
        pts_rect = np.concatenate((pts_rect, new_pts_rect), axis=0)
        pts_features_offline = np.concatenate((pts_features_offline, new_pts_intensity), axis=0)

        return True, pts_rect, pts_features_offline, extra_gt_boxes3d, extra_gt_obj_list

    def data_augmentation(self, aug_pts_rect, aug_gt_boxes3d, gt_alpha, sample_id=None, mustaug=False, stage=1):
        """
        :param aug_pts_rect: (N, 3)
        :param aug_gt_boxes3d: (N, 7)
        :param gt_alpha: (N)
        :return:
        """
        aug_list = cfg.AUG_METHOD_LIST
        aug_enable = 1 - np.random.rand(3)
        if mustaug is True:
            aug_enable[0] = -1
            aug_enable[1] = -1
        aug_method = []
        if 'rotation' in aug_list and aug_enable[0] < cfg.AUG_METHOD_PROB[0]:
            angle = np.random.uniform(-np.pi / cfg.AUG_ROT_RANGE, np.pi / cfg.AUG_ROT_RANGE)
            aug_pts_rect = kitti_utils.rotate_pc_along_y(aug_pts_rect, rot_angle=angle)
            if stage == 1:
                # xyz change, hwl unchange
                aug_gt_boxes3d = kitti_utils.rotate_pc_along_y(aug_gt_boxes3d, rot_angle=angle)

                # calculate the ry after rotation
                x, z = aug_gt_boxes3d[:, 0], aug_gt_boxes3d[:, 2]
                beta = np.arctan2(z, x)
                new_ry = np.sign(beta) * np.pi / 2 + gt_alpha - beta
                aug_gt_boxes3d[:, 6] = new_ry  # TODO: not in [-np.pi / 2, np.pi / 2]
            elif stage == 2:
                # for debug stage-2, this implementation has little float precision difference with the above one
                assert aug_gt_boxes3d.shape[0] == 2
                aug_gt_boxes3d[0] = self.rotate_box3d_along_y(aug_gt_boxes3d[0], angle)
                aug_gt_boxes3d[1] = self.rotate_box3d_along_y(aug_gt_boxes3d[1], angle)
            else:
                raise NotImplementedError

            aug_method.append(['rotation', angle])

        if 'scaling' in aug_list and aug_enable[1] < cfg.AUG_METHOD_PROB[1]:
            scale = np.random.uniform(0.95, 1.05)
            aug_pts_rect = aug_pts_rect * scale
            aug_gt_boxes3d[:, 0:6] = aug_gt_boxes3d[:, 0:6] * scale
            aug_method.append(['scaling', scale])

        if 'flip' in aug_list and aug_enable[2] < cfg.AUG_METHOD_PROB[2]:
            # flip horizontal
            aug_pts_rect[:, 0] = -aug_pts_rect[:, 0]
            aug_gt_boxes3d[:, 0] = -aug_gt_boxes3d[:, 0]
            # flip orientation: ry > 0: pi - ry, ry < 0: -pi - ry
            if stage == 1:
                aug_gt_boxes3d[:, 6] = np.sign(aug_gt_boxes3d[:, 6]) * np.pi - aug_gt_boxes3d[:, 6]
            elif stage == 2:
                assert aug_gt_boxes3d.shape[0] == 2
                aug_gt_boxes3d[0, 6] = np.sign(aug_gt_boxes3d[0, 6]) * np.pi - aug_gt_boxes3d[0, 6]
                aug_gt_boxes3d[1, 6] = np.sign(aug_gt_boxes3d[1, 6]) * np.pi - aug_gt_boxes3d[1, 6]
            else:
                raise NotImplementedError

            aug_method.append('flip')

        return aug_pts_rect, aug_gt_boxes3d, aug_method

    def get_rcnn_sample_jit(self, index):
        sample_id = int(self.sample_id_list[index])
        rpn_xyz, rpn_features, rpn_intensity, seg_mask = \
            self.get_rpn_features(self.rcnn_training_feature_dir, sample_id)

        # load rois and gt_boxes3d for this sample
        roi_file = os.path.join(self.rcnn_training_roi_dir, '%06d.txt' % sample_id)
        roi_obj_list = kitti_utils.get_objects_from_label(roi_file)
        roi_boxes3d = kitti_utils.objs_to_boxes3d(roi_obj_list)
        # roi_scores = kitti_utils.objs_to_scores(roi_obj_list)

        gt_obj_list = self.filtrate_objects(self.get_label(sample_id))
        gt_boxes3d = kitti_utils.objs_to_boxes3d(gt_obj_list)

        sample_info = {'sample_id': sample_id,
                       'rpn_xyz': rpn_xyz,
                       'rpn_features': rpn_features,
                       'rpn_intensity': rpn_intensity,
                       'seg_mask': seg_mask,
                       'roi_boxes3d': roi_boxes3d,
                       'gt_boxes3d': gt_boxes3d,
                       'pts_depth': np.linalg.norm(rpn_xyz, ord=2, axis=1)}

        return sample_info

    def collate_batch(self, batch):
        """
        用在创建Dataloader里面，collate_fn=collate_batch
        将返回的n个字典{"a": [28,28]},{"a": [28,28]},{"a": [28,28]} ……
        转换成：{"a": [n，28,28]}
        这里是把不同shape的'gt_boxes3d'的tensor变为相同形状，然后合并成一个字典
        @param batch:
        @return:dict
        """
        if self.mode != 'TRAIN' and cfg.RCNN.ENABLED and not cfg.RPN.ENABLED:
            assert batch.__len__() == 1
            return batch[0]

        batch_size = batch.__len__()
        ans_dict = {}

        for key in batch[0].keys():
            if cfg.RPN.ENABLED and key == 'gt_boxes3d' or \
                    (cfg.RCNN.ENABLED and cfg.RCNN.ROI_SAMPLE_JIT and key in ['gt_boxes3d', 'roi_boxes3d']):
                max_gt = 0
                for k in range(batch_size):
                    max_gt = max(max_gt, batch[k][key].__len__())
                batch_gt_boxes3d = np.zeros((batch_size, max_gt, 7), dtype=np.float32)
                for i in range(batch_size):
                    batch_gt_boxes3d[i, :batch[i][key].__len__(), :] = batch[i][key]
                ans_dict[key] = batch_gt_boxes3d
                continue

            if isinstance(batch[0][key], np.ndarray):
                if batch_size == 1:
                    ans_dict[key] = batch[0][key][np.newaxis, ...]
                else:
                    ans_dict[key] = np.concatenate([batch[k][key][np.newaxis, ...] for k in range(batch_size)],
                                                   axis=0)

            else:
                ans_dict[key] = [batch[k][key] for k in range(batch_size)]
                if isinstance(batch[0][key], int):
                    ans_dict[key] = np.array(ans_dict[key], dtype=np.int32)
                elif isinstance(batch[0][key], float):
                    ans_dict[key] = np.array(ans_dict[key], dtype=np.float32)

        return ans_dict


def create_logger():
    root_result_dir = os.path.join(os.getcwd(), 'output')
    log_file = os.path.join(root_result_dir, 'log_train.txt')

    log_format = '%(asctime)s  %(levelname)5s  %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=log_format, filename=log_file)
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(logging.Formatter(log_format))
    logging.getLogger(__name__).addHandler(console)
    return logging.getLogger(__name__)


if __name__ == '__main__':
    # root_dir = r'D:\code\EPNet\data'
    DATA_PATH = 'data'

    logger = create_logger()

    train_set = KittiSSDDataset(root_dir=DATA_PATH,
                                npoints=cfg.RPN.NUM_POINTS,
                                split=cfg.TRAIN.SPLIT,
                                mode='TRAIN',
                                logger=logger,
                                classes=cfg.CLASSES)

    item0 = train_set[0]
    print(len(train_set))
