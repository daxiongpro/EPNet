import numpy as np
import os
import pickle
import torch
from lib.datasets.kitti_dataset import KittiDataset
import lib.datasets.utils.kitti_utils as kitti_utils
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
    xy = xy * 2 / (normal_shape - 1.) - 1.
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

        self.aug_label_dir = os.path.join(aug_scene_root_dir, 'training', 'aug_label')
        self.aug_pts_dir = os.path.join(aug_scene_root_dir, 'training', 'rectified_data')
        self.gt_database = None

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
            if mode == 'TRAIN':
                self.preprocess_training_data()
            else:
                self.sample_id_list = [int(sample_id) for sample_id in self.image_idx_list]
        elif cfg.RCNN.ENABLED:
            self.sample_id_list = [int(sample_id) for sample_id in self.image_idx_list]

    def preprocess_training_data(self):
        """
        Discard丢弃 samples which don't have current classes, which will not be used for training.
        Valid sample_id is stored in self.sample_id_list
        """
        for idx in range(0, self.num_sample):
            sample_id = int(self.image_idx_list[idx])
            obj_list = self.filtrate_objects(self.get_label(sample_id))
            if len(obj_list) == 0:
                continue
            self.sample_id_list.append(sample_id)

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

    @staticmethod
    def generate_training_labels(pts_rect, gt_boxes3d):
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
            fg_pt_flag = kitti_utils.in_hull(pts_rect, box_corners)  # (N,)
            fg_pts_rect = pts_rect[fg_pt_flag]  # 在box里面的点
            cls_label[fg_pt_flag] = 1

            # enlarge the bbox3d, ignore nearby points
            extend_box_corners = extend_gt_corners[k]
            fg_enlarge_flag = kitti_utils.in_hull(pts_rect, extend_box_corners)
            ignore_flag = np.logical_xor(fg_pt_flag, fg_enlarge_flag)
            cls_label[ignore_flag] = -1

            # pixel offset of object center
            center3d = gt_boxes3d[k][0:3].copy()  # (x, y, z)
            center3d[1] -= gt_boxes3d[k][3] / 2  # y=y-h/2
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

    def __len__(self):
        return len(self.sample_id_list)
        # return len(self.image_idx_list)

    def __getitem__(self, index):
        """获取单个样本。每个值都是numpy类型。
        @param index: (int)
        @return: sample_info (dict):
        dict_keys([
        ----------------------------------------------配置参数
        'sample_id',
        'random_select',
        'aug_method',
        -----------------------------------------------图片：
        'img',
        -----------------------------------------------点云：
        'pts_origin_xy',点云在图像上对应的xy (N,2)
        'pts_input',点云原始输入(N, 3+C)，其中包含了pts_rect
        'pts_rect',点云在相机坐标系下的坐标(N, 3)
        'pts_features',输入的点云特征，初始为光照强度
        ----------------------------------------------label：
        'cls_label',分类标签
        'reg_label',回归标签(N,7)。判断这N 个点是否在box内部，若是则是这个点对应的box回归值
        'gt_boxes3d'真实回归框(M,7)
        ])
        """
        sample_info = {}  # 输出数据字典

        sample_id = int(self.sample_id_list[index])
        sample_info['sample_id'] = sample_id
        img = self.get_image_rgb_with_normal(sample_id)
        sample_info['img'] = img

        # get valid point (projected points should be in image)
        # 不做数据增强
        calib = self.get_calib(sample_id)
        pts_lidar = self.get_lidar(sample_id)
        pts_rect = calib.lidar_to_rect(pts_lidar[:, 0:3])  # 点云在相机坐标系下的坐标(N,3)
        pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)  # 点云在img上的坐标(N,2)，深度图的深度
        img_shape = self.get_image_shape(sample_id)
        pts_valid_flag = self.get_valid_flag(pts_rect, pts_img, pts_rect_depth, img_shape)
        pts_rect = pts_rect[pts_valid_flag][:, 0:3]  # 过滤掉在lidar相机上但不在img上的点
        """
        >>> a = np.array([1, 2, 3])
        >>> b = np.array([ True, False,  True])
        >>> a[b]
        array([1, 3])
        """
        # 选取16384个点  # make sure len(pts_rect) ==self.npoints
        if self.npoints < len(pts_rect):
            pts_depth = pts_rect[:, 2]  # (N,1)
            pts_near_flag = pts_depth < 40.0
            far_idxs_choice = np.where(pts_near_flag == 0)[0]  # 远的点都要
            near_idxs = np.where(pts_near_flag == 1)[0]
            near_idxs_choice = np.random.choice(near_idxs, self.npoints - len(far_idxs_choice),
                                                replace=False)  # 近的点选取部分

            choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0) \
                if len(far_idxs_choice) > 0 else near_idxs_choice
            np.random.shuffle(choice)  # 打乱
        else:  # 点的数量不够16384
            choice = np.arange(0, len(pts_rect), dtype=np.int32)  # 原来的点全要，还要再补一些点
            extra_choice = np.random.choice(choice, self.npoints - len(pts_rect),
                                            replace=False)  # 随机从原来的点中取一些点补上
            choice = np.concatenate((choice, extra_choice), axis=0)
            np.random.shuffle(choice)

        pts_origin_xy = pts_img[pts_valid_flag]  # 点云在img上的坐标
        sample_info['pts_origin_xy'] = pts_origin_xy[choice, :]  # 16384,2
        ret_pts_rect = pts_rect[choice, :]  # (N,3)

        if self.mode == 'TEST':  # 如果是TEST，则不用数据增强和标签
            sample_info['pts_input'] = ret_pts_rect
            sample_info['pts_rect'] = ret_pts_rect
            sample_info['pts_features'] = None
            return sample_info
        elif self.mode == 'TRAIN':
            # 数据增强
            gt_obj_list = self.filtrate_objects(self.get_label(sample_id))
            gt_boxes3d = kitti_utils.objs_to_boxes3d(gt_obj_list)
            gt_alpha = np.zeros((gt_obj_list.__len__()), dtype=np.float32)  # 旋转角度，都为0，占位
            for k, obj in enumerate(gt_obj_list):
                gt_alpha[k] = obj.alpha  # 给占位赋值
            aug_pts_rect, aug_gt_boxes3d, aug_method = self.data_augmentation(ret_pts_rect.copy(),
                                                                              gt_boxes3d.copy(),
                                                                              gt_alpha,
                                                                              sample_id)
            sample_info['pts_rect'] = aug_pts_rect  # 点云在相机坐标系下坐标 pts_rect: (N, 3)
            sample_info['gt_boxes3d'] = aug_gt_boxes3d
            sample_info['aug_method'] = aug_method

            pts_input = aug_pts_rect  # 本来是要加上强度的，本代码删除了所有强度
            sample_info['pts_input'] = pts_input  # xyz_intensity坐标
            sample_info['pts_features'] = None

            # 获取标签
            cls_label, reg_label = self.generate_training_labels(aug_pts_rect, aug_gt_boxes3d)
            sample_info['cls_label'] = cls_label  # (16384,)
            sample_info['reg_label'] = reg_label  # (16384, 7)

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
        if self.mode != 'TRAIN':
            assert batch.__len__() == 1
            return batch[0]

        batch_size = batch.__len__()
        ans_dict = {}

        for key in batch[0].keys():
            if key in ['gt_boxes3d', 'roi_boxes3d']:
                max_gt = 0
                for k in range(batch_size):
                    max_gt = max(max_gt, batch[k][key].__len__())
                batch_gt_boxes3d = np.zeros((batch_size, max_gt, 7), dtype=np.float32)
                for i in range(batch_size):
                    batch_gt_boxes3d[i, :batch[i][key].__len__(), :] = batch[i][key]
                ans_dict[key] = batch_gt_boxes3d

            elif isinstance(batch[0][key], np.ndarray):
                if batch_size == 1:
                    ans_dict[key] = batch[0][key][np.newaxis, ...]
                else:
                    ans_dict[key] = np.concatenate([batch[k][key][np.newaxis, ...] for k in range(batch_size)],
                                                   axis=0)

            else:  # True等非数字类型的
                ans_dict[key] = [batch[k][key] for k in range(batch_size)]
                if isinstance(batch[0][key], int):
                    ans_dict[key] = np.array(ans_dict[key], dtype=np.int32)
                elif isinstance(batch[0][key], float):
                    ans_dict[key] = np.array(ans_dict[key], dtype=np.float32)

            if isinstance(ans_dict[key], np.ndarray):  # 将所有的key转换成tensor
                ans_dict[key] = torch.from_numpy(ans_dict[key])

        return ans_dict


if __name__ == '__main__':
    # root_dir = r'D:\code\EPNet\data'
    DATA_PATH = 'data'
    train_set = KittiSSDDataset(root_dir=DATA_PATH,
                                npoints=cfg.RPN.NUM_POINTS,
                                split=cfg.TRAIN.SPLIT,
                                mode='TRAIN',
                                classes=cfg.CLASSES)

    item0 = train_set[0]
    # print(item0)
    for key in item0.keys():
        if type(item0[key]) is np.ndarray:
            print(key, ":", "(numpy.ndarray)", item0[key].shape)
        else:
            print(key, ":", item0[key])

    # img = item0['img']
    # pts_origin_xy = item0['pts_origin_xy']
    # aug_method = item0['aug_method']
    # pts_input = item0['pts_input']
    # pts_rect = item0['pts_rect']
    # pts_features = item0['pts_features']
    # cls_label = item0['cls_label']
    # reg_label = item0['reg_label']
    # gt_boxes3d = item0['gt_boxes3d']

    # pts_input = torch.tensor(pts_input).cuda()
    # net = PointnetSAModuleMSG(
    #     npoint=4096,  # [4096, 1024, 256, 64]
    #     radii=[0.1, 0.5],  # [[0.1, 0.5], [0.5, 1.0], [1.0, 2.0], [2.0, 4.0]]
    #     nsamples=[16, 32],  # [[16, 32], [16, 32], [16, 32], [16, 32]]
    #     mlps=[[1, 16, 32], [1, 16, 32]],  # mlps = cfg.RPN.SA_CONFIG.MLPS[k].copy()
    #     use_xyz=True,
    #     bn=True
    # ).cuda()
    #
    # B = 2
    # N = 16384
    # C = 16
    # xyz = pts[..., 0:3].contiguous().unsqueeze(0)
    # features = pts[..., 3:].contiguous().unsqueeze(0).transpose(1, 2)
    # new_xyz = None
    #
    # out = net(xyz, features, new_xyz)
