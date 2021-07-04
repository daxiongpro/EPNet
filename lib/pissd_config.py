from easydict import EasyDict as edict

__C = edict()
cfg = __C

# backbone config
# fps参数
cfg.backbone = edict()
cfg.backbone.npoints = [[4096], [1024], [256, 256]]
cfg.backbone.fps_type = [['D-FPS'], ['FS'], ['F-FPS', 'D-FPS']]
cfg.backbone.fps_range = [[-1], [-1], [256, -1]]
# groups参数
cfg.backbone.radii = [[0.2, 0.4, 0.8], [0.4, 0.8, 1.6], [1.6, 3.2, 4.8]]
cfg.backbone.nsamples = [[32, 32, 64], [32, 32, 64], [32, 32, 32]]
cfg.backbone.point_out_channels = [128, 256, 256]
# mlp参数
cfg.backbone.mlps = [[[3 + 3, 32, 32, 64], [3 + 3, 64, 64, 128], [3 + 3, 64, 96, 128]],
                     [[128 + 3, 64, 64, 128], [128 + 3, 128, 128, 256], [128 + 3, 128, 128, 256]],
                     [[256 + 3, 128, 128, 256], [256 + 3, 128, 128, 256], [256 + 3, 128, 256, 256]]]
# img channels
cfg.backbone.img_channels = [3, 64, 128, 256]

# cg_layer config
cfg.cg_layer = edict()
cfg.cg_layer.shift_mlp = [256, 128, 64, 3]
cfg.cg_layer.group_cfg = edict()
cfg.cg_layer.group_cfg.radius = 4
cfg.cg_layer.group_cfg.nsample = 32
cfg.cg_layer.group_cfg.npoint = 256
cfg.cg_layer.mlp = [256 + 3, 128, 128]

# head config
cfg.head = edict()
cfg.head.reg_mlp = [128, 64, 7]
cfg.head.cls_mlp = [128, 64, 1]
