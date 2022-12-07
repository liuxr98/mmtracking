import mmcv
from mmdet.apis import set_random_seed
import os.path as osp

from mmtrack.datasets import build_dataset
from mmdet.apis import train_detector as train_model
from mmdet.models import build_detector as build_model

# configs
cfg = mmcv.Config.fromfile('./configs/det/centernet_visdrone.py')
cfg.seed = 0
set_random_seed(0, deterministic=False)

## work_dir
cfg.work_dir = './tutorial_exps/detector/centernet'

## GPu
cfg.device = "cuda"
cfg.gpu_ids = [1] # cfg.gpu_ids = range(1)

## dataset config
cfg.data.train.classes = ('pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor')
cfg.data.val.classes = ('pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor')
cfg.data.test.classes = ('pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor')

## model config
cfg.model.detector.bbox_head.num_classes = 10

# optimizer
# Based on the default settings of modern detectors, the SGD effect is better
# than the Adam in the source code, so we use SGD default settings and
# if you use adam+lr5e-4, the map is 29.1.
# cfg.optimizer_config = dict(
#     _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
cfg.optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# learning policy
# Based on the default settings of modern detectors, we added warmup settings.
cfg.lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 1000,
    step=[18, 24])  # the real step is [18*5, 24*5]

# epoch
cfg.runner = dict(type='EpochBasedRunner', max_epochs=140)

# batch_size
cfg.auto_scale_lr = dict(base_batch_size=32)

print(f'Config:\n{cfg.pretty_text}')

# run and test
mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
model = build_model(cfg.model.detector)
print(model.eval())
model.train()
model.init_weights()
datasets = [build_dataset(cfg.data.train)]
model.CLASSES = datasets[0].CLASSES
train_model(model, datasets, cfg, validate=True)