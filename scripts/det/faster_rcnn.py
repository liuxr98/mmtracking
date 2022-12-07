import mmcv
from mmdet.apis import set_random_seed
import os.path as osp

from mmtrack.datasets import build_dataset
from mmdet.apis import train_detector as train_model
from mmdet.models import build_detector as build_model

# configs
cfg = mmcv.Config.fromfile('./configs/det/faster-rcnn_visdrone.py')
cfg.seed = 0
set_random_seed(0, deterministic=False)

## work_dir
cfg.work_dir = './tutorial_exps/detector/faster_rcnn'

## GPU
cfg.device = "cuda"
cfg.gpu_ids = [0] # cfg.gpu_ids = range(1)

## dataset config
cfg.data.train.classes = ('pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor')
cfg.data.val.classes = ('pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor')
cfg.data.test.classes = ('pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor')

## model config
cfg.model.detector.roi_head.bbox_head.num_classes = 10
cfg.model.detector.init_cfg.checkpoint='http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth'

## train set
cfg.optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
cfg.lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=0.01,
    step=[3])

# runtime settings
cfg.total_epochs = 80

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