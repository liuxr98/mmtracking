import mmcv
from mmdet.apis import set_random_seed
import os.path as osp
from mmtrack.datasets import build_dataset
from mmdet.apis import train_detector as train_model
from mmdet.models import build_detector as build_model


cfg = mmcv.Config.fromfile('./configs/det/faster-rcnn_r50_fpn_4e_visdrone_mot.py')
cfg.work_dir = './tutorial_exps/detector_faster_rcnn'
cfg.seed = 0
cfg.model.detector.roi_head.bbox_head.num_classes = 1
set_random_seed(0, deterministic=False)
cfg.device = "cuda"
# cfg.gpu_ids = range(1)
cfg.gpu_ids = [0]
print(f'Config:\n{cfg.pretty_text}')


mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
model = build_model(cfg.model.detector)
model.init_weights()
datasets = [build_dataset(cfg.data.train)]
model.CLASSES = datasets[0].CLASSES
train_model(model, datasets, cfg, validate=True)