# Copyright (c) DCLab. All rights reserved.
# This script converts VisDrone-MOT dataset into ReID dataset.
# Official website of the VisDrone-MOT dataset: https://github.com/VisDrone/VisDrone-Dataset
#
# File structure of VisDrone-MOT dataset:
#   annotations:
#       uav{%7d}_{%5d}.txt
#       ...
#   sequences:
#       uav{%7d}_{%5d}:
#           %7d.jpg
#           ...
#
#
# Label format in each uav{%7d}_{%5d}.txt:
#       <frame_id> # starts from 1,
#       <instance_id>,
#       <bbox_left>, <bbox_top>, <bbox_width>, <bbox_height>,
#       <conf> # conf is annotated as 0 if the object is ignored,
#       <class_id>, # 0~11
#       <truncation>,
#       <occlusion>
#
#
# Classes in VisDrone-MOT:
#   0: 'ignored regions'
#   1: 'pedestrian'
#   2: 'people'
#   3: 'bicycle'
#   4: 'car'
#   5: 'van'
#   6: 'truck'
#   7: 'tricycle'
#   8: 'awning-tricycle'
#   9: 'bus'
#   10: 'motor',
#   11: 'others'
#
#   USELESS classes and IGNORES classes will not be selected
#   into the dataset for reid model training.
import argparse
import os
import os.path as osp
from collections import defaultdict

import mmcv
import numpy as np
from tqdm import tqdm

# USELESS = [3, 4, 5, 6, 9, 10, 11]
# IGNORES = [2, 7, 8, 12, 13]
USE = [1, 2]

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert VisDrone-MOT label and detections to COCO-VID format.')
    parser.add_argument('-d', '--dir', help='path of VisDrone-MOT data')
    return parser.parse_args()


def parse_annotation(annotation):
    annotation = annotation.strip().split(',')
    frame_id, instance_id = map(int, annotation[:2])
    bbox = list(map(float, annotation[2:6]))
    area = bbox[2]*bbox[3]
    score = int(annotation[6])
    category_id = int(annotation[7])
    output = dict(
        bbox = bbox,
        area = area,
        category_id = category_id,
        instance_id = instance_id,
        iscrowd = False,
    )
    return score==1, frame_id, output


def main():
    args = parse_args()
    vid_id, img_id, ann_id = 1, 1, 1
    ins_id = 0

    ann_path = osp.join(args.dir, 'annotations')
    seq_path = osp.join(args.dir, 'sequences')

    videos = os.listdir(seq_path)
    outputs = defaultdict(list)
    outputs['categories'] = [
        dict(id=1, name='pedestrian'),
        dict(id=2, name='people'),
    ]
    query_id = dict()
    for video in tqdm(videos):
        # update videos part
        outputs["videos"].append(dict(
            id = vid_id,
            name = osp.join('sequences', video),
            fps = 30,
            width = 1344,
            height = 756,
        ))
        # update images part
        images = os.listdir(osp.join(seq_path, video))
        images = sorted(images)
        for frame_id, image in enumerate(images):
            outputs['images'].append(dict(
                id = img_id,
                file_name = osp.join(osp.join('sequences', video), image),
                width = 1344,
                height = 756,
                frame_id = frame_id,
                video_id = vid_id,
            ))
            query_id[osp.join(osp.join('sequences', video), image)] = img_id
            img_id += 1
        # update annotations part
        with open(osp.join(ann_path, video+'.txt')) as f:
            annotations = f.readlines()
        for annotation in annotations:
            ok, frame_id, ans = parse_annotation(annotation)
            if ok:
                ans.update(dict(
                    id = ann_id,
                    image_id = query_id[osp.join(osp.join('sequences', video), "%07d.jpg" % frame_id)],
                    video_id = vid_id,
                ))
                outputs['annotations'].append(ans)
                ann_id += 1
            else:
                continue

        vid_id += 1

    mmcv.dump(outputs, osp.join(args.dir, 'cocoformat.json'))


if __name__ == '__main__':
    main()
