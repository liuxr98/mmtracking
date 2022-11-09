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
import random
import pandas as pd

import mmcv
import numpy as np
from tqdm import tqdm
import pdb

# USELESS = [3, 4, 5, 6, 9, 10, 11]
# IGNORES = [2, 7, 8, 12, 13]
USE = [1, 2]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert VisDrone-MOT label and detections to ReID format.')
    parser.add_argument('-d', '--dir', help='path of VisDrone-MOT data')
    parser.add_argument(
        '--get-images', 
        action='store_true',
        help='corp images from frames')
    parser.add_argument(
        '--get-labels', 
        action='store_true',
        help='generate labels for classification')
    parser.add_argument(
        '--min-thr',
        type=int,
        default=8,
        help='minimum number of images for each person')
    parser.add_argument(
        '--max-thr',
        type=int,
        default=1000,
        help='maxmum number of images for each person')
    parser.add_argument(
        '--split-ratio',
        type=float,
        default=0.9,
        help='split each class/object images set, 90% for train, 10% for test')
    return parser.parse_args()


def get_images(args):
    args = parse_args()
    ann_folder = osp.join(args.dir, 'annotations')
    video_folder = osp.join(args.dir, 'sequences')
    reid_folder = osp.join(args.dir, 'reid')
    if not osp.exists(reid_folder):
        os.makedirs(reid_folder)
    videos = os.listdir(video_folder)
    columns = [
        'frame_id',
        'instance_id',
        'bbox_left',
        'bbox_top',
        'bbox_width',
        'bbox_height',
        'score',
        'object_category',
        'truncation',
        'occlusion',]
    print("corp reid images...")
    for video in tqdm(videos):
        # pdb.set_trace()
        df = pd.read_csv(osp.join(ann_folder, video+'.txt'), names=columns)
        image_folder = osp.join(video_folder, video)
        images = os.listdir(image_folder)
        images.sort()
        for image in images:
            # read image
            raw_img = mmcv.imread(f'{image_folder}/{image}')
            # found annotations in this image
            frame_id = int(image.split('.')[0])
            _df = df[df.frame_id == frame_id]
            for _, row in _df.iterrows():
                if row.score == 0 or row.object_category not in USE:
                    continue
                xyxy = np.asarray(
                    [row.bbox_left, row.bbox_top, row.bbox_left + row.bbox_width, row.bbox_top + row.bbox_height])
                reid_img = mmcv.imcrop(raw_img, xyxy)
                mmcv.imwrite(reid_img, f'{reid_folder}/{video}_{row.instance_id:06d}/{row.frame_id:06d}.jpg')

                
def get_labels(args):
    reid_dir = osp.join(args.dir, 'reid')
    assert osp.exists(reid_dir)
    reid_meta_dir = osp.join(args.dir, 'reid_meta')
    if not osp.exists(reid_meta_dir):
        os.makedirs(reid_meta_dir)
    img_dirs = os.listdir(reid_dir)
    train_labels = []
    test_labels = []
    label_id = -1
    for img_dir in tqdm(img_dirs):
        imgs = os.listdir(osp.join(reid_dir, img_dir))
        if len(imgs)<args.min_thr:
            continue
        label_id += 1
        if len(imgs)>args.max_thr:
            imgs = random.sample(imgs, args.max_thr)
        for img in imgs[:int(len(imgs)*args.split_ratio)]:
            train_labels.append(f'{osp.join(img_dir, img)} {label_id}\n')   
        for img in imgs[int(len(imgs)*args.split_ratio):]:
            test_labels.append(f'{osp.join(img_dir, img)} {label_id}\n')     
            
    with open(osp.join(reid_meta_dir, 'train_labels.txt'), 'w') as f:
        f.writelines(train_labels)
    with open(osp.join(reid_meta_dir, 'test_labels.txt'), 'w') as f:
        f.writelines(test_labels)
    print(f'num classes: {label_id+1}, lines of train_labels: {len(train_labels)}, lines of test_labels: {len(test_labels)}')
                

def main():
    args = parse_args()
    assert not (args.get_images and args.get_labels)
    if args.get_images:
        get_images(args)
    elif args.get_labels:
        get_labels(args)


if __name__ == '__main__':
    main()
