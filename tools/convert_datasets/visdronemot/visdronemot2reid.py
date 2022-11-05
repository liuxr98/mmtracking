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
        '--min-per-person',
        type=int,
        default=8,
        help='minimum number of images for each person')
    parser.add_argument(
        '--max-per-person',
        type=int,
        default=1000,
        help='maxmum number of images for each person')
    return parser.parse_args()


# def parse_annotation(annotation):
#     annotation = annotation.strip().split(',')
#     frame_id, instance_id = map(int, annotation[:2])
#     bbox = list(map(float, annotation[2:6]))
#     score = int(annotation[6])
#     category_id = int(annotation[7])
#     output = dict(
#         frame_id = frame_id,
#         instance_id = instance_id,
#         bbox = bbox,
#         score = score,
#         category_id = category_id,
#     )
#     return output


def main():
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
        # with open(osp.join(ann_path, video+'.txt')) as f:
        #     annotations = f.readlines()
        # annotations = list(map(parse_annotation, annotations))
        # annotations.sort(key=lambda x: x.frame_id)
        image_folder = osp.join(video_folder, video)
        images = os.listdir(image_folder)
        images.sort()
        for image in images:
            # read image
            raw_img = mmcv.imread(
                f'{image_folder}/{image}')
            # found annotations in this image
            frame_id = int(image.split('.')[0])
            _df = df[df.frame_id == frame_id]
            for _, row in _df.iterrows():
                if row.score == 0 or row.object_category not in USE:
                    continue
                xyxy = np.asarray(
                    [row.bbox_left, row.bbox_top, row.bbox_left + row.bbox_width, row.bbox_top + row.bbox_height])
                reid_img = mmcv.imcrop(raw_img, xyxy)
                mmcv.imwrite(reid_img,
                             f'{reid_folder}/{video}_{row.instance_id:06d}/{row.frame_id:06d}.jpg')

    # generate label infos
    print("generate reid labels...")
    reid_meta_folder = osp.join(args.dir, 'reid_meta')
    if not osp.exists(reid_meta_folder):
        os.makedirs(reid_meta_folder)
    reid_img_folder_names = sorted(os.listdir(reid_folder))
    train_label, val_label = 0, 0
    random.seed(0)
    reid_dataset_list = []
    for reid_img_folder_name in reid_img_folder_names:
        reid_img_names = os.listdir(
            f'{reid_folder}/{reid_img_folder_name}')
        # ignore ids whose number of image is less than min_per_person
        if len(reid_img_names) < args.min_per_person:
            continue
        # down-sampling when there are too many images owned by one id
        if len(reid_img_names) > args.max_per_person:
            reid_img_names = random.sample(reid_img_names, args.max_per_person)
        # training set
        for reid_img_name in reid_img_names:
            reid_dataset_list.append(
                f'{reid_img_folder_name}/{reid_img_name} {train_label}\n')
        train_label += 1

    with open(osp.join(reid_meta_folder, 'train.txt'), 'w') as f:
        f.writelines(reid_dataset_list)



if __name__ == '__main__':
    main()
