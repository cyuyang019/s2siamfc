from __future__ import absolute_import

import os
import glob
import numpy as np
import cv2

from got10k.datasets.vot import VOT
import random

from siamfc.siamfc_weight_dropping import TrackerSiamFC

import torch
torch.manual_seed(123456) # cpu
torch.cuda.manual_seed(123456) #gpu
np.random.seed(123456) #numpy
random.seed(123456) #random and transforms

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def _corner2rect(corners, center=False):
    cx = np.mean(corners[0::2])
    cy = np.mean(corners[1::2])

    x1 = np.min(corners[0::2])
    x2 = np.max(corners[0::2])
    y1 = np.min(corners[1::2])
    y2 = np.max(corners[1::2])

    area1 = np.linalg.norm(corners[0:2] - corners[2:4]) * \
        np.linalg.norm(corners[2:4] - corners[4:6])
    area2 = (x2 - x1) * (y2 - y1)
    scale = np.sqrt(area1 / area2)
    w = scale * (x2 - x1) + 1
    h = scale * (y2 - y1) + 1

    if center:
        return np.array([cx, cy, w, h]).T
    else:
        return np.array([cx - w / 2, cy - h / 2, w, h]).T

if __name__ == '__main__':
    save_dir = './checkpoints/'
    save_path = os.path.join(save_dir, 'finetune_result')
    print("cuda: ", torch.cuda.is_available())
    root_dir = '/Users/chenyuyang/Desktop/VOT2016'      #Dataset path
    track_item = 'racing'
    seqs = VOT(root_dir, version=2016, download=False)

    # finetuning
    tracker = TrackerSiamFC(loss_setting=[0.5, 2.0, 0], net_path = './checkpoints/S2SiamFC/siamfc_alexnet_e50.pth')
    tracker.finetune(seqs, save_dir=save_path, finetune_target=track_item)

    # illustrate
    seq_dir = os.path.join(root_dir, track_item) + "/"
    seq_dir = os.path.expanduser(seq_dir)
    print(seq_dir)
    img_files = sorted(glob.glob(seq_dir + '*.jpg'))

    groundtruth_array = []
    with open(seq_dir + '/groundtruth.txt') as f:
        while True:
            ann = f.readline()
            if ann == '':
                break
            ann = ann.replace(',', ' ').split()
            ann = np.array(ann, dtype=np.float64)
            ann = _corner2rect(ann)
            groundtruth_array.append(ann)
    anno = groundtruth_array[0]
    #print(len(img_files))
    tracker.track(img_files, anno, visualize=True, groundtruth_array=groundtruth_array)
    cv2.destroyAllWindows()