from __future__ import absolute_import

import os
import glob
import numpy as np
import cv2

#from siamfc.siamfc_gradcam import TrackerSiamFC
#from siamfc import TrackerSiamFC
#from siamfc.siamfc_linear import TrackerSiamFC
from siamfc.siamfc_weight_dropping import TrackerSiamFC 


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
    seq_dir = os.path.expanduser('E:/SiamMask/data/VOT2018/bag/')
    img_files = sorted(glob.glob(seq_dir + '*.jpg'))

    with open(seq_dir + 'groundtruth.txt') as f:
        anno = f.readline()
        anno = anno.replace(',', ' ').split()
        anno = np.array(anno, dtype=np.float)       
        anno = _corner2rect(anno)
#        anno[0:2] += 20    
    
#    img_files = sorted(glob.glob(seq_dir + 'img/*.jpg'))
#    anno = np.loadtxt(seq_dir + 'groundtruth_rect.txt')[0]

# =============================================================================
#     seq_dir = os.path.expanduser('D:/UDT_pytorch/track/dataset/OTB2015/Coke/')   
#     img_files = sorted(glob.glob(seq_dir + 'img/*.jpg'))
#     with open(seq_dir + 'groundtruth_rect.txt') as f:
#         anno = f.readline()
#         anno = anno.replace(',', ' ').split()
#         anno = np.asarray(anno, dtype=np.float)
# =============================================================================
    

    
#    net_path = 'pretrained_base_wolrnrom/siamfc_alexnet_e50.pth'
    net_path = './checkpoints/Thesis_rewrite/siamfc_alexnet_e50.pth'
    tracker = TrackerSiamFC(net_path=net_path)
    tracker.track(img_files, anno, visualize=True)
    cv2.destroyAllWindows()
#    score = tracker.score_dict