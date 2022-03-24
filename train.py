from __future__ import absolute_import

import os
from time import localtime, strftime


from datasets.vid import ImageNetVID
#from datasets.coco import Coco
import random

#from siamfc.ssiamfc import TrackerSiamFC 
#from siamfc.siamfc_stn import TrackerSiamFC 
from siamfc.siamfc_weight_dropping import TrackerSiamFC 

import torch
import numpy as np
torch.manual_seed(123456) # cpu
torch.cuda.manual_seed(123456) #gpu
np.random.seed(123456) #numpy
random.seed(123456) #random and transforms

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    
    save_dir = './checkpoints/'
    save_path = os.path.join(save_dir, 'S2SiamFC')
    
    neg_dir = ['./seq2neg_dict.json', './cluster_dict.json']
    root_dir = 'D:\ILSVRC2015'      #Dataset path
    seqs = ImageNetVID(root_dir, subset=['train'], neg_dir=neg_dir[0])
    
# =============================================================================
#     root_dir = 'E:\SiamMask\data\coco'
#     seqs = Coco(root_dir, subset=['train'])
# =============================================================================
    
# =============================================================================
#     save_dir = './checkpoints/eccv_rot_rcrop_mask_0515m0sig_got'
#     root_dir = 'E:/GOT10K'
#     seqs = GOT10k(root_dir, subset='train')
# =============================================================================
    
    mode = ['supervised', 'self-supervised']
    
    tracker = TrackerSiamFC(loss_setting=[0.5, 2.0, 0])
    tracker.train_over(seqs, supervised=mode[1], save_dir=save_path)

    print(strftime("%Y-%m-%d %H:%M:%S", localtime()))