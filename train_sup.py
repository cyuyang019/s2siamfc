from __future__ import absolute_import

import os
from got10k.datasets import *

import random

#from siamfc.ssiamfc import TrackerSiamFC 
#from siamfc.siamfc_stn import TrackerSiamFC 
#from siamfc.siamfc_weight_dropping import TrackerSiamFC 
from siamfc.siamfc import TrackerSiamFC 

import torch
import numpy as np
torch.manual_seed(123456) # cpu
torch.cuda.manual_seed(123456) #gpu
np.random.seed(123456) #numpy
random.seed(123456) #random and transforms

# =============================================================================
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# =============================================================================

if __name__ == '__main__':

    save_dir = './checkpoints/SiamFC_bn'

    root_dir = 'D:\ILSVRC2015'
    seqs = ImageNetVID(root_dir, subset=['train'])          #remember modify code
    
    mode = ['supervised', 'single-unsupervised']
    
    tracker = TrackerSiamFC()
    tracker.train_over(seqs, supervised=mode[0], save_dir=save_dir)

