# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 22:33:04 2020

@author: Alan
"""

from __future__ import absolute_import

import os
# =============================================================================
# from got10k.datasets import *
# from got10k.experiments import *
# =============================================================================
from datasets.vid import ImageNetVID
import random

#from siamfc.ssiamfc import TrackerSiamFC 
#from siamfc.siamfc_stn import TrackerSiamFC 
from siamfc.siamfc_weight_dropping import TrackerSiamFC 
from siamfc.backbones import AlexNet
from siamfc.heads import SiamFC
from siamfc.datasets import Pair
from utils.img_loader import cv2_RGB_loader

import pickle
from tqdm import tqdm
import torch
import torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
torch.manual_seed(123456) # cpu
torch.cuda.manual_seed(123456) #gpu
np.random.seed(123456) #numpy
random.seed(123456) #random and transforms

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class Net(nn.Module):
    def __init__(self, backbone, head):
        super(Net, self).__init__()
        self.backbone = backbone
        self.head = head
    
    def forward(self, z, x):
        z = self.backbone(z)
        x = self.backbone(x)
        return self.head(z, x)


if __name__ == '__main__':
    
# =============================================================================
#     save_dir = './checkpoints/'
#     save_path = os.path.join(save_dir, 'Thesis_rewrite')
# =============================================================================
    
    net_path = './checkpoints/Thesis_rewrite/siamfc_alexnet_e50.pth'
#    save_dir = './checkpoints/stna_ft10_SGD_ccrop_wolossde_1v05_grid005'

    root_dir = 'D:\ILSVRC2015'
    seqs = ImageNetVID(root_dir, subset=['train'])
    
    net = Net(
    backbone=AlexNet(),
    head=SiamFC()).cuda()

    net.load_state_dict(torch.load(
        net_path, map_location=lambda storage, loc: storage))
    
    net.eval()
    
    transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize(255),
            torchvision.transforms.ToTensor()])

    dataset = Pair(
        seqs=seqs,
        transforms=transforms, supervised='feature', img_loader=cv2_RGB_loader)
    
    # setup dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False)
    
    feature_list = []
    for batch in tqdm(dataloader):
        img = batch[0].cuda()
        path = batch[1]
        seq_name = batch[2]
        
        feat = F.adaptive_avg_pool2d(net.backbone(img), (1, 1)).squeeze().detach().cpu().numpy()
        
        feature_list.append([feat, seq_name[0]])
        
    print(len(feature_list))
    print(len(seqs))
    pickle.dump(feature_list, open('feature_gap.pkl', 'wb'))
    