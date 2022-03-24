# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 13:50:12 2020

@author: Alan
"""

from __future__ import absolute_import, division

import numpy as np
import cv2
from torch.utils.data import Dataset

__all__ = ['Pair']

class Single_image_ft(Dataset):

    def __init__(self, img, transforms=None, pairs_per_img=1, supervised='unsupervised', box=None):
        super(Single_image_ft, self).__init__()
        self.img = img
        self.transforms = transforms
        self.pairs_per_img = pairs_per_img
        self.supervised = supervised
        if supervised == 'supervised':
            self.box = box
            self.box = np.array([
            box[1] - 1 + (box[3] - 1) / 2,
            box[0] - 1 + (box[2] - 1) / 2,
            box[3], box[2]], dtype=np.float32)
            self.target_pos, self.target_sz = box[:2], box[2:]


    def __getitem__(self, index):
        # get filename lists and annotations
        neg = False
        img_files = self.img
        
# =============================================================================
#         # filter out noisy frames
#         if self.supervised == 'supervised':
#             neg = self.neg and self.neg > np.random.rand()
#             val_indices = self._filter(
#                 cv2.imread(img_files[0], cv2.IMREAD_COLOR),
#                 anno, vis_ratios)
#             
#             if len(val_indices) < 2:
#                 index = np.random.choice(len(self))
#                 return self.__getitem__(index)
#     
#             # sample a frame pair
#             rand_z, rand_x = self._sample_pair(val_indices)
#             z = cv2.imread(img_files[rand_z], cv2.IMREAD_COLOR)
#             x = cv2.imread(img_files[rand_x], cv2.IMREAD_COLOR)
#             z = cv2.cvtColor(z, cv2.COLOR_BGR2RGB)
#             x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
#             
#             box_z = anno[rand_z]
#             box_x = anno[rand_x]
#             
#             item = (z, x, box_z, box_x)
#             if self.transforms is not None:
#                 item = self.transforms(*item)
#             
#             return item + (neg, )
# =============================================================================

        z = np.array(img_files)     #RGB
#        z = cv2.imread(img_files, cv2.IMREAD_COLOR)
#            x = cv2.imread(img_files[random_fid], cv2.IMREAD_COLOR)
        z = cv2.cvtColor(z, cv2.COLOR_RGB2BGR)
#            x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        
        if self.supervised == 'supervised':
            box = self._cxy_wh_2_bbox(self.target_pos, self.target_sz)
            item = (z, z, box, box)
        else:
            img_h, img_w, _ = z.shape
    #                target_pos = [img_w//2, img_h//2]
    #                target_sz = [img_w//6, img_h//6]
    
            target_sz = [img_w//np.random.randint(4, 9), img_h//np.random.randint(4, 9)]
            target_pos = [np.random.randint(target_sz[0], (img_w-target_sz[0])), np.random.randint(target_sz[1], (img_h-target_sz[1]))]
    
    
            box = self._cxy_wh_2_bbox(target_pos, target_sz)
            
            item = (z, z, box, box)
        if self.transforms is not None:
            item = self.transforms(*item)
               
        return item + (neg, )

    
    def __len__(self):
        return 1
    
    def _cxy_wh_2_bbox(self, cxy, wh):
        return np.array([cxy[0] - wh[0] / 2, cxy[1] - wh[1] / 2, cxy[0] + wh[0] / 2, cxy[1] + wh[1] / 2])


