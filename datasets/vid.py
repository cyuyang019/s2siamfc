# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 22:58:34 2020

@author: Alan
"""

from __future__ import absolute_import, print_function

import os
import glob
import six
import numpy as np
import xml.etree.ElementTree as ET
import json
from collections import OrderedDict

class ImageNetVID(object):
    r"""`ImageNet Video Image Detection (VID) <https://image-net.org/challenges/LSVRC/2015/#vid>`_ Dataset.
    Publication:
        ``ImageNet Large Scale Visual Recognition Challenge``, O. Russakovsky,
            J. deng, H. Su, etc. IJCV, 2015.
    
    Args:
        root_dir (string): Root directory of dataset where ``Data``, and
            ``Annotation`` folders exist.
        subset (string, optional): Specify ``train``, ``val`` or (``train``, ``val``)
            subset(s) of ImageNet-VID. Default is a tuple (``train``, ``val``).
        cache_dir (string, optional): Directory for caching the paths and annotations
            for speeding up loading. Default is ``cache/imagenet_vid``.
    """
    def __init__(self, root_dir, subset=('train', 'val'),
                 cache_dir='cache/imagenet_vid', neg_dir=None):
        self.root_dir = root_dir
        self.cache_dir = cache_dir
        self.neg_dir = neg_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        if isinstance(subset, str):
            assert subset in ['train', 'val']
            self.subset = [subset]
        elif isinstance(subset, (list, tuple)):
            assert all([s in ['train', 'val'] for s in subset])
            self.subset = subset
        else:
            raise Exception('Unknown subset')
        
        # cache filenames and annotations to speed up training
        self.seq_dict = self._cache_meta()
        self.seq_names = [n for n in self.seq_dict]

    def __getitem__(self, index):
        r"""        
        Args:
            index (integer or string): Index or name of a sequence.
        
        Returns:
            tuple: (img_files, anno), where ``img_files`` is a list of
                file names and ``anno`` is a N x 4 (rectangles) numpy array.
        """
        seq_name = self.seq_names[index]
        
        
        if self.neg_dir:
            seq_dir, frames, cluster_id = self.seq_dict[seq_name]
            img_files = [os.path.join(seq_dir, '%06d.JPEG' % f) for f in frames]
            return img_files, seq_name, cluster_id
        
        else:
            seq_dir, frames = self.seq_dict[seq_name]
            img_files = [os.path.join(seq_dir, '%06d.JPEG' % f) for f in frames]

            return img_files, seq_name

    def __len__(self):
        return len(self.seq_dict)           #number of seq

    def _cache_meta(self):
        cache_file = os.path.join(self.cache_dir, 'seq_dict.json')
        if os.path.isfile(cache_file):
            print('Dataset already cached.')
            with open(cache_file) as f:
                seq_dict = json.load(f, object_pairs_hook=OrderedDict)
            return seq_dict
        
        if self.neg_dir:
            neg_dict = json.load(open(self.neg_dir), object_pairs_hook=OrderedDict)
        
        # image and annotation paths
        print('Gather sequence paths...')
        seq_dirs = []
        if 'train' in self.subset:
            seq_dirs_ = sorted(glob.glob(os.path.join(
                self.root_dir, 'Data/VID/train/ILSVRC*/ILSVRC*')))
            seq_dirs += seq_dirs_
        if 'val' in self.subset:
            seq_dirs_ = sorted(glob.glob(os.path.join(
                self.root_dir, 'Data/VID/val/ILSVRC2015_val_*')))
            seq_dirs += seq_dirs_
        seq_names = [os.path.basename(s) for s in seq_dirs]

        # cache paths
        print('Caching annotations to %s, ' % self.cache_dir + \
            'it may take a few minutes...')
        seq_dict = OrderedDict()

        for s, seq_name in enumerate(seq_names):
            if s % 100 == 0 or s == len(seq_names) - 1:
                print('--Caching sequence %d/%d: %s' % \
                    (s + 1, len(seq_names), seq_name))

            key = '%s' % (seq_name)
            
            
            if self.neg_dir:
                neg_cluster_id = neg_dict[seq_name]
                frames_num = len(glob.glob(seq_dirs[s] + '\*'))
                # store paths
                seq_dict.update([(key, [seq_dirs[s], list(map(int, np.arange(frames_num))), neg_cluster_id])])
            else:
                frames_num = len(glob.glob(seq_dirs[s] + '\*'))
                # store paths
                seq_dict.update([(key, [seq_dirs[s], list(map(int, np.arange(frames_num)))])])
        
        # store seq_dict
        with open(cache_file, 'w') as f:
            json.dump(seq_dict, f, indent=4)

        return seq_dict
