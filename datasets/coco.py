# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 17:31:15 2020

@author: Alan
"""

# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import os
import glob
import six
import numpy as np
import xml.etree.ElementTree as ET
import json
from collections import OrderedDict

class Coco(object):
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
                 cache_dir='cache/coco', neg_dir=None):
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
        self.img_dict = self._cache_meta()
        self.img_names = [n for n in self.img_dict]

    def __getitem__(self, index):
        r"""        
        Args:
            index (integer or string): Index or name of a sequence.
        
        Returns:
            tuple: (img_files, anno), where ``img_files`` is a list of
                file names and ``anno`` is a N x 4 (rectangles) numpy array.
        """
        img_name = self.img_names[index]
        
        
        if self.neg_dir:
            img_path, frames, cluster_id = self.img_dict[img_name]
            return [img_path], img_name, cluster_id
        
        else:
            img_path, frames = self.img_dict[img_name]
            return [img_path], img_name

    def __len__(self):
        return len(self.img_dict)           #number of seq

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
        img_paths = []
        if 'train' in self.subset:
            img_path_ = sorted(glob.glob(os.path.join(
                self.root_dir, 'train2017/*')))
            img_paths += img_path_
        if 'val' in self.subset:
            img_path_ = sorted(glob.glob(os.path.join(
                self.root_dir, 'val2017/*')))
            img_paths += img_path_
        
        img_names = [os.path.splitext(os.path.basename(s))[0] for s in img_paths]
        # cache paths
        seq_dict = OrderedDict()

        for s, img_name in enumerate(img_names):
            if s % 100 == 0 or s == len(img_names) - 1:
                print('--Caching sequence %d/%d: %s' % \
                    (s + 1, len(img_names), img_name))

            key = '%s' % (img_name)
            
            if self.neg_dir:
                neg_cluster_id = neg_dict[img_name]
                # store paths
                seq_dict.update([(key, [img_paths[s], [0], neg_cluster_id])])
            else:
                # store paths
                seq_dict.update([(key, [img_paths[s], [0]])])
        
        # store seq_dict
        with open(cache_file, 'w') as f:
            json.dump(seq_dict, f, indent=4)

        return seq_dict
