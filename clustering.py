# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 16:09:45 2019

@author: Alan
"""

import pickle
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
import json


feat_file = open('feature_gap.pkl', 'rb')

feat_list = np.asarray(pickle.load(feat_file))
feat_file.close()

feat = np.asarray([feat_list[:, 0][i] for i in range(len(feat_list[:, 0]))])
path = np.asarray([str(feat_list[:, 1][i]) for i in range(len(feat_list[:, 1]))])

#clustering = DBSCAN(eps=0.5, min_samples=10).fit(feat)
clustering = KMeans(n_clusters=100, random_state=123456, init='k-means++').fit(feat)
c = clustering.labels_

seq_dict = dict()
cluster_dict = dict()

for cid, p in zip(c, path):
    seq_dict[p] = int(cid)

for i in range(100):
    cluster_dict[i] = [p[0] for p in path[np.argwhere(c==i)]]

with open('seq2neg_dict.json', 'w') as f:
    json.dump(seq_dict, f, indent=4)
    
with open('cluster_dict.json', 'w') as f:
    json.dump(cluster_dict, f, indent=4)

#with open('cluster_dict.json') as f:
#    cluster_dict = json.load(f, object_pairs_hook=dict)    
#
#
#def get_key(val): 
#    for key, value in cluster_dict.items(): 
#        try:
#            if value.index(val): 
#                return key
#        except:
#            pass