# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 00:46:58 2020

@author: Alan
"""

import cv2 

def cv2_RGB_loader(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)