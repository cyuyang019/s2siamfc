from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import cv2
import sys
import os
from collections import namedtuple
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from got10k.trackers import Tracker
from utils.average_meter_helper import AverageMeter
import torchvision
import math

from . import ops
from .backbones import AlexNetV1
from .heads import SiamFC
from .losses import BalancedLoss
from .datasets import Pair
from .transforms import SiamFCTransforms
from utils.lr_helper import build_lr_scheduler

__all__ = ['TrackerSiamFC']


class Net(nn.Module):

    def __init__(self, backbone, head):
        super(Net, self).__init__()
        self.backbone = backbone
        self.head = head
    
    def forward(self, z, x):
        z = self.backbone(z)
        x = self.backbone(x)
        return self.head(z, x)

class TrackerSiamFC(Tracker):

    def __init__(self, net_path=None, name='SiamFC', **kwargs):
        super(TrackerSiamFC, self).__init__(name, True)
        self.cfg = self.parse_args(**kwargs)

        # setup GPU device if available
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')

        # setup model
        self.net = Net(
            backbone=AlexNetV1(),
            head=SiamFC(self.cfg.out_scale))
        ops.init_weights(self.net)
        
        # load checkpoint if provided
        if net_path is not None:
            self.net.load_state_dict(torch.load(
                net_path, map_location=lambda storage, loc: storage))
        self.net = self.net.to(self.device)

        # setup criterion
        self.criterion = BalancedLoss()

        # setup optimizer
        self.optimizer = optim.SGD(
            self.net.parameters(),
            lr=self.cfg.initial_lr,
            weight_decay=self.cfg.weight_decay,
            momentum=self.cfg.momentum)
        
        # setup lr scheduler
        gamma = np.power(
            self.cfg.ultimate_lr / self.cfg.initial_lr,
            1.0 / self.cfg.epoch_num)
        self.lr_scheduler = ExponentialLR(self.optimizer, gamma)
#        self.lr_scheduler = build_lr_scheduler(self.optimizer, True, self.cfg.initial_lr, self.cfg.ultimate_lr, self.cfg.epoch_num)
        
        self.score_dict = {0:[], 1:[], 2:[]}

    def parse_args(self, **kwargs):
        # default parameters
        cfg = {
            # basic parameters
            'out_scale': 0.001,
            'exemplar_sz': 127,
            'instance_sz': 255,
            'context': 0.5,
            # inference parameters
            'scale_num': 3,
            'scale_step': 1.0375,
#            'scale_step': 1.0375,
            'scale_lr': 0.59,
            'scale_penalty': 0.9745,
            'window_influence': 0.176,
            'response_sz': 17,
            'response_up': 16,
            'total_stride': 8,
            # train parameters
            'epoch_num': 50,
            'batch_size': 8,
            'num_workers': 3,
            'initial_lr': 1e-2,
            'ultimate_lr': 1e-5,
            'weight_decay': 5e-4,
            'momentum': 0.9,
            'r_pos': 16,
            'r_neg': 0}
        
        for key, val in kwargs.items():
            if key in cfg:
                cfg.update({key: val})
        return namedtuple('Config', cfg.keys())(**cfg)
    
    @torch.no_grad()
    def init(self, img, box):
        # set to evaluation mode
        self.net.eval()

        # convert box to 0-indexed and center based [y, x, h, w]
        box = np.array([
            box[1] - 1 + (box[3] - 1) / 2,
            box[0] - 1 + (box[2] - 1) / 2,
            box[3], box[2]], dtype=np.float32)
        self.center, self.target_sz = box[:2], box[2:]

        # create hanning window
        self.upscale_sz = self.cfg.response_up * self.cfg.response_sz
        self.hann_window = np.outer(
            np.hanning(self.upscale_sz),
            np.hanning(self.upscale_sz))
        self.hann_window /= self.hann_window.sum()

        # search scale factors
        self.scale_factors = self.cfg.scale_step ** np.linspace(
            -(self.cfg.scale_num // 2),
            self.cfg.scale_num // 2, self.cfg.scale_num)

        # exemplar and search sizes
        context = self.cfg.context * np.sum(self.target_sz)
        self.z_sz = np.sqrt(np.prod(self.target_sz + context))
        self.x_sz = self.z_sz * \
            self.cfg.instance_sz / self.cfg.exemplar_sz
        
        # exemplar image
        self.avg_color = np.mean(img, axis=(0, 1))
        z = ops.crop_and_resize(
            img, self.center, self.z_sz,
            out_size=self.cfg.exemplar_sz,
            border_value=self.avg_color)            #return cv2
       
        z = cv2.normalize(z, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        

        
        # exemplar features
        self.z = torch.from_numpy(z).to(
            self.device).permute(2, 0, 1).unsqueeze(0).float()
        
        
        self.kernel = self.net.backbone(self.z)

    @torch.no_grad()
    def update(self, img, fid=None, visualize=True):
        # set to evaluation mode
        self.net.eval()
        
        # search images
        x = [ops.crop_and_resize(
            img, self.center, self.x_sz * f,
            out_size=self.cfg.instance_sz,
            border_value=self.avg_color) for f in self.scale_factors]
        x_img = [cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) for img in x]       #BGR
        x = np.stack(x_img, axis=0)
        x_tensor = torch.from_numpy(x).to(
            self.device).permute(0, 3, 1, 2).float()
        
        # responses
        x = self.net.backbone(x_tensor)
        
        responses = self.net.head(self.kernel, x)               
        
        responses = responses.squeeze(1).cpu().numpy()
        def sigmoid(x):
            s = 1 / (1 + np.exp(-x))
            return s
#        print(sigmoid(responses[0]).shape)
#        print(sigmoid(responses[0])[sigmoid(responses[0])>0.5].shape)
#        print(sigmoid(responses[0]).max())
#        print(sigmoid(responses[0]).min())
#        print(sigmoid(responses[0]).mean())
#        raise ''
        
        # upsample responses and penalize scale changes
        responses_ori = np.stack([cv2.resize(
            u, (self.upscale_sz, self.upscale_sz),
            interpolation=cv2.INTER_CUBIC)
            for u in responses])
        responses = responses_ori.copy()
        responses[:self.cfg.scale_num // 2] *= self.cfg.scale_penalty
        responses[self.cfg.scale_num // 2 + 1:] *= self.cfg.scale_penalty

        # peak scale
        scale_id = np.argmax(np.amax(responses, axis=(1, 2)))

        # peak location
        response = responses[scale_id]
#        print(np.sum(sigmoid(response) > 0.8)/np.prod(response.shape))
#        raise
        if False:
#            print(np.max(responses_ori[scale_id]))
#            score_map = cv2.normalize(responses_ori[scale_id], None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            if not os.path.exists('./test/demo_template'):
                os.makedirs('./test/demo_template')

            torchvision.utils.save_image(self.z, './test/demo_template/%d.jpg'%fid,  normalize=True)

            for sid in range(self.cfg.scale_num):
                score_map = sigmoid(responses_ori[sid])
                self.score_dict[sid].append([score_map.max(), score_map.min(), score_map.mean(), score_map.std()])
                score_map[score_map <= 0] = 0
                if not os.path.exists('./test/demo_search_tensor_%d'%(sid)):
                    os.makedirs('./test/demo_search_tensor_%d'%(sid))
                if not os.path.exists("./test/demo_score_map_%d"%(sid)):
                    os.makedirs("./test/demo_score_map_%d"%(sid))
                cv2.imwrite("./test/demo_score_map_%d/score_map_%d.jpg"%(sid,fid), np.uint8(255 * score_map))
                torchvision.utils.save_image(x_tensor[sid], './test/demo_search_tensor_%d/%d.jpg'%(sid,fid),  normalize=True)
                
#            torchvision.utils.save_image(x_tensor[scale_id], './test/demo_search_tensor/%d_%f.jpg'%(fid, np.max(responses_ori[scale_id])))

#            cv2.imwrite("./test/demo_search/%d.jpg"%fid, np.asarray(x_img[scale_id]*255, dtype=np.uint8))
        
        response -= response.min()
        response /= response.sum() + 1e-16

        response = (1 - self.cfg.window_influence) * response + \
            self.cfg.window_influence * self.hann_window
            
            
        loc = np.unravel_index(response.argmax(), response.shape)

        # locate target center
        disp_in_response = np.array(loc) - (self.upscale_sz - 1) / 2
        disp_in_instance = disp_in_response * \
            self.cfg.total_stride / self.cfg.response_up
        disp_in_image = disp_in_instance * self.x_sz * \
            self.scale_factors[scale_id] / self.cfg.instance_sz
        self.center += disp_in_image

        # update target size
        scale =  (1 - self.cfg.scale_lr) * 1.0 + \
            self.cfg.scale_lr * self.scale_factors[scale_id]
        self.target_sz *= scale
        self.z_sz *= scale
        self.x_sz *= scale

        # return 1-indexed and left-top based bounding box
        box = np.array([
            self.center[1] + 1 - (self.target_sz[1] - 1) / 2,
            self.center[0] + 1 - (self.target_sz[0] - 1) / 2,
            self.target_sz[1], self.target_sz[0]])

        return box
    
    def track(self, img_files, box, visualize=True):
        frame_num = len(img_files)
        boxes = np.zeros((frame_num, 4))
        boxes[0] = box
        times = np.zeros(frame_num)

        for f, img_file in enumerate(img_files):
            img = ops.read_image(img_file)
            
            begin = time.time()
            if f == 0:
                self.init(img, box)
            else:
                boxes[f, :] = self.update(img, fid=f, visualize=visualize)
            times[f] = time.time() - begin

            if visualize:
                ops.show_image(img, boxes[f, :], fid=f, visualize=visualize)

        return boxes, times
    
    def train_step(self, batch, backward=True):
        # set network mode
        self.net.train(backward)

        # parse batch data
        z = batch[0].to(self.device, non_blocking=self.cuda)
        x = batch[1].to(self.device, non_blocking=self.cuda)
        
        neg = batch[-1]

        # inference
        responses = self.net(z, x)
        r_b, r_c, r_w, r_h = responses.size()
        
        labels = []
        if all(neg):
            labels = torch.zeros(responses.size()).to(self.device)
        else:
            for n in neg:
                if n:
                    labels.append(torch.zeros([1, r_w, r_h]).to(self.device))
                else:
                    labels.append(self._create_label(responses.size()))
            labels = torch.stack(labels)
        
        

        # calculate loss
#        labels = self._create_labels(responses.size())
        loss = self.criterion(responses, labels)
        
        return loss, responses

    @torch.enable_grad()
    def train_over(self, seqs, val_seqs=None,
                   save_dir='pretrained', supervised='supervised'):
        avg = AverageMeter()
        # set to train mode
        self.net.train()

        # create save_dir folder
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # setup dataset
        transforms = SiamFCTransforms(
            exemplar_sz=self.cfg.exemplar_sz,
            instance_sz=self.cfg.instance_sz,
            context=self.cfg.context)
        dataset = Pair(
            seqs=seqs,
            transforms=transforms, supervised=supervised)
        
        # setup dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cuda,
            drop_last=True)
        
        
        end = time.time()
        # loop over epochs
        for epoch in range(self.cfg.epoch_num):
            # update lr at each epoch
            self.lr_scheduler.step(epoch=epoch)

            # loop over dataloader
            for it, batch in enumerate(dataloader):
                
#                torchvision.utils.save_image(batch[0][0], 'test0.png')
#                torchvision.utils.save_image(batch[1][0], 'test1.png')
#                
#                raise ""
                
                data_time = time.time() -end
                
                loss, responses = self.train_step(batch)
                
                # back propagation
                self.optimizer.zero_grad()
                loss.backward()
#                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 10)
                self.optimizer.step()
                
                batcn_time = time.time() - end
                end = time.time()
                avg.update(loss=loss, batch_time=batcn_time, data_time=data_time)
                
                if (it+1) %50 == 0:
                    print('Epoch: {} [{}/{}] {:.5f} {:.5f} {:.5f}'.format(
                        epoch + 1, it + 1, len(dataloader), avg.loss, avg.batch_time, avg.data_time))
                    sys.stdout.flush()
                
            # save checkpoint
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            net_path = os.path.join(
                save_dir, 'siamfc_alexnet_e%d.pth' % (epoch + 1))
            torch.save(self.net.state_dict(), net_path)
    
    def _create_labels(self, size):
        # skip if same sized labels already created
        if hasattr(self, 'labels') and self.labels.size() == size:
            return self.labels

        def logistic_labels(x, y, r_pos, r_neg):
            dist = np.abs(x) + np.abs(y)  # block distance
            labels = np.where(dist <= r_pos,
                              np.ones_like(x),
                              np.where(dist < r_neg,
                                       np.ones_like(x) * 0.5,
                                       np.zeros_like(x)))
            return labels

        # distances along x- and y-axis
        n, c, h, w = size
        x = np.arange(w) - (w - 1) / 2
        y = np.arange(h) - (h - 1) / 2
        x, y = np.meshgrid(x, y)

        # create logistic labels
        r_pos = self.cfg.r_pos / self.cfg.total_stride
        r_neg = self.cfg.r_neg / self.cfg.total_stride
        labels = logistic_labels(x, y, r_pos, r_neg)

        # repeat to size
        labels = labels.reshape((1, 1, h, w))
        labels = np.tile(labels, (n, c, 1, 1))

        # convert to tensors
        self.labels = torch.from_numpy(labels).to(self.device).float()
        
        return self.labels

    def _create_label(self, size):
        # skip if same sized labels already created
        if hasattr(self, 'labels') and self.labels.size() == size:
            return self.label

        def logistic_labels(x, y, r_pos, r_neg):
            dist = np.abs(x) + np.abs(y)  # block distance
            labels = np.where(dist <= r_pos,
                              np.ones_like(x),
                              np.where(dist < r_neg,
                                       np.ones_like(x) * 0.5,
                                       np.zeros_like(x)))
            return labels

        # distances along x- and y-axis
        n = 1
        _, c, h, w = size
        x = np.arange(w) - (w - 1) / 2
        y = np.arange(h) - (h - 1) / 2
        x, y = np.meshgrid(x, y)

        # create logistic labels
        r_pos = self.cfg.r_pos / self.cfg.total_stride
        r_neg = self.cfg.r_neg / self.cfg.total_stride
        label = logistic_labels(x, y, r_pos, r_neg)

        # repeat to size
        label = label.reshape((1, h, w))
#        label = np.tile(label, (n, c, 1, 1))

        # convert to tensors
        self.label = torch.from_numpy(label).to(self.device).float()
        
        return self.label