from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import json
import time
import cv2
import sys
import os
from collections import namedtuple
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
import torchvision
from torch.autograd import Variable
from six import string_types
from PIL import Image

from . import ops
from .backbones import AlexNet, Resnet18, Inception, VGG16
from .heads import SiamFC, SiamFC_1x1_DW
from .losses import AC_BalancedLoss, BalancedLoss, RankLoss
from .datasets import Pair
from .transforms import SiamFCTransforms
from utils.lr_helper import build_lr_scheduler
from utils.img_loader import cv2_RGB_loader
from got10k.trackers import Tracker
from utils.average_meter_helper import AverageMeter

__all__ = ['TrackerSiamFC']


class Net(nn.Module):
    def __init__(self, backbone, head):
        super(Net, self).__init__()
        self.backbone = backbone
        self.head = head
        self.gradients = dict()

        def backward_hook_x(grad):
            self.gradients['x'] = grad
        def backward_hook_z(grad):
            self.gradients['z'] = grad

        self.back_x = backward_hook_x
        self.back_z = backward_hook_z
    
    def forward(self, z, x):
        z = self.backbone(z)
        x = self.backbone(x)
        return self.head(z, x)

class TrackerSiamFC(Tracker):

    def __init__(self, net_path=None, name='SiamFC', loss_setting=[0, 1.5, 0],**kwargs):
        super(TrackerSiamFC, self).__init__(name, True)
        self.cfg = self.parse_args(**kwargs)

        # setup GPU device if available
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')

        # setup model
        self.net = Net(
            backbone=AlexNet(),
            head=SiamFC())
        ops.init_weights(self.net)
        
        # load checkpoint if provided
        if net_path is not None:
            self.net.load_state_dict(torch.load(
                net_path, map_location=lambda storage, loc: storage))
        self.net = self.net.to(self.device)

        # setup criterion
        self.criterion = AC_BalancedLoss(pos_thres=loss_setting[0], alpha=loss_setting[1], margin=loss_setting[2])
#        self.rank_loss = RankLoss()
#        self.criterion = BalancedLoss()

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

    def parse_args(self, **kwargs):
        # default parameters
        cfg = {
            # basic parameters
            'out_scale': 0.001,
            'exemplar_sz': 127,
            'instance_sz': 255,
            'context': 0.5,
            'backbone': 'Alex',
            # inference parameters
            'scale_num': 3,
            'scale_step': 1.0375,
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
            'r_neg': 0,
            'neg' : 0.2,
            # loss weighting
            'no_mask' : 0.7,
            'masked' : 0.15,
            }
        
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
        
        # loader img if path is given
        if isinstance(img, string_types):
#            img = cv2_RGB_loader(img)
            img = Image.open(img)
        
# =============================================================================
#         self.norm_trans = torchvision.transforms.Compose([
#         torchvision.transforms.ToPILImage(),
#         torchvision.transforms.ToTensor()])
# =============================================================================

        
        # exemplar image
        self.avg_color = np.mean(img, axis=(0, 1))
        z = ops.crop_and_resize(
            img, self.center, self.z_sz,
            out_size=self.cfg.exemplar_sz,
            border_value=self.avg_color)            #return cv2
        
        z = cv2.normalize(z, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        
        # exemplar features
        z = torch.from_numpy(z).to(
            self.device).permute(2, 0, 1).unsqueeze(0).float()
        
        
# =============================================================================
#         z = self.norm_trans(z).unsqueeze(0).to(self.device)
# =============================================================================
        
        self.kernel = self.net.backbone(z)

    @torch.no_grad()
    def update(self, img):
        # set to evaluation mode
        self.net.eval()
        
        if isinstance(img, string_types):
#            img = cv2_RGB_loader(img)
            img = Image.open(img)
        
        # search images
        x = [ops.crop_and_resize(
            img, self.center, self.x_sz * f,
            out_size=self.cfg.instance_sz,
            border_value=self.avg_color) for f in self.scale_factors]
        x = [cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) for img in x]
        x = np.stack(x, axis=0)
        x = torch.from_numpy(x).to(
            self.device).permute(0, 3, 1, 2).float()
        
# =============================================================================
#         x = [self.norm_trans(ops.crop_and_resize(
#             img, self.center, self.x_sz * f,
#             out_size=self.cfg.instance_sz,
#             border_value=self.avg_color)) for f in self.scale_factors]
#         x = torch.stack(x).to(self.device)
# =============================================================================
        
        # responses
        x = self.net.backbone(x)
        
        responses = self.net.head(self.kernel, x)               
        
        responses = responses.squeeze(1).cpu().numpy()
        # upsample responses and penalize scale changes
        
# =============================================================================
#         print(np.max(responses[1]))
#         cv2.imwrite("./res.jpg", responses[1]*255)
#         raise
# =============================================================================
        
        responses = np.stack([cv2.resize(
            u, (self.upscale_sz, self.upscale_sz),
            interpolation=cv2.INTER_CUBIC)
            for u in responses])
        responses[:self.cfg.scale_num // 2] *= self.cfg.scale_penalty
        responses[self.cfg.scale_num // 2 + 1:] *= self.cfg.scale_penalty

        # peak scale
        scale_id = np.argmax(np.amax(responses, axis=(1, 2)))

        # peak location
        response = responses[scale_id]
                
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
                boxes[f, :] = self.update(img)
            times[f] = time.time() - begin

            if visualize:
                ops.show_image(img, boxes[f, :])

        return boxes, times
    
    def train_step(self, batch, backward=True):
        def get_labels(responses):      
            # create labels
            r_b, r_c, r_w, r_h = responses.size()
            # calculate loss
            
            labels = []
            if all(neg):
                labels = torch.zeros(responses.size()).to(self.device)
            elif not any(neg):
                labels = self._create_labels(responses.size())
            else:
                for n in neg:
        #            print(n)
                    if n:
                        labels.append(torch.zeros([1, r_w, r_h]).to(self.device))
                    else:
                        labels.append(self._create_label(responses.size()))
                labels = torch.stack(labels)
            
            return labels
        
        def get_grad(labels, responses):
            # get the grad from loss with pos idx
            pos_idx = Variable(labels.view(-1).nonzero().squeeze()).to(self.device, non_blocking=self.cuda)
    
            pred_pos_value = torch.index_select(responses.view(-1), 0, pos_idx).mean()
            
            self.net.zero_grad()
            pred_pos_value.backward(retain_graph=True)
            
            grad_z = self.net.gradients['z']
            
            return grad_z

        def get_adv_mask_img(z, feat_z, grad_z):
            # get the masked image using grad-cam like approach
            b, _, w, h = feat_z.size()
        
            z_for_dropping = z.clone().detach()
            z_dropping = []
            for bid in range(b):
                z_dropping.append(gradcam_dropping(grad_z[bid], feat_z[bid], z_for_dropping[bid]))
            z_dropping = torch.stack(z_dropping)
            
            return z_dropping
        
        def gradcam_dropping(grad, activations, image):
            # cal the saliency region using grad-cam like approach
            k, u, v = grad.size()
            
            alpha = grad.view(1, k, -1).mean(2)
            weights = alpha.view(1, k, 1, 1)
            
            h, w = image.size()[1:]     #c, w, h

            atten_map = (weights*activations)
            atten_map = F.relu(atten_map)
            atten_map = F.upsample(atten_map, size=(h, w), mode='bilinear', align_corners=False)
            
        
            atten_map_max, atten_map_min = atten_map.max(), atten_map.min()
            atten_map = (atten_map - atten_map_min).div(atten_map_max - atten_map_min)
            
            atten_map_thres_neg_idx = atten_map < 0.5
            atten_map_thres_pos_idx = atten_map >= 0.5
            atten_map[atten_map_thres_neg_idx] = 0          #keep don't care part
            atten_map[atten_map_thres_pos_idx] = 1          #erase attened part 
            
            
            high_response_channel = torch.unique(atten_map[0].nonzero()[:, 0]).detach().cpu().numpy()
            
            
            avg_color = image.mean([1, 2])
            assert avg_color.size(0) == 3
            
            
            candidate = []
            for ele in high_response_channel:
                if atten_map[0][ele].sum()/(w*h) < 0.5:
                    candidate.append(ele)
            if len(candidate) != 0:
                random_idx = np.random.choice(candidate)
            else:
                random_idx = 0
            
            dropping_mask = atten_map[0][random_idx].bool()
            
            for channel in range(3):
#                print(atten_map[0][random_idx].view(-1).nonzero())
                image[channel][dropping_mask] = avg_color[channel]
                image[channel][image[channel] == 0] = avg_color[channel]
            
            image_dropping = image
#            image_dropping = image * torch.abs(1 - atten_map[0][random_idx])
            
            return image_dropping    

        # set network mode
        self.net.train(backward)

        # parse batch data
        z = batch[0].to(self.device, non_blocking=self.cuda)
        x = batch[1].to(self.device, non_blocking=self.cuda)
        neg = batch[-1]
        
#        torchvision.utils.save_image(z, './test_z.png')
#        torchvision.utils.save_image(x, './test_x.png')
#        raise
        # inference
        feat_z, feat_x = self.net.backbone(z), self.net.backbone(x)
        
        feat_z.register_hook(self.net.back_z)
        
        responses = self.net.head(feat_z, feat_x)
        
        labels = get_labels(responses)
        
        grad_z = get_grad(labels, responses)

        z_masked_1 = get_adv_mask_img(z, feat_z, grad_z)
        z_masked_2 = get_adv_mask_img(z, feat_z, grad_z)      
        
#        torchvision.utils.save_image(z_dropping, './test_z_dropping.png')
#        torchvision.utils.save_image(z_dropping_2, './test_z_dropping_2.png')
#        torchvision.utils.save_image(z_dropping_3, './test_z_dropping_3.png')
#        raise
        
        responses_masked_1 = self.net(z_masked_1, x)
        responses_masked_2 = self.net(z_masked_2, x)

        raw_loss = self.criterion(responses, labels)
        masked_1_loss = self.criterion(responses_masked_1, labels)
        masked_2_loss = self.criterion(responses_masked_2, labels)

        
        loss_siam = self.cfg.no_mask * raw_loss + self.cfg.masked * masked_1_loss + self.cfg.masked *masked_2_loss

        
        return loss_siam, responses
    
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
            transforms=transforms, supervised=supervised, img_loader=cv2_RGB_loader, neg=self.cfg.neg)
        
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
            

            # loop over dataloader
            for it, batch in enumerate(dataloader):
                
#                torchvision.utils.save_image(batch[0][0], 'test0.png')
#                torchvision.utils.save_image(batch[1][0], 'test1.png')
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
#                    print('Num_high:{:d}'.format(torch.sum(responses.detach()>0.8)))

                    sys.stdout.flush()
                    
            self.lr_scheduler.step(epoch=epoch)
            
            # save checkpoint
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            net_path = os.path.join(
                save_dir, 'siamfc_alexnet_e%d.pth' % (epoch + 1))
            torch.save(self.net.state_dict(), net_path)
        
        json.dump(self.cfg._asdict(), open(os.path.join(save_dir, 'config.json'), 'w'), indent=4)
    
    def _create_labels(self, size):
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
        def logistic_labels(x, y, r_pos, r_neg):
            dist = np.abs(x) + np.abs(y)  # block distance
            labels = np.where(dist <= r_pos,
                              np.ones_like(x),
                              np.where(dist < r_neg,
                                       np.ones_like(x) * 0.5,
                                       np.zeros_like(x)))
            return labels

        # distances along x- and y-axis
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