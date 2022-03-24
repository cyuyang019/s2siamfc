from __future__ import absolute_import, division

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


__all__ = ['BalancedLoss', 'FocalLoss', 'GHMCLoss', 'OHNMLoss']


def log_sigmoid(x):
    # for x > 0: 0 - log(1 + exp(-x))
    # for x < 0: x - log(1 + exp(x))
    # for x = 0: 0 (extra term for gradient stability)
    return torch.clamp(x, max=0) - torch.log(1 + torch.exp(-torch.abs(x))) + \
        0.5 * torch.clamp(x, min=0, max=0)


def log_minus_sigmoid(x):
    # for x > 0: -x - log(1 + exp(-x))
    # for x < 0:  0 - log(1 + exp(x))
    # for x = 0: 0 (extra term for gradient stability)
    return torch.clamp(-x, max=0) - torch.log(1 + torch.exp(-torch.abs(x))) + \
        0.5 * torch.clamp(x, min=0, max=0)

class RankLoss(nn.Module):
    def __init__(self, eps = 0):
        super(RankLoss, self).__init__()
        self.eps = eps
        
    def forward(self, input, target):
        pos_index = target.view(-1).eq(1).nonzero().squeeze()
        neg_index = target.view(-1).eq(0).nonzero().squeeze()
        mean_pos_score = torch.index_select(input.view(-1), 0, pos_index).mean()
        mean_neg_score = torch.index_select(input.view(-1), 0, neg_index).mean()
#        response_mean = input.mean()
        
        rank_loss = F.relu(mean_neg_score - mean_pos_score + self.eps)
        
        return rank_loss

class Weighted_BalancedLoss_FT(nn.Module):

    def __init__(self, neg_weight=1.0, pos_thres = 0, alpha= 1.5, margin=0):
        super(Weighted_BalancedLoss_FT, self).__init__()
        self.neg_weight = neg_weight
        self.pos_thres = pos_thres
        self.alpha = alpha
        self.margin = margin
    
    def forward(self, input, target):
        pos_mask = (target == 1)
        neg_mask = (target == 0)
#        pos_num = pos_mask.sum().float()
        neg_num = neg_mask.sum().float()
        weight = target.new_zeros(target.size())
        weight[pos_mask] = 0            #only focus neg
        weight[neg_mask] = 1 / neg_num * self.neg_weight
        weight /= weight.sum()
        
        #batch ver        
        num_high = (input > self.pos_thres).nonzero().size()[0]
#        num_high = (F.sigmoid(input) > self.pos_thres).nonzero().size()[0]
        weight *= (1 + self.margin - num_high/np.prod(input.size()))**self.alpha
        
#        num_high = (input[neg_mask] > self.pos_thres).sum()
#        weight *= (1 - num_high/neg_num)
        
# =============================================================================
#         #sample-wise ver
#         w, h = input.size()[2:]
#         batch_weight = (1 + self.margin - (torch.Tensor([b.nonzero().size()[0] for b in (input > self.pos_thres)]) / (w * h)))**self.alpha
#         weight = torch.stack([b* batch_weight[idx] for idx, b in enumerate(weight)])
# =============================================================================
        
#        for idx, sample in enumerate(input):
#            neg_label = (target[idx] == 0)
#            num_high = (sample[neg_label] > self.pos_thres).sum()
#            weight[idx] *= 1 - (num_high/neg_label.sum())
        
        return F.binary_cross_entropy_with_logits(
            input, target, weight, reduction='sum')
      
class BalancedLoss(nn.Module):

    def __init__(self, neg_weight=1.0):
        super(BalancedLoss, self).__init__()
        self.neg_weight = neg_weight
    
    def forward(self, input, target):
        pos_mask = (target == 1)
        neg_mask = (target == 0)
        pos_num = pos_mask.sum().float()
        neg_num = neg_mask.sum().float()
        weight = target.new_zeros(target.size())
        weight[pos_mask] = 1 / pos_num
        weight[neg_mask] = 1 / neg_num * self.neg_weight
        weight /= weight.sum()

        
        return F.binary_cross_entropy_with_logits(
            input, target, weight, reduction='sum')

class AC_BalancedLoss(nn.Module):

    def __init__(self, neg_weight=1.0, pos_thres = 0.8, alpha= 1.5, margin=0):
        super(AC_BalancedLoss, self).__init__()
        self.neg_weight = neg_weight
        self.pos_thres = pos_thres
        self.alpha = alpha
        self.margin = margin
    
    def normalize(self, tensor):
        b, c, w, h = tensor.size()
        array = tensor.detach().cpu().numpy()
        
        min_array = np.reshape(np.min(array, axis=(2, 3)), (b, 1, 1, 1))
        max_array = np.reshape(np.max(array, axis=(2, 3)), (b, 1, 1, 1))
        
        return (array - min_array) / (max_array - min_array)
        
    
    def forward(self, input, target):
        pos_mask = (target == 1)
        neg_mask = (target == 0)
        pos_num = pos_mask.sum().float()
        neg_num = neg_mask.sum().float()
        weight = target.new_zeros(target.size())
        weight[pos_mask] = 1 / pos_num
        weight[neg_mask] = 1 / neg_num * self.neg_weight
        weight /= weight.sum()
        

        num_high = (self.normalize(input) > self.pos_thres).sum()
        weight *= (1 + self.margin - num_high/np.prod(input.size()))**self.alpha
        
        return F.binary_cross_entropy_with_logits(
            input, target, weight, reduction='sum')

# =============================================================================
# class Weighted_BalancedLoss(nn.Module):
# 
#     def __init__(self, neg_weight=1.0, pos_thres = 0.8, alpha= 1.5, margin=0):
#         super(Weighted_BalancedLoss, self).__init__()
#         self.neg_weight = neg_weight
#         self.pos_thres = pos_thres
#         self.alpha = alpha
#         self.margin = margin
#     
#     def normalize(self, tensor):
#         b, w, h = tensor.size()
#         array = tensor.detach().cpu().numpy()
#         
#         min_array = np.min(array, axis=(1, 2)).rehape(b, 1, 1)
#         max_array = np.max(array, axis=(1, 2)).rehape(b, 1, 1)
#         
#         return (array - min_array) / (max_array - min_array)
#         
#     
#     def forward(self, input, target):
#         pos_mask = (target == 1)
#         neg_mask = (target == 0)
#         pos_num = pos_mask.sum().float()
#         neg_num = neg_mask.sum().float()
#         weight = target.new_zeros(target.size())
#         weight[pos_mask] = 1 / pos_num
#         weight[neg_mask] = 1 / neg_num * self.neg_weight
#         weight /= weight.sum()
#         
#         
#         #batch ver        
# #        num_high = (input > self.pos_thres).nonzero().size()[0]
# #        num_high = (F.sigmoid(input) > self.pos_thres).nonzero().size()[0]
#         num_high = (self.normalize(input) > self.pos_thres).nonzero().size()[0]
#         weight *= (1 + self.margin - num_high/np.prod(input.size()))**self.alpha
#         
# #        num_high = (input[neg_mask] > self.pos_thres).sum()
# #        weight *= (1 - num_high/neg_num)
#         
# # =============================================================================
# #         #sample-wise ver
# #         w, h = input.size()[2:]
# #         batch_weight = (1 + self.margin - (torch.Tensor([b.nonzero().size()[0] for b in (input > self.pos_thres)]) / (w * h)))**self.alpha
# #         weight = torch.stack([b* batch_weight[idx] for idx, b in enumerate(weight)])
# # =============================================================================
#         
# #        for idx, sample in enumerate(input):
# #            neg_label = (target[idx] == 0)
# #            num_high = (sample[neg_label] > self.pos_thres).sum()
# #            weight[idx] *= 1 - (num_high/neg_label.sum())
#         
#         return F.binary_cross_entropy_with_logits(
#             input, target, weight, reduction='sum')
# 
# =============================================================================

class AC_FocalLoss(nn.Module):

    def __init__(self, gamma=2, pos_thres = 0.8, alpha= 1.5, margin=0):
        super(AC_FocalLoss, self).__init__()
        self.gamma = gamma
        self.pos_thres = pos_thres
        self.alpha = alpha
        self.margin = margin
    
    def normalize(self, tensor):
        b, c, w, h = tensor.size()
        array = tensor.detach().cpu().numpy()
        
        min_array = np.reshape(np.min(array, axis=(2, 3)), (b, 1, 1, 1))
        max_array = np.reshape(np.max(array, axis=(2, 3)), (b, 1, 1, 1))
        
        return (array - min_array) / (max_array - min_array)
    
    def forward(self, input, target):
        pos_log_sig = log_sigmoid(input)
        neg_log_sig = log_minus_sigmoid(input)

        prob = torch.sigmoid(input)
        pos_weight = torch.pow(1 - prob, self.gamma)
        neg_weight = torch.pow(prob, self.gamma)

        loss = -(target * pos_weight * pos_log_sig + \
            (1 - target) * neg_weight * neg_log_sig)
        
        avg_weight = target * pos_weight + (1 - target) * neg_weight
        
        num_high = (self.normalize(input) > self.pos_thres).sum()
        ac_weight = (1 + self.margin - num_high/np.prod(input.size()))**self.alpha
        
        
        loss /= avg_weight.mean()
        
        loss *= ac_weight
        
        return loss.mean()

class FocalLoss(nn.Module):

    def __init__(self, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
    
    def forward(self, input, target):
        pos_log_sig = log_sigmoid(input)
        neg_log_sig = log_minus_sigmoid(input)

        prob = torch.sigmoid(input)
        pos_weight = torch.pow(1 - prob, self.gamma)
        neg_weight = torch.pow(prob, self.gamma)

        loss = -(target * pos_weight * pos_log_sig + \
            (1 - target) * neg_weight * neg_log_sig)
        
        avg_weight = target * pos_weight + (1 - target) * neg_weight
        loss /= avg_weight.mean()

        return loss.mean()


class GHMCLoss(nn.Module):
    
    def __init__(self, bins=30, momentum=0.5):
        super(GHMCLoss, self).__init__()
        self.bins = bins
        self.momentum = momentum
        self.edges = [t / bins for t in range(bins + 1)]
        self.edges[-1] += 1e-6
        if momentum > 0:
            self.acc_sum = [0.0 for _ in range(bins)]
    
    def forward(self, input, target):
        edges = self.edges
        mmt = self.momentum
        weights = torch.zeros_like(input)

        # gradient length
        g = torch.abs(input.sigmoid().detach() - target)

        tot = input.numel()
        n = 0
        for i in range(self.bins):
            inds = (g >= edges[i]) & (g < edges[i + 1])
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                if mmt > 0:
                    self.acc_sum[i] = mmt * self.acc_sum[i] \
                        + (1 - mmt) * num_in_bin
                    weights[inds] = tot / self.acc_sum[i]
                else:
                    weights[inds] = tot / num_in_bin
                n += 1
        if n > 0:
            weights /= weights.mean()

        loss = F.binary_cross_entropy_with_logits(
            input, target, weights, reduction='sum') / tot
        
        return loss


class OHNMLoss(nn.Module):
    
    def __init__(self, neg_ratio=3.0):
        super(OHNMLoss, self).__init__()
        self.neg_ratio = neg_ratio
    
    def forward(self, input, target):
        pos_logits = input[target > 0]
        pos_labels = target[target > 0]

        neg_logits = input[target == 0]
        neg_labels = target[target == 0]

        pos_num = pos_logits.numel()
        neg_num = int(pos_num * self.neg_ratio)
        neg_logits, neg_indices = neg_logits.topk(neg_num)
        neg_labels = neg_labels[neg_indices]

        loss = F.binary_cross_entropy_with_logits(
            torch.cat([pos_logits, neg_logits]),
            torch.cat([pos_labels, neg_labels]),
            reduction='mean')
        
        return loss
