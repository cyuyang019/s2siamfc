from __future__ import absolute_import

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

__all__ = ['SiamFC']

class _BatchNorm2d(nn.BatchNorm2d):

    def __init__(self, num_features, *args, **kwargs):
        super(_BatchNorm2d, self).__init__(
            num_features, *args, eps=1e-6, momentum=0.05, **kwargs)
        

class SiamFC(nn.Module):
    def __init__(self, out_scale=0.001, BN=True):
        super(SiamFC, self).__init__()
        self.out_scale = out_scale
        self.BN = BN
        if BN:
            self.map_norm = nn.BatchNorm2d(1)
        
    def forward(self, z, x):
        if self.BN:
            return self.map_norm(self._fast_xcorr(z, x))
        else:
            return self._fast_xcorr(z, x) * self.out_scale
    
    def _fast_xcorr(self, z, x):
        # fast cross correlation
        nz = z.size(0)
        nx, c, h, w = x.size()
        x = x.view(-1, nz * c, h, w)
        out = F.conv2d(x, z, groups=nz)
        out = out.view(nx, -1, out.size(-2), out.size(-1))
        return out



class SiamFC_1x1_DW(nn.Module):
    def __init__(self, in_channel=512):
        super(SiamFC_1x1_DW, self).__init__()
        self.conv_kernel = nn.Sequential(
                nn.Conv2d(in_channel, in_channel, kernel_size=1, bias=False),
                nn.BatchNorm2d(in_channel),
                nn.ReLU(inplace=True),
                )
        self.conv_search = nn.Sequential(
                nn.Conv2d(in_channel, in_channel, kernel_size=1, bias=False),
                nn.BatchNorm2d(in_channel),
                nn.ReLU(inplace=True),
                )
        self.head = nn.Sequential(
                nn.Conv2d(in_channel, in_channel, kernel_size=1, bias=False),
                nn.BatchNorm2d(in_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channel, 1, kernel_size=1)
                )

    def forward(self, kernel, search):
        kernel = self.conv_kernel(kernel)
        search = self.conv_search(search)
        feature = self.xcorr_depthwise(search, kernel)
        out = self.head(feature)
        return out

# =============================================================================
#     def forward(self, z, x):
#         feat_dwcorr = self.conv2d_dw_group(z, x)
#         return self.head(feat_dwcorr) * self.out_scale
# =============================================================================
    
    def xcorr_depthwise(self, x, kernel):
        """depthwise cross correlation
        """
        batch = kernel.size(0)
        channel = kernel.size(1)
        x = x.view(-1, batch*channel, x.size(2), x.size(3))
        kernel = kernel.view(batch*channel, -1, kernel.size(2), kernel.size(3))
        out = F.conv2d(x, kernel, groups=batch*channel)
        out = out.view(-1, channel, out.size(2), out.size(3))
        return out
# =============================================================================
#     def conv2d_dw_group(self, kernel, x):
#         batch, channel = kernel.shape[:2]
#         batch_x = x.shape[0]
#         x = x.view(-1, batch*channel, x.size(2), x.size(3))  # 1 * (b*c) * k * k
#         kernel = kernel.view(batch*channel, -1, kernel.size(2), kernel.size(3))  # (b*c) * 1 * H * W
#         out = F.conv2d(x, kernel, groups=batch*channel)
#         out = out.view(batch_x, channel, out.size(2), out.size(3))
#         return out
# =============================================================================
