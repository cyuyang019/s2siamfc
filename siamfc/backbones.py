from __future__ import absolute_import

import torch.nn as nn
import torch.nn.functional as F
from .resnet import resnet18
from .googlenet import Inception3
from .vgg import VGG

__all__ = ['AlexNetV1', 'AlexNetV2', 'AlexNetV3']


class _BatchNorm2d(nn.BatchNorm2d):

    def __init__(self, num_features, *args, **kwargs):
        super(_BatchNorm2d, self).__init__(
            num_features, *args, eps=1e-6, momentum=0.05, **kwargs)

class conv_bn_relu(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 stride=1,
                 kszie=3,
                 pad=0,
                 has_bn=True,
                 has_relu=True,
                 bias=True,
                 groups=1):
        r"""
        Basic block with one conv, one bn, one relu in series.
        Arguments
        ---------
        in_channel: int
            number of input channels
        out_channel: int
            number of output channels
        stride: int
            stride number
        kszie: int
            kernel size
        pad: int
            padding on each edge
        has_bn: bool
            use bn or not
        has_relu: bool
            use relu or not
        bias: bool
            conv has bias or not
        groups: int or str
            number of groups. To be forwarded to torch.nn.Conv2d
        """
        super(conv_bn_relu, self).__init__()
        self.conv = nn.Conv2d(in_channel,
                              out_channel,
                              kernel_size=kszie,
                              stride=stride,
                              padding=pad,
                              bias=bias,
                              groups=groups)

        if has_bn:
            self.bn = nn.BatchNorm2d(out_channel)
        else:
            self.bn = None

        if has_relu:
            self.relu = nn.ReLU()
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class AlexNet(nn.Module):
    r"""
    AlexNet
    Hyper-parameters
    ----------------
    pretrain_model_path: string
        Path to pretrained backbone parameter file,
        Parameter to be loaded in _update_params_
    """
    default_hyper_params = {"pretrain_model_path": ""}

    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1 = conv_bn_relu(3, 96, stride=2, kszie=11, pad=0)
        self.pool1 = nn.MaxPool2d(3, 2, 0, ceil_mode=True)
        self.conv2 = conv_bn_relu(96, 256, 1, 5, 0)
        self.pool2 = nn.MaxPool2d(3, 2, 0, ceil_mode=True)
        self.conv3 = conv_bn_relu(256, 384, 1, 3, 0)
        self.conv4 = conv_bn_relu(384, 384, 1, 3, 0)
        self.conv5 = conv_bn_relu(384, 256, 1, 3, 0, has_relu=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

class Resnet18(nn.Module):
    def __init__(self, pretrained=False):
        super(Resnet18, self).__init__()
        self.backbone = resnet18(used_layers=[2, 3, 4])

    def forward(self, x):
        out = self.backbone(x)
        return out[-1]  #last conv
    
class Inception(nn.Module):
    def __init__(self):
        super(Inception, self).__init__()
        self.backbone = Inception3()
        self.backbone.update_params()

    def forward(self, x):
        out = self.backbone(x)
        return out
    
class VGG16(nn.Module):
    def __init__(self,):
        super(VGG16, self).__init__()
        self.backbone = VGG()

    def forward(self, x):
        out = self.backbone(x)
        return out