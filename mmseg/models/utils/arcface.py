# Obtained from https://github.com/ronghuaiyang/arcface-pytorch/tree/master/models/metrics.py
# Modifications:
# - Modified shape of feat in forward()
# - Add ModePoolLabel to downsample label to size of feat
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math

def mode_pool_label(label, num_classes, kernal_size=4, stride=4):

    label_one_hot = F.one_hot(label.squeeze(1), 
        num_classes).permute(0,3,1,2).float() # (batch, c, h, w)
    kernel = torch.ones((num_classes, 1, kernal_size, kernal_size), 
        device=label.device, requires_grad=True, dtype=torch.float)
    label_conv = F.conv2d(label_one_hot, kernel, bias=None, 
        stride=stride, groups=num_classes) # (batch, c, h/stride, w/stride)
    label_pool = torch.argmax(label_conv, 
        dim=1, keepdim=True).to(label.dtype) # (batch, 1, h/stride, w/stride)

    return label_pool

def easy_pool_label(label, kernal_size=4, stride=4):
    # return label[:,:,kernal_size//2::stride,kernal_size//2::stride]
    return label[:,:,::stride,::stride]

class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each feat sample
            out_features: size of each output sample
            s: norm of feat feature
            m: margin

            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        # self.ignore_index = ignore_index

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, feat, label=None, ignore_index=None):
        """
        Args:
            feat (tensor): (batch, in_features, H, W)
            label (tensor | optional): (batch, 1, k*H, k*W). After mitb5 and deformer_header, usually k = 4.
        """
        feat = feat.permute(0,2,3,1) # (batch, H, W, in_features)
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(feat), F.normalize(self.weight)) # (batch, H, W, out_features)

        if label is not None: # train
            k1 = int(label.shape[2] / feat.shape[1])
            k2 = int(label.shape[3] / feat.shape[2])
            assert (k1 == k2) and (label.shape[2] % k1 == 0) and (label.shape[3] % k2 == 0), \
                "k1 = {}, k2 = {}, label.shape={}, feat.shape={}".format(k1, k2, label.shape, feat.shape)
            
            if ignore_index is not None: # convert ignore_index (default 255) to normal index
                label_mdf = label.clone()
                label_mdf[label_mdf == ignore_index] = self.out_features-1 # suppose index start from 0

            # pool label to make feat and label have same dims
            # label = mode_pool_label(label, feat.shape[1], k1, k1)
            label_pool = easy_pool_label(label_mdf, k1, k1) # (batch, 1, H, W)

            sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1)) # (batch, H, W, out_features)
            phi = cosine * self.cos_m - sine * self.sin_m   # cos(theta + m)
            if self.easy_margin:
                phi = torch.where(cosine > 0, phi, cosine)
            else:
                phi = torch.where(cosine > self.th, phi, cosine - self.mm)
            # --------------------------- convert label to one-hot ---------------------------
            one_hot = F.one_hot(label_pool.squeeze(1) , num_classes=self.out_features) # (batch, H, W, out_features)
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
            output *= self.s

            return output.permute(0,3,1,2) # (batch, out_features, H, W)
        else: # test
            return cosine.permute(0,3,1,2) # (batch, out_features, H, W)