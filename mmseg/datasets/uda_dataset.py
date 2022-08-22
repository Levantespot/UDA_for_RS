import json
import os.path as osp

import mmcv
import numpy as np
import torch

from .builder import DATASETS

@DATASETS.register_module()
class UDADataset(object):

    def __init__(self, source, target, cfg):
        self.source = source
        self.target = target
        self.ignore_index = target.ignore_index
        self.CLASSES = target.CLASSES
        self.PALETTE = target.PALETTE
        assert target.ignore_index == source.ignore_index
        assert target.CLASSES == source.CLASSES
        assert target.PALETTE == source.PALETTE

    def __getitem__(self, idx):
        s1 = self.source[idx // len(self.target)]
        s2 = self.target[idx % len(self.target)]
        return {
            **s1, 'target_img_metas': s2['img_metas'],
            'target_img': s2['img']
        }

    def __len__(self):
        return len(self.source) * len(self.target)
