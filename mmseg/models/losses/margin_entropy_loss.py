# Modified from https://github.com/valeoai/ADVENT/advent/utils/loss/

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES

def entropy_loss(v):
    """
        Entropy loss for probabilistic prediction vectors
        input: batch_size x channels x h x w
        output: batch_size x 1 x h x w
    """
    assert v.dim() == 4
    n, c, h, w = v.size()
    return -torch.sum(torch.mul(v, torch.log2(v + 1e-30))) / (n * h * w * torch.log2(c))


@LOSSES.register_module()
class MarginEntropyLoss(nn.Module):
    """EntropyLoss.
    Loss in Advent, DOI:10.1109/CVPR.2019.00262
    Args:
        margin (float, optional): keep gradient in [margin, 1-margin].
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
    """

    def __init__(self,
                 margin=0.0,
                 loss_weight=1.0,
                 **kwargs):
        super(MarginEntropyLoss, self).__init__()
        self.loss_weight = loss_weight
        self.margin = margin

    def forward(self,
                cls_score,
                **kwargs):
        """Forward function."""
        cls_pro = F.softmax(cls_score, dim=1)
        mask = (cls_pro.ge(self.margin) & cls_pro.le(1-self.margin)).long()
        return self.loss_weight * mask * entropy_loss(cls_pro)
