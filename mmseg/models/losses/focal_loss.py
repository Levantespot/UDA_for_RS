import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import reduce_loss

from .cross_entropy_loss import binary_cross_entropy, mask_cross_entropy, cross_entropy

@LOSSES.register_module()
class FocalLoss(nn.Module):
 
    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 alpha=1.0,
                 gamma=2.0):
        super(FocalLoss, self).__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.alpha = alpha
        self.gamma = gamma
 
        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        elif self.use_mask:
            self.cls_criterion = mask_cross_entropy
        else:
            self.cls_criterion = cross_entropy
        
 
    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
        else:
            class_weight = None
        
        ce_loss = self.cls_criterion(
            cls_score,
            label,
            weight,
            class_weight=class_weight,
            reduction='none',
            avg_factor=avg_factor,
            **kwargs)
        
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        loss_cls = self.loss_weight * reduce_loss(focal_loss, reduction=reduction)
        return loss_cls