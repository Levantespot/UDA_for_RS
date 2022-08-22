# The ema model and the domain-mixing are based on:
# https://github.com/vikolss/DACS
# Modificaions:
# Add funciton `mix_train_img`
# Support local pseudo weight


import math
import os
import random
from copy import deepcopy

import mmcv
import numpy as np
import torch
from torch.nn import Parameter
from torch.nn.functional import one_hot, conv2d
from matplotlib import pyplot as plt
import proplot as pplt
from timm.models.layers import DropPath
from torch.nn.modules.dropout import _DropoutNd

from mmseg.core import add_prefix
from mmseg.models import UDA, build_segmentor
from mmseg.models.uda.uda_decorator import UDADecorator, get_module
from mmseg.models.utils.dacs_transforms import (denorm, get_class_masks,
                                                get_mean_std, strong_transform)
from mmseg.models.utils.visualization import subplotimg
from mmseg.utils.utils import downscale_label_ratio


def calc_grad_magnitude(grads, norm_type=2.0):
    norm_type = float(norm_type)
    if norm_type == math.inf:
        norm = max(p.abs().max() for p in grads)
    else:
        norm = torch.norm(
            torch.stack([torch.norm(p, norm_type) for p in grads]), norm_type)

    return norm


@UDA.register_module()
class DACS(UDADecorator):

    def __init__(self, **cfg):
        super(DACS, self).__init__(**cfg)
        self.local_iter = 0
        self.max_iters = cfg['max_iters']
        self.alpha = cfg['alpha'] # parameter for EMA
        self.dynamic_class_weight = cfg['dynamic_class_weight']
        self.pseudo_threshold = cfg['pseudo_threshold']
        self.mix = cfg['mix']
        self.blur = cfg['blur']
        self.color_jitter_s = cfg['color_jitter_strength']
        self.color_jitter_p = cfg['color_jitter_probability']
        self.debug_img_interval = cfg['debug_img_interval']
        self.print_grad_magnitude = cfg['print_grad_magnitude']
        self.pseudo_kernal_size = cfg['pseudo_kernal_size']
        self.local_ps_weight_type = cfg['local_ps_weight_type']
        assert self.mix == 'class'

        self.debug_gt_rescale = None

        self.class_probs = {}
        ema_cfg = deepcopy(cfg['model'])
        self.ema_model = build_segmentor(ema_cfg)

        self.num_classes = self.get_model().num_classes

        if self.dynamic_class_weight is True:
            self.class_weights = None
            mmcv.print_log(f'\n Dynamic class weight enabled! \n')

        if self.pseudo_kernal_size is not None:
            assert isinstance(self.pseudo_kernal_size, int) and (self.pseudo_kernal_size % 2 == 1) \
                and (self.pseudo_kernal_size > 0), 'pseudo_kernal_size should be positive odd number'   
            self.local_kernel = None

    def get_ema_model(self):
        return get_module(self.ema_model)

    def _init_ema_weights(self):
        for param in self.get_ema_model().parameters():
            param.detach_()
        mp = list(self.get_model().parameters())
        mcp = list(self.get_ema_model().parameters())
        for i in range(0, len(mp)):
            if not mcp[i].data.shape:  # scalar tensor
                mcp[i].data = mp[i].data.clone()
            else:
                mcp[i].data[:] = mp[i].data[:].clone()

    def _update_ema(self, iter):
        alpha_teacher = min(1 - 1 / (iter + 1), self.alpha)
        for ema_param, param in zip(self.get_ema_model().parameters(),
                                    self.get_model().parameters()):
            if not param.data.shape:  # scalar tensor
                ema_param.data = \
                    alpha_teacher * ema_param.data + \
                    (1 - alpha_teacher) * param.data
            else:
                ema_param.data[:] = \
                    alpha_teacher * ema_param[:].data[:] + \
                    (1 - alpha_teacher) * param[:].data[:]

    def get_dynamic_class_weight(self, labels, T=0.1, alpha=0.9):
        """Calulate weight for each class.
        
        Args:
        labels: (batch_size, 1, H, W)
        T : Temperature. Greater one leads to a uniform distribution, 
            smaller one pay more attention to the one occur less
        alpha: The importance of current states. Refered as beta in our paper.
        Return:
        weights (Tensor): (batch_size, H, W)
        """
        labels = labels.detach()
        bs = labels.shape[0]

        if self.class_weights is None:
            self.class_weights = torch.ones(self.num_classes, 1, device=labels.device)

        masks = torch.stack([(labels == c) for \
            c in range(self.num_classes)]).squeeze(2) # (num_class, bs, H, W)
        freq = masks.sum(dim=(2,3)) / masks.sum(dim=(0,2,3)) # (num_class, bs)
        e_1_minus_freq = torch.exp( (1-freq+1e-6)/T )
        cur_class_weights = e_1_minus_freq / e_1_minus_freq.sum(dim=0) * self.num_classes # (num_class, bs)
        assert not torch.isnan(cur_class_weights).any(), 'freq : {}\ne_1_minus_freq: {}'.format(freq, e_1_minus_freq)
        
        cur_class_weights = alpha * self.class_weights + (1-alpha) * cur_class_weights
        weights = (masks * cur_class_weights.view(self.num_classes, bs, 1, 1)).sum(dim=0)

        self.class_weights = torch.mean(cur_class_weights, dim=1, keepdim=True)
        assert self.class_weights.shape == (self.num_classes, 1)

        return weights

    def generate_local_pseudo_weight(self, pseudo_label, pseudo_prob, num_classes=None, type='label'):
        """
        Args:
            pseudo_label (Tensor) : (batch, H, W)
            pseudo_prob (Tensor) : (batch, H, W)
            num_classes (int)
            type (str) : If type = 'label', count valid pseudo label around.
                         If type = 'class', count same pseudo label around.
        """
        dev = pseudo_label.device
        k_size = self.pseudo_kernal_size
        # assert len(pseudo_label.shape) == 3, 'pseudo_label should have only 3 dimensions'

        if type == 'class':
            p_one_hot = one_hot(pseudo_label, num_classes).permute(0,3,1,2).float()
            if self.local_kernel is None: # initialize only once
                self.local_kernel = torch.ones((num_classes, 1, k_size, k_size), 
                    dtype=torch.float, device=dev, requires_grad=False)
            conv_p = conv2d(p_one_hot, self.local_kernel, bias=None, 
                stride=1, padding='same', groups=num_classes)
            neighbor_cnt = conv_p.sum(dim=1)
            pseudo_weight = conv_p / neighbor_cnt.unsqueeze(1)
            pseudo_weight = torch.gather(pseudo_weight, dim=1, 
                index=pseudo_label.unsqueeze(1)).squeeze(1) # (batch, H, W)
        elif type == 'label':
            if len(pseudo_prob.shape) == 3:
                pseudo_prob = pseudo_prob.unsqueeze(1)

            if self.local_kernel is None: # initialize only once
                self.local_kernel = torch.ones((1, 1, k_size, k_size), 
                    dtype=torch.float, device=dev, requires_grad=False)
            ps_large_p = pseudo_prob.ge(self.pseudo_threshold).float()
            ps_conv = conv2d(ps_large_p, self.local_kernel, bias=None, 
                stride=1, padding='same').squeeze(1)
            pseudo_weight = ps_conv / k_size**2
        elif type == None: # used for debug
            pseudo_weight = torch.ones(pseudo_prob.shape, device=dev)
        else:
            raise NotImplementedError
            
        return pseudo_weight



    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """
        optimizer.zero_grad()
        log_vars = self(**data_batch)
        optimizer.step()

        log_vars.pop('loss', None)  # remove the unnecessary 'loss'
        outputs = dict(
            log_vars=log_vars, num_samples=len(data_batch['img_metas']))
        return outputs

    def forward_train(self, img, img_metas, gt_semantic_seg, target_img,
                      target_img_metas):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        log_vars = {}
        batch_size = img.shape[0]
        dev = img.device

        # Init/update ema model
        if self.local_iter == 0:
            self._init_ema_weights()
            # assert _params_equal(self.get_ema_model(), self.get_model())

        if self.local_iter > 0:
            self._update_ema(self.local_iter)
            # assert not _params_equal(self.get_ema_model(), self.get_model())
            # assert self.get_ema_model().training

        # Train on source images
        
        if self.dynamic_class_weight: # Apply mixing class on train images
            weights = self.get_dynamic_class_weight(gt_semantic_seg)
            clean_losses = self.get_model().forward_train(
                img, img_metas, gt_semantic_seg, weights, return_feat=True)
        else:
            clean_losses = self.get_model().forward_train(
            img, img_metas, gt_semantic_seg, return_feat=True)
        
        
        src_feat = clean_losses.pop('features')
        clean_loss, clean_log_vars = self._parse_losses(clean_losses)
        log_vars.update(clean_log_vars)
        # clean_loss.backward(retain_graph=self.enable_fdist)
        clean_loss.backward()
        if self.print_grad_magnitude:
            params = self.get_model().backbone.parameters()
            seg_grads = [
                p.grad.detach().clone() for p in params if p.grad is not None
            ]
            grad_mag = calc_grad_magnitude(seg_grads)
            mmcv.print_log(f'Seg. Grad.: {grad_mag}', 'mmseg')

        # Generate pseudo-label
        for m in self.get_ema_model().modules():
            if isinstance(m, _DropoutNd):
                m.training = False
            if isinstance(m, DropPath):
                m.training = False
        ema_logits = self.get_ema_model().encode_decode(
            target_img, target_img_metas)

        ema_softmax = torch.softmax(ema_logits.detach(), dim=1)
        pseudo_prob, pseudo_label = torch.max(ema_softmax, dim=1)

        if self.pseudo_kernal_size is None: # global pseudo weight
            ps_large_p = pseudo_prob.ge(self.pseudo_threshold).long() == 1
            ps_size = np.size(np.array(pseudo_label.cpu()))
            pseudo_weight = torch.sum(ps_large_p).item() / ps_size
            pseudo_weight = pseudo_weight * torch.ones(
                pseudo_prob.shape, device=dev)
        else: # local pseudo weight
            pseudo_weight = self.generate_local_pseudo_weight(pseudo_label, 
                pseudo_prob, ema_logits.shape[1], type=self.local_ps_weight_type)

        # Apply mixing target with clippings of source
        means, stds = get_mean_std(img_metas, dev)
        strong_parameters = {
            'mix': None,
            'color_jitter': random.uniform(0, 1),
            'color_jitter_s': self.color_jitter_s,
            'color_jitter_p': self.color_jitter_p,
            'blur': random.uniform(0, 1) if self.blur else 0,
            'mean': means[0].unsqueeze(0),  # assume same normalization
            'std': stds[0].unsqueeze(0)
        }

        mixed_img, mixed_lbl = [None] * batch_size, [None] * batch_size
        mix_masks = get_class_masks(gt_semantic_seg)

        gt_pixel_weight = torch.ones((pseudo_weight.shape), device=dev)
        for i in range(batch_size):
            strong_parameters['mix'] = mix_masks[i]
            mixed_img[i], mixed_lbl[i] = strong_transform(
                strong_parameters,
                data=torch.stack((img[i], target_img[i])),
                target=torch.stack((gt_semantic_seg[i][0], pseudo_label[i])))
            _, pseudo_weight[i] = strong_transform(
                strong_parameters,
                target=torch.stack((gt_pixel_weight[i], pseudo_weight[i])))
        mixed_img = torch.cat(mixed_img)
        mixed_lbl = torch.cat(mixed_lbl)

        # Train on mixed images
        mix_losses = self.get_model().forward_train(
            mixed_img, img_metas, mixed_lbl, pseudo_weight, return_feat=True)
        mix_losses.pop('features')
        mix_losses = add_prefix(mix_losses, 'mix')
        mix_loss, mix_log_vars = self._parse_losses(mix_losses)
        log_vars.update(mix_log_vars)
        mix_loss.backward()

        
        # if self.local_iter % self.debug_img_interval == 0:
        if self.local_iter % 400 == 0:
            ############# class weights #############
            if self.dynamic_class_weight:
                out_dir = os.path.join(self.train_cfg['work_dir'],
                                    'class_weights_debug')
                os.makedirs(out_dir, exist_ok=True)
                
                torch.save(self.class_weights.cpu(), os.path.join(out_dir, 'class_weights_{}'.format(self.local_iter)))

        self.local_iter += 1

        return log_vars
