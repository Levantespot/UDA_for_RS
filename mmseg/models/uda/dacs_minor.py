# The ema model and the domain-mixing are based on:
# https://github.com/vikolss/DACS
# Modificaions:
# Add funciton `mix_source_img`
# Support local pseudo weight


from ast import Not
import math
import os
import random
from copy import deepcopy

import mmcv
import numpy as np
import torch
from torch.nn.functional import one_hot, conv2d
from matplotlib import pyplot as plt
from timm.models.layers import DropPath
from torch.nn.modules.dropout import _DropoutNd

from mmseg.core import add_prefix
from mmseg.models import UDA, build_segmentor
from mmseg.models.uda.uda_decorator import UDADecorator, get_module
from mmseg.models.utils.dacs_transforms import (denorm, get_class_masks, one_mix,
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
        self.alpha = cfg['alpha'] # weight for teacher model
        self.mix_class_threshold = cfg['mix_class_threshold']
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

        self.enable_mix_source = False
        if self.mix_class_threshold is not None:
            assert 0 <= self.mix_class_threshold <= 1
            self.enable_mix_source = True
            self.mix_source_type = 'MinorClassMix' # default
            if 'mix_source_type' in cfg:
                assert cfg['mix_source_type'] in ['MinorClassMix', 'CutMix', 'ClassMix']
                self.mix_source_type = cfg['mix_source_type']

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

    def mix_source_img(self, img_batch, gt_semantic_seg_batch, type):
        """Mix source images to alleviate class imbalance.

        Mix a class of image with that of another image in a batch
        if the ratio of the class is below self.mix_class_threshold
        """
        batch_size = img_batch.shape[0]
        img_size = img_batch[0].size().numel()
        assert img_size > 1, 'No mix will be applied to source domain when batch_size = 1.'
        H, W = img_batch.size()[-2:]
        dev = img_batch.device

        mixed_img_batch, mixed_gt_batch = [None] * batch_size, [None] * batch_size
        
        if type == 'MinorClassMix':
            gt_class_pros = [None] * batch_size
            class_with_idx = dict() # class to idxs
            
            # calculate dict from class to idx
            for idx, gt_seg in enumerate(gt_semantic_seg_batch):
                gt_class, gt_cnt = torch.unique(gt_seg, return_counts=True)
                gt_pro = gt_cnt / img_size
                gt_class_pros[idx] = dict(zip(gt_class, gt_pro))
                for cls in gt_class:
                    if cls not in class_with_idx.keys():
                        class_with_idx[cls] = [idx]
                    else:
                        class_with_idx.append(idx)

            all_classes = list(class_with_idx.keys())
            for idx, (img, gt) in enumerate(zip(img_batch, gt_semantic_seg_batch)):
                cls_under_thre = [cls for cls in all_classes if cls not in gt_class_pros[idx] or \
                    gt_class_pros[idx][cls] < self.mix_class_threshold] # classes to be extented
                for cls in cls_under_thre:
                    idx_list = [_idx for _idx in class_with_idx[cls] if _idx != idx]
                    if len(idx_list) == 0: continue # pass if only one exists
                    mixed_idx = np.random.choice(idx_list)
                    mask = (gt_semantic_seg_batch[mixed_idx] == cls).long()
                    img = (1-mask) * img + mask * img_batch[mixed_idx]
                    gt = (1-mask) * gt + mask * gt_semantic_seg_batch[mixed_idx]
                mixed_img_batch[idx] = img
                mixed_gt_batch[idx] = gt
        elif type == 'CutMix':
            # Modified from https://github.com/clovaai/CutMix-PyTorch
            # Modifications: generate diffderent ratios for H and W instead of a same one.
            for idx in range(batch_size):
                # in CutMix, default alpha and beta in beta distribution both are 1, 
                # which equals to uniform distribution
                cut_h_ratio = torch.rand(1) 
                cut_w_ratio = torch.rand(1) 
                                           
                idx_list = np.arange(batch_size)
                idx_list = np.delete(idx_list, idx) # remove itself
                mixed_idx = np.random.choice(idx_list)

                cut_w = np.int(W * cut_w_ratio)
                cut_h = np.int(H * cut_h_ratio)
                # uniform
                cx = np.random.randint(W)
                cy = np.random.randint(H)

                bbx1 = np.clip(cx - cut_w // 2, 0, W)
                bby1 = np.clip(cy - cut_h // 2, 0, H)
                bbx2 = np.clip(cx + cut_w // 2, 0, W)
                bby2 = np.clip(cy + cut_h // 2, 0, H)

                mask = torch.zeros((H, W), dtype=torch.long, device=dev)
                mask[bby1:bby2, bbx1:bbx2] = 1

                mixed_img_batch[idx] = mask * img_batch[idx] + (1-mask) * img_batch[mixed_idx]
                mixed_gt_batch[idx] = mask * gt_semantic_seg_batch[idx] + (1-mask) * gt_semantic_seg_batch[mixed_idx]
        elif type == 'ClassMix':
            for idx in range(batch_size):
                mix_masks = get_class_masks(gt_semantic_seg_batch[idx])

                idx_list = np.arange(batch_size)
                idx_list = np.delete(idx_list, idx) # remove itself
                mixed_idx = np.random.choice(idx_list)

                mixed_img, mixed_gt = one_mix(
                    mask=mix_masks,
                    data=torch.stack((img_batch[idx], img_batch[mixed_idx])),
                    target=torch.stack((gt_semantic_seg_batch[idx], gt_semantic_seg_batch[mixed_idx]))
                )
                mixed_img_batch[idx], mixed_gt_batch[idx] = mixed_img.squeeze(), mixed_gt.squeeze().unsqueeze(0)
                # assert len(mixed_img_batch[idx].shape) == 3, "{}, {}".format(mixed_img.shape, mixed_gt.shape)
        else:
            raise NotImplementedError

        return torch.stack(mixed_img_batch), torch.stack(mixed_gt_batch)

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
            img (Tensor): Input images. (B, 3, H, W)
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task. (B, 1, H, W)

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
        
        if self.enable_mix_source:
            # Apply mixing on source images while leaving original images and labels unchanged
            img_source_mix, gt_semantic_seg_source_mix = self.mix_source_img(img, gt_semantic_seg, self.mix_source_type)
            clean_losses = self.get_model().forward_train(
                img_source_mix, img_metas, gt_semantic_seg_source_mix, return_feat=True)
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

        if self.local_iter % self.debug_img_interval == 0:
            out_dir = os.path.join(self.train_cfg['work_dir'],
                                   'class_mix_debug')
            os.makedirs(out_dir, exist_ok=True)
            vis_img = torch.clamp(denorm(img, means, stds), 0, 1)
            vis_trg_img = torch.clamp(denorm(target_img, means, stds), 0, 1)
            vis_mixed_img = torch.clamp(denorm(mixed_img, means, stds), 0, 1)
            for j in range(batch_size):
                rows, cols = 2, 5
                fig, axs = plt.subplots(
                    rows,
                    cols,
                    figsize=(3 * cols, 3 * rows),
                    gridspec_kw={
                        'hspace': 0.1,
                        'wspace': 0,
                        'top': 0.95,
                        'bottom': 0,
                        'right': 1,
                        'left': 0
                    },
                )
                subplotimg(axs[0][0], vis_img[j], 'Source Image')
                subplotimg(axs[1][0], vis_trg_img[j], 'Target Image')
                subplotimg(
                    axs[0][1],
                    gt_semantic_seg[j],
                    'Source Seg GT',
                    cmap='cityscapes')
                subplotimg(
                    axs[1][1],
                    pseudo_label[j],
                    'Target Seg (Pseudo) GT',
                    cmap='cityscapes')
                subplotimg(axs[0][2], vis_mixed_img[j], 'Mixed Image')
                subplotimg(
                    axs[1][2], mix_masks[j][0], 'Domain Mask', cmap='gray')
                # subplotimg(axs[0][3], pred_u_s[j], "Seg Pred",
                #            cmap="cityscapes")
                subplotimg(
                    axs[1][3], mixed_lbl[j], 'Seg Targ', cmap='cityscapes')
                subplotimg(
                    axs[0][3], pseudo_weight[j], 'Pseudo W.', vmin=0, vmax=1)
                if self.debug_gt_rescale is not None:
                    subplotimg(
                        axs[1][4],
                        self.debug_gt_rescale[j],
                        'Scaled GT',
                        cmap='cityscapes')
                for ax in axs.flat:
                    ax.axis('off')
                plt.savefig(
                    os.path.join(out_dir,
                                 f'{(self.local_iter + 1):06d}_{j}.png'))
                plt.close()
        self.local_iter += 1

        return log_vars
