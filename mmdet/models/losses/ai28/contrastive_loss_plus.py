# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...builder import LOSSES
from .contrastive_loss import supcontrast
from .divergence import kl_div


class SupJSD(nn.Module):
    def __init__(self, reduction='mean', use_softmax=False):
        super(SupJSD, self).__init__()
        self.reduction = reduction
        self.use_softmax = use_softmax

    def forward(self, logits_clean, logits_aug1, logits_aug2, labels=None, eps=1e-7):
        if self.use_softmax:
            logits_clean, logits_aug1, logits_aug2 = F.softmax(logits_clean, dim=1), \
                                                     F.softmax(logits_aug1, dim=1), \
                                                     F.softmax(logits_aug2, dim=1),
        else: # F.normalize
            logits_clean, logits_aug1, logits_aug2 = F.normalize(logits_clean, dim=1), \
                                                     F.normalize(logits_aug1, dim=1), \
                                                     F.normalize(logits_aug2, dim=1),

        probs = torch.stack([logits_clean, logits_aug1, logits_aug2], dim=0) # (M, B, D)
        probs = probs.reshape(-1, probs.shape[-1]) # (M*B, D)
        labels = labels.repeat(3) # (M,)

        weighted_kld_list = []
        for i in range(len(probs)):
            # Anchor
            anchor = probs[i]

            # Mixture of positives (with same label)
            y_mask = torch.eq(labels, labels[i])
            num_pos = torch.sum(y_mask)
            y_mask = y_mask.unsqueeze(-1).expand(probs.size())

            mixture = torch.sum(probs * y_mask, dim=0) / num_pos

            # Calculate JSD
            weighted_kld = kl_div(mixture.clamp(min=eps).log().unsqueeze(0),
                                  anchor.unsqueeze(0),
                                  reduction=self.reduction) / num_pos
            weighted_kld_list.append(weighted_kld)

        weighted_kld = torch.stack(weighted_kld_list, dim=0)
        jsd = torch.sum(weighted_kld)
        return jsd


class SelfJSD(nn.Module):
    def __init__(self, jsd_criterion, reduction='mean', use_softmax=False):
        super(SelfJSD, self).__init__()
        self.jsd_criterion = jsd_criterion
        self.reduction = reduction
        self.use_softmax = use_softmax

    def forward(self, logits_clean, logits_aug1, logits_aug2, labels=None):
        if self.use_softmax:
            logits_clean, logits_aug1, logits_aug2 = F.softmax(logits_clean, dim=1), \
                                                     F.softmax(logits_aug1, dim=1), \
                                                     F.softmax(logits_aug2, dim=1),
        else: # F.normalize
            logits_clean, logits_aug1, logits_aug2 = F.normalize(logits_clean, dim=1), \
                                                     F.normalize(logits_aug1, dim=1), \
                                                     F.normalize(logits_aug2, dim=1),

        prob_list = [logits_aug1, logits_aug2]
        targets = logits_clean
        jsd = self.jsd_criterion(prob_list, targets, reduction=self.reduction)

        return jsd


@LOSSES.register_module()
class ContrastiveLossPlus(nn.Module):

    def __init__(self,
                 version,
                 loss_weight=1,
                 **kwargs):
        """ContrastiveLossPlus."""
        super(ContrastiveLossPlus, self).__init__()
        self.version = version
        self.loss_weight = loss_weight
        self.kwargs = kwargs

        if self.version in ['1.1']:
            # self.loss_criterion = supcontrast
            pass
        else:
            raise NotImplementedError(f'does not support version=={version}')

    def forward(self,
                cont_feats, labels, label_weights,
                **kwargs):
        """Forward function.

        Args:
        Returns:
            torch.Tensor: The calculated loss.
        """
        if len(cont_feats) == 0:
            return torch.zeros(1)

        mask = None # TODO


        # loss_feat = self.loss_criterion(cont_feats, labels, label_weights,
        #                                 # TODO
        #                                 **kwargs)
        loss_feat = torch.zeros(1)

        return self.loss_weight * loss_feat

