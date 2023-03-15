# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...builder import LOSSES
from mmdet.models.losses.ai28.contrastive_loss import supcontrastv1_0, supcontrastv1_1, supcontrastv1_2


@LOSSES.register_module()
class ContrastiveLossPlus(nn.Module):

    def __init__(self,
                 version,
                 loss_weight=1,
                 temperature=0.07,
                 memory=0,
                 num_classes=None,
                 dim=0,
                 num_views=1,
                 max_views=-1,
                 normalized_input=True,
                 iou_act='x',
                 iou_th=0.7,
                 min_samples=10,
                 **kwargs):
        """ContrastiveLossPlus."""
        super(ContrastiveLossPlus, self).__init__()
        self.version = version
        self.loss_weight = loss_weight
        self.temperature = temperature
        self.base_temperature = 1 # required?
        self.memory = memory
        self.num_views = num_views
        self.max_samples = 1024 # to do
        self.max_views = max_views  # -1: no sampling, 1: default
        self.normalized_input = normalized_input
        self.iou_act = iou_act
        self.iou_th = iou_th
        self.min_samples = min_samples
        self.kwargs = kwargs

        if self.version in ['1.0']:   # OA-contrastive
            self.loss = supcontrastv1_0
        elif self.version in ['1.1']:   # Supcon
            self.loss = supcontrastv1_1
        elif self.version in ['1.2']:   # FSCE
            self.loss = supcontrastv1_2

        else:
            raise NotImplementedError(f'does not support version=={version}')

    def forward(self,
                cont_feats, pred_cls, labels, label_weights, fg_iou,
                reduction='mean', **kwargs):
        """Forward function.

        Args:
        Returns:
            torch.Tensor: The calculated loss.
        """
        if len(cont_feats) == 0:
            return torch.zeros(1)
        if self.normalized_input:
            cont_feats = F.normalize(cont_feats, dim=1)
        # feats_: [2048, 256], lables_: [2048, ]
        loss = self.loss(cont_feats, labels, temper=self.temperature, min_samples=self.min_samples,
                         fg_iou=fg_iou, iou_act=self.iou_act, iou_th=self.iou_th)
        return self.loss_weight * loss

