# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...builder import LOSSES
from .contrastive_loss import supcontrast
from .divergence import kl_div


@LOSSES.register_module()
class ContrastiveLossPlus(nn.Module):

    def __init__(self,
                 version,
                 loss_weight=1,
                 normalized_input=True,
                 **kwargs):
        """ContrastiveLossPlus."""
        super(ContrastiveLossPlus, self).__init__()
        self.version = version
        self.loss_weight = loss_weight
        self.normalized_input = normalized_input
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
        if self.normalized_input:
            cont_feats = F.normalize(cont_feats, dim=1)

        # loss_feat = self.loss_criterion(cont_feats, labels, label_weights,
        #                                 # TODO
        #                                 **kwargs)
        loss_feat = torch.zeros(1)

        return self.loss_weight * loss_feat

