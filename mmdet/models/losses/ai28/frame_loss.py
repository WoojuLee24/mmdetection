# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

from ...builder import LOSSES
from ..utils import weight_reduce_loss


def fpn_loss(pred,
             **kwargs):
    """Calculate the jsdfp loss.
    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        temper (int, optional): temperature scaling for softmax function.
    Returns:
        torch.Tensor: The calculated loss
    """
    pdb.set_trace()
    temper = kwargs['temper'] if 'temper' in kwargs else 1
    add_act = kwargs['add_act'] if 'add_act' in kwargs else None
    loss_type = kwargs['loss_type'] if 'loss_type' in kwargs else 'mse'

    # chunk the data to get p_orig
    split = 2
    curr_feature, prev_feature = torch.chunk(pred[0], split)
    # curr.view(-1, 1), prev.view(-1, 1)

    if add_act == None:
        # sigmoid and softmax function for rpn_cls and roi_cls
        p_curr, p_prev = F.softmax(curr_feature / temper, dim=1), F.softmax(prev_feature / temper, dim=1)
    elif add_act == 'softmax':
        p_curr, p_prev = F.softmax(curr_feature / temper, dim=1), F.softmax(prev_feature / temper, dim=1)
    elif add_act == 'sigmoid':
        p_curr, p_prev = F.sigmoid(curr_feature / temper, dim=1), F.sigmoid(prev_feature / temper, dim=1)
    elif add_act == 'pred':
        p_curr, p_prev = curr_feature, prev_feature

    if loss_type == 'jsd':
        # Clamp mixture distribution to avoid exploding KL divergence
        p_mixture = torch.clamp((p_curr + p_prev) / 2., 1e-7, 1).log()
        loss = (F.kl_div(p_mixture, p_curr, reduction='none') +
                F.kl_div(p_mixture, p_prev, reduction='none')) / 2.

    elif loss_type == 'mse':
        loss = F.mse_loss(p_curr, p_prev, reduction='none')

    # apply weights and do the reduction

    # logging
    p_distribution = {'p_curr': p_curr,
                      'p_prev': p_prev,
                      }

    return loss, p_distribution