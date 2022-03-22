# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import torch
import torch.nn as nn

from ..builder import LOSSES
from .utils import weighted_loss, weighted_loss2
import torch.nn.functional as F


@mmcv.jit(derivate=True, coderize=True)
@weighted_loss2
def smooth_l1_loss_augmix(pred, target, beta=1.0):
    """Smooth L1 loss.

    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.

    Returns:
        torch.Tensor: Calculated loss
    """
    pred_orig, pred_aug1, pred_aug2 = torch.chunk(pred, 3)
    target, _, _ = torch.chunk(target,3)

    assert beta > 0
    if target.numel() == 0:
        return pred_orig.sum() * 0

    assert pred_orig.size() == target.size()
    diff = torch.abs(pred_orig - target)
    loss_orig = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)

    p_orig = F.softmax(pred_orig, dim=1)
    p_aug1 = F.softmax(pred_aug1, dim=1)
    p_aug2 = F.softmax(pred_aug2, dim=1)
    p_mixture = torch.clamp((p_orig + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
    loss_aug = (F.kl_div(p_mixture, p_orig, reduction='batchmean') +
                F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.

    return loss_orig, loss_aug


@mmcv.jit(derivate=True, coderize=True)
@weighted_loss2
def l1_loss_augmix(pred, target):
    """L1 loss.

    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.

    Returns:
        torch.Tensor: Calculated loss
    """
    pred_orig, pred_aug1, pred_aug2 = torch.chunk(pred, 3)
    target, _, _ = torch.chunk(target, 3)

    if target.numel() == 0:
        return pred_orig.sum() * 0

    assert pred_orig.size() == target.size()
    loss_orig = torch.abs(pred_orig - target)

    p_orig = F.softmax(pred_orig, dim=1)
    p_aug1 = F.softmax(pred_aug1, dim=1)
    p_aug2 = F.softmax(pred_aug2, dim=1)
    p_mixture = torch.clamp((p_orig + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
    loss_aug = (F.kl_div(p_mixture, p_orig, reduction='batchmean') +
                F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.

    return loss_orig, loss_aug


@LOSSES.register_module()
class SmoothL1LossAugmix(nn.Module):
    """Smooth L1 loss.

    Args:
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum". Defaults to "mean".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, beta=1.0, reduction='mean', loss_weight=1.0):
        super(SmoothL1LossAugmix, self).__init__()
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_bbox = self.loss_weight * smooth_l1_loss_augmix(
            pred,
            target,
            weight,
            beta=self.beta,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_bbox


@LOSSES.register_module()
class L1LossAugMix(nn.Module):
    """L1 loss.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(L1LossAugMix, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_bbox = self.loss_weight * l1_loss_augmix(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss_bbox
