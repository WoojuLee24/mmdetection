# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...builder import LOSSES
from ..utils import weight_reduce_loss
import mmdet.models.detectors.base as base
from .contrastive_loss import supcontrast_clean_fg, supcontrast_clean_fg_bg, \
    analyze_representations_1input, analyze_representations_2input, analyze_representations_3input


def cross_entropy(pred,
                  label,
                  weight=None,
                  reduction='mean',
                  avg_factor=None,
                  class_weight=None,
                  ignore_index=-100,
                  num_views=3):
    """Calculate the CrossEntropy loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        label (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.
        ignore_index (int | None): The label index to be ignored.
            If None, it will be set to default value. Default: -100.

    Returns:
        torch.Tensor: The calculated loss
    """
    # The default value of ignore_index is the same as F.cross_entropy
    ignore_index = -100 if ignore_index is None else ignore_index

    pred_orig = torch.chunk(pred, num_views)[0]
    label = torch.chunk(label, num_views)[0]

    loss = F.cross_entropy(
        pred_orig,
        label,
        weight=class_weight,
        reduction='none',
        ignore_index=ignore_index)

    # apply weights and do the reduction
    weight = torch.chunk(weight, num_views)[0]
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def _expand_onehot_labels(labels, label_weights, label_channels, ignore_index):
    """Expand onehot labels to match the size of prediction."""
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    valid_mask = (labels >= 0) & (labels != ignore_index)
    inds = torch.nonzero(
        valid_mask & (labels < label_channels), as_tuple=False)

    if inds.numel() > 0:
        bin_labels[inds, labels[inds]] = 1

    valid_mask = valid_mask.view(-1, 1).expand(labels.size(0),
                                               label_channels).float()
    if label_weights is None:
        bin_label_weights = valid_mask
    else:
        bin_label_weights = label_weights.view(-1, 1).repeat(1, label_channels)
        bin_label_weights *= valid_mask

    return bin_labels, bin_label_weights


def binary_cross_entropy(pred,
                         label,
                         weight=None,
                         reduction='mean',
                         avg_factor=None,
                         class_weight=None,
                         ignore_index=-100,
                         num_views=3):
    """Calculate the binary CrossEntropy loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, 1).
        label (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.
        ignore_index (int | None): The label index to be ignored.
            If None, it will be set to default value. Default: -100.

    Returns:
        torch.Tensor: The calculated loss.
    """
    # The default value of ignore_index is the same as F.cross_entropy
    ignore_index = -100 if ignore_index is None else ignore_index
    if pred.dim() != label.dim():
        label, weight = _expand_onehot_labels(label, weight, pred.size(-1),
                                              ignore_index)

    # weighted element-wise losses
    weight = torch.chunk(weight, num_views)[0]
    if weight is not None:
        weight = weight.float()

    pred_orig = torch.chunk(pred, num_views)[0]
    label = torch.chunk(label, num_views)[0]

    loss = F.binary_cross_entropy_with_logits(
        pred_orig, label.float(), pos_weight=class_weight, reduction='none')

    # do the reduction for the weighted loss
    loss = weight_reduce_loss(
        loss, weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def mask_cross_entropy(pred,
                       target,
                       label,
                       reduction='mean',
                       avg_factor=None,
                       class_weight=None,
                       ignore_index=None,
                       num_views=3):
    """Calculate the CrossEntropy loss for masks.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C, *), C is the
            number of classes. The trailing * indicates arbitrary shape.
        target (torch.Tensor): The learning label of the prediction.
        label (torch.Tensor): ``label`` indicates the class label of the mask
            corresponding object. This will be used to select the mask in the
            of the class which the object belongs to when the mask prediction
            if not class-agnostic.
        reduction (str, optional): The method used to reduce the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.
        ignore_index (None): Placeholder, to be consistent with other loss.
            Default: None.

    Returns:
        torch.Tensor: The calculated loss

    Example:
        >>> N, C = 3, 11
        >>> H, W = 2, 2
        >>> pred = torch.randn(N, C, H, W) * 1000
        >>> target = torch.rand(N, H, W)
        >>> label = torch.randint(0, C, size=(N,))
        >>> reduction = 'mean'
        >>> avg_factor = None
        >>> class_weights = None
        >>> loss = mask_cross_entropy(pred, target, label, reduction,
        >>>                           avg_factor, class_weights)
        >>> assert loss.shape == (1,)
    """
    assert ignore_index is None, 'BCE loss does not support ignore_index'
    # TODO: handle these two reserved arguments
    assert reduction == 'mean' and avg_factor is None
    num_rois = pred.size()[0]
    inds = torch.arange(0, num_rois, dtype=torch.long, device=pred.device)
    pred_slice = pred[inds, label].squeeze(1)

    pred_orig = torch.chunk(pred_slice, num_views)[0]
    label = torch.chunk(label, num_views)[0]

    loss = F.binary_cross_entropy_with_logits(
        pred_orig, target, weight=class_weight, reduction='mean')[None]

    return loss



def jsdv1_3(pred,
            label,
            weight=None,
            reduction='mean',
            avg_factor=None,
            use_cls_weight=False,
            **kwargs):
    """Calculate the jsdv1.3 loss.
    jsd loss (sigmoid, 1-sigmoid) for rpn head, softmax for roi head
    divided by batchmean, divided by 768 (256*3) for rpn, 1056 (352*3) for roi
    reduction parameter does not affect the loss

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        label (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.

    Returns:
        torch.Tensor: The calculated loss
    """

    pred_orig, pred_aug1, pred_aug2 = torch.chunk(pred, 3)

    if pred_orig.shape[-1] == 1:  # if rpn
        p_clean, p_aug1, p_aug2 = torch.cat((torch.sigmoid(pred_orig), 1 - torch.sigmoid(pred_orig)), dim=1), \
                                  torch.cat((torch.sigmoid(pred_aug1), 1 - torch.sigmoid(pred_aug1)), dim=1), \
                                  torch.cat((torch.sigmoid(pred_aug2), 1 - torch.sigmoid(pred_aug2)), dim=1),
        neg_ind = 1 # WARN: only for cityscapes dataset

    else:  # else roi
        p_clean, p_aug1, p_aug2 = F.softmax(pred_orig, dim=1), \
                                  F.softmax(pred_aug1, dim=1), \
                                  F.softmax(pred_aug2, dim=1)
        neg_ind = 8 # WARN: only for cityscapes dataset

    p_clean, p_aug1, p_aug2 = p_clean.reshape((1,) + p_clean.shape).contiguous(), \
                              p_aug1.reshape((1,) + p_aug1.shape).contiguous(), \
                              p_aug2.reshape((1,) + p_aug2.shape).contiguous()

    # Clamp mixture distribution to avoid exploding KL divergence
    p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()

    # log_pos_ratio
    loss = (F.kl_div(p_mixture, p_clean, reduction='none') +
            F.kl_div(p_mixture, p_aug1, reduction='none') +
            F.kl_div(p_mixture, p_aug2, reduction='none')) / 3.
    loss = torch.sum(loss, dim=-1).squeeze(0)

    # [DEV] imbalance: alpha-balanced loss
    label, _, _ = torch.chunk(label, 3)
    if use_cls_weight:
        class_weight = kwargs['class_weight'] \
            if kwargs['add_class_weight'] is None else kwargs['add_class_weight']

        for i in range(len(class_weight)):
            mask = (label == i)
            loss[mask] = loss[mask] * class_weight[i]

    # ANALYSIS[CODE=004]: analysis pos_ratio: the ration of positive JSD loss over negative JSD loss
    loss_neg = torch.sum(loss[(label == neg_ind)])
    loss_pos = torch.sum(loss[(label != neg_ind)])
    pos_ratio = loss_pos / loss_neg

    # Compute loss
    loss = torch.sum(loss) / len(p_aug1)

    # apply weights and do the reduction
    if weight is not None:
        weight, _, _ = torch.chunk(weight, 3)
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    p_distribution = {# 'p_clean': torch.clamp(p_clean, 1e-7, 1).log(),
                      # 'p_aug1': torch.clamp(p_aug1, 1e-7, 1).log(),
                      # 'p_aug2': torch.clamp(p_aug2, 1e-7, 1).log(),
                      # 'p_mixture': p_mixture,
                      'loss_pos': loss_pos,
                      'loss_neg': loss_neg,
                      'pos_ratio': pos_ratio, # ANALYSIS[CODE=004]
                      }

    return loss, p_distribution


def jsdv1_3_2aug(pred,
            label,
            weight=None,
            reduction='mean',
            avg_factor=None,
            use_cls_weight=False,
            **kwargs):
    """Calculate the jsdv1.3 loss.
    jsd loss (sigmoid, 1-sigmoid) for rpn head, softmax for roi head
    divided by batchmean, divided by 768 (256*3) for rpn, 1056 (352*3) for roi
    reduction parameter does not affect the loss

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        label (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.

    Returns:
        torch.Tensor: The calculated loss
    """

    pred_orig, pred_aug1 = torch.chunk(pred, 2)

    if pred_orig.shape[-1] == 1:  # if rpn
        p_clean, p_aug1 = torch.cat((torch.sigmoid(pred_orig), 1 - torch.sigmoid(pred_orig)), dim=1), \
                                  torch.cat((torch.sigmoid(pred_aug1), 1 - torch.sigmoid(pred_aug1)), dim=1)
        neg_ind = 1 # WARN: only for cityscapes dataset

    else:  # else roi
        p_clean, p_aug1 = F.softmax(pred_orig, dim=1), \
                          F.softmax(pred_aug1, dim=1)
        neg_ind = 8 # WARN: only for cityscapes dataset

    p_clean, p_aug1 = p_clean.reshape((1,) + p_clean.shape).contiguous(), \
                      p_aug1.reshape((1,) + p_aug1.shape).contiguous()

    # Clamp mixture distribution to avoid exploding KL divergence
    p_mixture = torch.clamp((p_clean + p_aug1) / 2., 1e-7, 1).log()

    # log_pos_ratio
    loss = (F.kl_div(p_mixture, p_clean, reduction='none') +
            F.kl_div(p_mixture, p_aug1, reduction='none')) / 2.
    loss = torch.sum(loss, dim=-1).squeeze(0)

    # [DEV] imbalance: alpha-balanced loss
    label, _ = torch.chunk(label, 2)
    if use_cls_weight:
        class_weight = kwargs['class_weight'] \
            if kwargs['add_class_weight'] is None else kwargs['add_class_weight']

        for i in range(len(class_weight)):
            mask = (label == i)
            loss[mask] = loss[mask] * class_weight[i]

    # ANALYSIS[CODE=004]: analysis pos_ratio: the ration of positive JSD loss over negative JSD loss
    loss_neg = torch.sum(loss[(label == neg_ind)])
    loss_pos = torch.sum(loss[(label != neg_ind)])
    pos_ratio = loss_pos / loss_neg

    # Compute loss
    loss = torch.sum(loss) / len(p_aug1)

    # apply weights and do the reduction
    if weight is not None:
        weight, _ = torch.chunk(weight, 2)
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    p_distribution = {# 'p_clean': torch.clamp(p_clean, 1e-7, 1).log(),
                      # 'p_aug1': torch.clamp(p_aug1, 1e-7, 1).log(),
                      # 'p_aug2': torch.clamp(p_aug2, 1e-7, 1).log(),
                      # 'p_mixture': p_mixture,
                      'loss_pos': loss_pos,
                      'loss_neg': loss_neg,
                      'pos_ratio': pos_ratio, # ANALYSIS[CODE=004]
                      }

    return loss, p_distribution


def analyze_shared_fcs_1input(pred,
                        label,
                        weight=None,
                        reduction='mean',
                        avg_factor=None,
                        analysis=False,
                        **kwargs):

    """Calculate the ntxent loss

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        label (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.

    Returns:
        torch.Tensor: The calculated loss
    """

    # avg_factor = None
    temper = kwargs['temper']
    add_act = kwargs['add_act']
    feature_analysis = {}
    loss = 0

    from mmdet.models.roi_heads.standard_roi_head import feature_cls_feats
    feature = feature_cls_feats

    feature_analysis = analyze_representations_1input(feature, label,)

    return loss, feature_analysis


def analyze_shared_fcs_2input(pred,
                        label,
                        weight=None,
                        reduction='mean',
                        avg_factor=None,
                        analysis=False,
                        **kwargs):

    """Calculate the ntxent loss

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        label (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.

    Returns:
        torch.Tensor: The calculated loss
    """

    # avg_factor = None
    temper = kwargs['temper']
    add_act = kwargs['add_act']
    feature_analysis = {}
    loss = 0

    from mmdet.models.roi_heads.standard_roi_head import feature_cls_feats
    feature = feature_cls_feats
    feature_clean, feature_aug1, _ = torch.chunk(feature, 3)
    label, _, _ = torch.chunk(label, 3)
    feature_analysis = analyze_representations_2input(feature_clean, feature_aug1, label,)

    pred_clean, pred_aug1, _ = torch.chunk(pred, 3)
    pred_analysis = analyze_representations_2input(pred_clean, pred_aug1, label,)
    feature_analysis.update(pred_analysis)


    return loss, feature_analysis


def ntxent_clean_fg(pred,
                    label,
                    weight=None,
                    reduction='mean',
                    avg_factor=None,
                    analysis=False,
                    num_views=3,
                    name='ntxent.clean.fg',
                    **kwargs):
    """Calculate the ntxent loss on the clean samples
    Divide feature samples by torch.chunk

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        label (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.

    Returns:
        torch.Tensor: The calculated loss
    """

    # avg_factor = None
    temper = kwargs['temper']
    add_act = kwargs['add_act']
    # temper_ratio = kwargs['temper_ratio']
    # k = kwargs['kpositive']
    # classes = kwargs['classes']
    feature_analysis = {}
    loss = 0

    from mmdet.models.detectors.base import FEATURES

    for key, feature in FEATURES.items():
        k = feature[0]
        if k.dim() == 4: # if rpn
            ntxent_loss = 0
            pass
        elif k.dim() == 2: # if roi
            features_orig = torch.chunk(k, num_views)[0]
            label_orig = torch.chunk(label, num_views)[0]
            label_orig = label_orig.unsqueeze(dim=1)
            if name == 'ntxent.clean.fg':
                ntxent_loss = supcontrast_clean_fg(features_orig, label_orig, temper=temper, min_samples=10)
            elif name == 'ntxent.clean.fg.bg':
                ntxent_loss = supcontrast_clean_fg_bg(features_orig, label_orig, temper=temper, min_samples=10)
            # supcontrast_clean_kpositivev1_1, supcontrast_clean_kpositivev1_1, supcontrastv0_01, supcontrastv0_02, supcontrast_clean

        loss += ntxent_loss

    return loss, feature_analysis


def ntxent_all(pred,
               label,
               weight=None,
               reduction='mean',
               avg_factor=None,
               analysis=False,
               num_views=3,
               name='ntxent.clean.fg',
               **kwargs):

    """Calculate the ntxent loss on the clean samples
    Divide feature samples by torch.chunk

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        label (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.

    Returns:
        torch.Tensor: The calculated loss
    """

    # avg_factor = None
    temper = kwargs['temper']
    add_act = kwargs['add_act']
    # temper_ratio = kwargs['temper_ratio']
    # k = kwargs['kpositive']
    # classes = kwargs['classes']
    feature_analysis = {}
    loss = 0

    from mmdet.models.detectors.base import FEATURES

    for key, feature in FEATURES.items():
        k = feature[0]
        if k.dim() == 4: # if rpn
            ntxent_loss = 0
            pass
        elif k.dim() == 2: # if roi
            features_orig = torch.chunk(k, num_views)[0]
            label_orig = torch.chunk(label, num_views)[0]
            label_orig = label_orig.unsqueeze(dim=1)
            if name == 'ntxent.clean.fg':
                ntxent_loss = supcontrast_clean_fg(features_orig, label_orig, temper=temper, min_samples=10)
            elif name == 'ntxent.clean.fg.bg':
                ntxent_loss = supcontrast_clean_fg_bg(features_orig, label_orig, temper=temper, min_samples=10)
            # supcontrast_clean_kpositivev1_1, supcontrast_clean_kpositivev1_1, supcontrastv0_01, supcontrastv0_02, supcontrast_clean

        loss += ntxent_loss

    return loss, feature_analysis


def assert_positive_loss(loss, **kwargs):
    def print_kwargs(keyword, save=False):
        if keyword in kwargs:
            obj = kwargs[keyword]
            if save:
                path = f'{filename}_{keyword}.pt'
                torch.save(obj, path)
                if isinstance(obj, dict):
                    f.write(f"{keyword} {obj.shape}: {path}\n")
                elif isinstance(obj, list):
                    f.write(f"{keyword} {len(obj)}: {path}\n")
                elif torch.is_tensor(obj):
                    f.write(f"{keyword} : {path}\n")
            else:
                f.write(f"{keyword} : {obj}\n")
    try:
        if loss <= 0:
            raise ValueError
        else:
            return loss
    except ValueError:
        import time
        if loss == 0:
            filename = f"/ws/data/dshong/invalid_jsd_value/{time.strftime('%Y%m%d_%H%M%S')}"
            f = open(f"{filename}.txt", 'w')
            f.write(f"[1] Information \n")
            print_kwargs('type')
            print_kwargs('name')
            f.write("\n\n")

            f.write("[2] Value\n")
            print_kwargs('pred_orig', True)
            print_kwargs('label', True)
            print_kwargs('weight', True)
            print_kwargs('pred_aug1', True)
            print_kwargs('pred_aug2', True)
            print_kwargs('class_weight', True)

            f.close()
        else:
            filename = f"/ws/data/dshong/invalid_jsd_value/negative_loss"
            f = open(f"{filename}.txt", 'a')
            f.write(f"{time.strftime('%Y%m%d_%H%M%S')}")
            f.close()
        return 0


@LOSSES.register_module()
class CrossEntropyLossPlus(nn.Module):

    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 reduction='mean',
                 class_weight=None,
                 ignore_index=None,
                 loss_weight=1.0,
                 additional_loss='jsd',
                 additional_loss_weight_reduce=False,
                 lambda_weight=0.0001,
                 additional_loss2=None,
                 lambda_weight2=0.0001,
                 kpositive=3,
                 classes=9,
                 temper=1,
                 temper_ratio=0.2,
                 analysis=False,
                 add_act=None,
                 wandb_name=None,
                 use_cls_weight=False,
                 add_class_weight=None,
                 num_views=3,
                 **kwargs):
        """CrossEntropyLossPlus.

        Args:
            use_sigmoid (bool, optional): Whether the prediction uses sigmoid
                of softmax. Defaults to False.
            use_mask (bool, optional): Whether to use mask cross entropy loss.
                Defaults to False.
            reduction (str, optional): . Defaults to 'mean'.
                Options are "none", "mean" and "sum".
            class_weight (list[float], optional): Weight of each class.
                Defaults to None.
            ignore_index (int | None): The label index to be ignored.
                Defaults to None.
            loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
            temper (int, optional): temperature scaling for softmax function.
        """
        super(CrossEntropyLossPlus, self).__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.ignore_index = ignore_index
        self.additional_loss = additional_loss
        self.additional_loss_weight_reduce = additional_loss_weight_reduce
        self.additional_loss2 = additional_loss2
        self.lambda_weight = lambda_weight
        self.lambda_weight2 = lambda_weight2
        self.kpositive = kpositive
        self.classes = classes
        self.temper = temper
        self.temper_ratio = temper_ratio
        self.analysis = analysis
        self.add_act = add_act
        self.wandb_name = wandb_name
        self.use_cls_weight = use_cls_weight
        self.add_class_weight = add_class_weight
        self.num_views = num_views

        self.kwargs = kwargs

        self.wandb_features = dict()
        self.wandb_features[f'additional_loss({self.wandb_name})'] = []
        self.wandb_features[f'lam_additional_loss({self.wandb_name})'] = []
        self.wandb_features[f'additional_loss2({self.wandb_name})'] = []
        self.wandb_features[f'lam_additional_loss2({self.wandb_name})'] = []
        self.wandb_features[f'ce_loss({self.wandb_name})'] = []
        self.wandb_features[f'additional_loss_ratio({self.wandb_name})'] = []
        self.wandb_features[f'rel_pos({self.wandb_name})'] = []
        self.wandb_features[f'rel_neg({self.wandb_name})'] = []

        self.sum_features = dict()

        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        elif self.use_mask:
            self.cls_criterion = mask_cross_entropy
        else:
            self.cls_criterion = cross_entropy

        # additional loss (jsd)
        if self.additional_loss == 'jsdv1_3':
            self.cls_additional = jsdv1_3
        elif self.additional_loss == 'jsdv1_3_2aug':
            self.cls_additional = jsdv1_3_2aug
        else:
            self.cls_additional = None

        # additional loss2: (nt-xent)
        if self.additional_loss2 == 'ntxent.clean.fg':
            self.cls_additional2 = ntxent_clean_fg
        elif self.additional_loss2 == 'ntxent.clean.fg.bg':
            self.cls_additional2 = ntxent_clean_fg
        elif self.additional_loss2 == 'analyze_shared_fcs_1input':
            self.cls_additional2 = analyze_shared_fcs_1input
        elif self.additional_loss2 == 'analyze_shared_fcs_2input':
            self.cls_additional2 = analyze_shared_fcs_2input
        else:
            self.cls_additional2 = None

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=None,
                **kwargs):
        """Forward function.

        Args:
            cls_score (torch.Tensor): The prediction.
            label (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): Sample-wise loss weight.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The method used to reduce the
                loss. Options are "none", "mean" and "sum".
            ignore_index (int | None): The label index to be ignored.
                If not None, it will override the default value. Default: None.
        Returns:
            torch.Tensor: The calculated loss.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if ignore_index is None:
            ignore_index = self.ignore_index

        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(
                self.class_weight, device=cls_score.device)
        else:
            class_weight = None

        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=kwargs['original_avg_factor'] if 'original_avg_factor' in kwargs else avg_factor,
            ignore_index=ignore_index,
            num_views=self.num_views)

        loss_additional = 0
        p_distribution = dict()
        if self.cls_additional is not None:
            if self.additional_loss_weight_reduce == False:
                weight = None
            loss_additional, p_distribution = self.cls_additional(
                cls_score,
                label,
                weight,
                reduction=reduction,
                avg_factor=avg_factor,
                temper=self.temper,
                add_act=self.add_act,
                ignore_index=ignore_index,
                class_weight=class_weight,
                lambda_weight=self.lambda_weight,
                use_cls_weight=self.use_cls_weight,
                add_class_weight=self.add_class_weight,
                )

            # wandb for rpn
            if self.use_sigmoid:
                if len(self.wandb_features[f'ce_loss({self.wandb_name})']) == 5:
                    self.wandb_features[f'ce_loss({self.wandb_name})'].clear()
                    self.wandb_features[f'additional_loss({self.wandb_name})'].clear()
                    self.wandb_features[f'lam_additional_loss({self.wandb_name})'].clear()
                    self.wandb_features[f'additional_loss_ratio({self.wandb_name})'].clear()
                self.wandb_features[f'ce_loss({self.wandb_name})'].append(loss_cls)
                self.wandb_features[f'additional_loss({self.wandb_name})'].append(loss_additional)
                self.wandb_features[f'lam_additional_loss({self.wandb_name})'].append(
                    self.lambda_weight * loss_additional)
                self.wandb_features[f'additional_loss_ratio({self.wandb_name})'].append(self.lambda_weight * loss_additional / loss_cls)
            else:
                self.wandb_features[f'ce_loss({self.wandb_name})'] = loss_cls
                self.wandb_features[f'additional_loss({self.wandb_name})'] = loss_additional
                self.wandb_features[f'lam_additional_loss({self.wandb_name})'] = self.lambda_weight * loss_additional
                self.wandb_features[f'additional_loss_ratio({self.wandb_name})'] = self.lambda_weight * loss_additional / loss_cls

            for key, value in p_distribution.items():
                self.wandb_features[f'{key}({self.wandb_name})'] = value

        loss_additional2 = 0
        if self.cls_additional2 is not None:
            if self.additional_loss_weight_reduce == False:
                weight = None
            loss_additional2, p_distribution2 = self.cls_additional2(
                cls_score,
                label,
                weight,
                reduction=reduction,
                avg_factor=avg_factor,
                temper=self.temper,
                temper_ratio=self.temper_ratio,
                add_act=self.add_act,
                ignore_index=ignore_index,
                class_weight=class_weight,
                lambda_weight=self.lambda_weight,
                kpositive=self.kpositive,
                classes=self.classes,
                num_views=self.num_views,
                name=self.additional_loss2,
            )

            # wandb for rpn
            if self.use_sigmoid:
                if len(self.wandb_features[f'ce_loss({self.wandb_name})']) == 5:
                    self.wandb_features[f'additional_loss2({self.wandb_name})'].clear()
                    self.wandb_features[f'lam_additional_loss2({self.wandb_name})'].clear()
                self.wandb_features[f'additional_loss2({self.wandb_name})'].append(loss_additional2)
                self.wandb_features[f'lam_additional_loss2({self.wandb_name})'].append(
                    self.lambda_weight2 * loss_additional2)
            else:
                self.wandb_features[f'additional_loss2({self.wandb_name})'] = loss_additional2
                self.wandb_features[f'lam_additional_loss2({self.wandb_name})'] = self.lambda_weight2 * loss_additional2

            if any(self.sum_features) == False:
                self.sum_features[f'iteration'] = 1
                for key, value in p_distribution2.items():
                    self.sum_features[f'{key}({self.wandb_name})'] = 0

            for key, value in p_distribution2.items():
                self.sum_features[f'{key}({self.wandb_name})'] += value

            self.wandb_features.update(self.sum_features)

                # self.wandb_features[f'{key}({self.wandb_name})'] = value


        loss = loss_cls + self.lambda_weight * loss_additional + self.lambda_weight2 * loss_additional2
        # self.wandb_features[f'loss({self.wandb_name})'] = loss
        # self.wandb_features[f'additional_loss({self.wandb_name})'] = loss_additional

        # For analysis_cfg: loss_weight
        if not hasattr(self, 'outputs'):
            self.outputs = dict()
            for key in ['loss_cls', 'loss_additional', 'loss_additional2']:
                self.outputs[key] = []
        self.outputs['loss_cls'].append(float(loss_cls))
        self.outputs['loss_additional'].append(float(self.lambda_weight * loss_additional))
        self.outputs['loss_additional2'].append(float(self.lambda_weight2 * loss_additional2))

        # ANALYSIS[CODE=005]: analysis relative pos and neg loss: The ration of positiv JSD loss over total loss
        if 'loss_pos' in p_distribution:
            rel_pos = p_distribution['loss_pos'] / float(loss)
            rel_neg = p_distribution['loss_neg'] / float(loss)
            if self.use_sigmoid:  # RPN
                if len(self.wandb_features[f'rel_pos({self.wandb_name})']) == 5:
                    self.wandb_features[f'rel_pos({self.wandb_name})'].clear()
                    self.wandb_features[f'rel_neg({self.wandb_name})'].clear()
                self.wandb_features[f'rel_pos({self.wandb_name})'].append(rel_pos)
                self.wandb_features[f'rel_neg({self.wandb_name})'].append(rel_neg)
            else:
                self.wandb_features[f'rel_pos({self.wandb_name})'] = rel_pos
                self.wandb_features[f'rel_neg({self.wandb_name})'] = rel_neg

        return loss

