import torch
import torch.nn.functional as F


def supcontrast(logits_clean, logits_aug1, logits_aug2, labels=None, lambda_weight=0.1, temper=0.07, reduction='batchmean', contrast_mode='all'):

    """
    original supcontrast loss
    """

    mask = None
    # contrast_mode = 'all'
    base_temper = temper
    device = logits_clean.device

    # temporary deprecated
    logits_clean, logits_aug1, logits_aug2 = F.normalize(logits_clean, dim=1), \
                                             F.normalize(logits_aug1, dim=1), \
                                             F.normalize(logits_aug2, dim=1),


    logits_clean, logits_aug1, logits_aug2 = torch.unsqueeze(logits_clean, dim=1), \
                                             torch.unsqueeze(logits_aug1, dim=1), \
                                             torch.unsqueeze(logits_aug2, dim=1)
    features = torch.cat([logits_clean, logits_aug1, logits_aug2], dim=1)

    if len(features.shape) < 3:
        raise ValueError('`features` needs to be [bsz, n_views, ...],'
                         'at least 3 dimensions are required')
    if len(features.shape) > 3:
        features = features.view(features.shape[0], features.shape[1], -1)

    batch_size = features.shape[0]
    if labels is not None and mask is not None:
        raise ValueError('Cannot define both `labels` and `mask`')
    elif labels is None and mask is None:
        mask = torch.eye(batch_size, dtype=torch.float32).to(device)
    elif labels is not None:
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(device)
    else:
        mask = mask.float().to(device)

    contrast_count = features.shape[1]
    contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
    if contrast_mode == 'one':
        anchor_feature = features[:, 0]
        anchor_count = 1
    elif contrast_mode == 'all':
        anchor_feature = contrast_feature
        anchor_count = contrast_count
    else:
        raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

    # compute logits
    anchor_dot_contrast = torch.div(
        torch.matmul(anchor_feature, contrast_feature.T),
        temper)
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # tile mask
    mask = mask.repeat(anchor_count, contrast_count)
    # mask-out self-contrast cases
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
        0
    )
    mask = mask * logits_mask

    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

    # loss
    loss = - (temper / base_temper) * mean_log_prob_pos
    loss = loss.view(anchor_count, batch_size).mean()

    return loss



def supcontrast_maskv0_01(logits_anchor, logits_contrast, targets, mask_anchor, mask_contrast, lambda_weight=0.1, temper=0.07):

    base_temper = temper

    logits_anchor, logits_contrast = F.normalize(logits_anchor, dim=1), F.normalize(logits_contrast, dim=1)

    anchor_dot_contrast = torch.div(torch.matmul(logits_anchor, logits_contrast.T), temper)
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    exp_logits = torch.exp(logits) * mask_contrast
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
    mean_log_prob_pos = (mask_anchor * log_prob).sum(1) / mask_anchor.sum(1)
    loss = - (temper / base_temper) * mean_log_prob_pos
    loss = loss.mean()

    return loss


def supcontrastv0_01(logits_clean, logits_aug1, logits_aug2, labels=None, lambda_weight=0.1, temper=0.07, reduction='batchmean'):

    """
        supcontrast loss
        mask (mask anchor): augmented instance, original same class, augmented same class [3*B, 3*B]
        logits_mask (mask contrast): Self-instance case was excluded already, so we don't have to exclude it explicitly.
    """

    mask = None
    contrast_mode = 'all'
    base_temper = temper
    device = logits_clean.device
    batch_size = logits_clean.size()[0]
    targets = labels

    targets = targets.contiguous().view(-1, 1)
    mask_anchor = torch.eq(targets, targets.T).float()  # [B, B]
    mask_anchor_np = mask_anchor.detach().cpu().numpy()
    mask_contrast = torch.ones([batch_size, batch_size], dtype=torch.float32).to(device)
    mask_contrast_np = mask_contrast.detach().cpu().numpy()

    # mask_same_instance_diff_class_np = mask_same_instance_diff_class.detach().cpu().numpy()

    loss1 = supcontrast_maskv0_01(logits_clean, logits_aug1, targets, mask_anchor, mask_contrast, lambda_weight, temper)
    loss2 = supcontrast_maskv0_01(logits_clean, logits_aug2, targets, mask_anchor, mask_contrast, lambda_weight, temper)
    loss3 = supcontrast_maskv0_01(logits_aug1, logits_aug2, targets, mask_anchor, mask_contrast, lambda_weight, temper)

    loss = (loss1 + loss2 + loss3) / 3

    return loss


def supcontrastv0_02(logits_clean, logits_aug1, logits_aug2, labels=None, lambda_weight=0.1, temper=0.07, reduction='batchmean'):

    """
        supcontrast loss
        mask (mask anchor): augmented instance [3*B, 3*B]
        logits_mask (mask contrast): exclude only clean, augmented class case. self-instance excluded already
    """

    mask = None
    contrast_mode = 'all'
    base_temper = temper
    device = logits_clean.device
    batch_size = logits_clean.size()[0]
    targets = labels

    # temporary deprecated
    targets = targets.contiguous().view(-1, 1)
    mask_same_instance = torch.eye(batch_size, dtype=torch.float32).to(device)  # [B, B]
    mask_same_class = torch.eq(targets, targets.T).float()  # [B, B]
    # mask_same_class_np = mask_same_class.detach().cpu().numpy()
    mask_diff_class = 1 - mask_same_class  # [B, B]
    # mask_diff_class_np = mask_diff_class.detach().cpu().numpy()
    mask_same_instance_diff_class = mask_same_instance + mask_diff_class
    mask_same_instance_diff_class_np = mask_same_instance_diff_class.detach().cpu().numpy()

    loss1 = supcontrast_maskv0_01(logits_clean, logits_aug1, targets, mask_same_instance, mask_same_instance_diff_class, lambda_weight, temper)
    loss2 = supcontrast_maskv0_01(logits_clean, logits_aug2, targets, mask_same_instance, mask_same_instance_diff_class, lambda_weight, temper)
    loss3 = supcontrast_maskv0_01(logits_aug1, logits_aug2, targets, mask_same_instance, mask_same_instance_diff_class, lambda_weight, temper)

    loss = (loss1 + loss2 + loss3) / 3

    return loss