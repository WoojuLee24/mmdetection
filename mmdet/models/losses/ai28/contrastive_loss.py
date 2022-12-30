import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


def supcontrast(logits_clean, logits_aug1, logits_aug2, labels=None, lambda_weight=0.1, temper=0.07, reduction='batchmean', contrast_mode='all', eps=1e-8):

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
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + eps)

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

    # loss
    loss = - (temper / base_temper) * mean_log_prob_pos
    loss = loss.view(anchor_count, batch_size).mean()

    return loss


def make_matrix(p, q, criterion, reduction='none'):
    '''
    Input:
        p: (B,C)
        q: (B,C)
        criterion:  {class} reduction='none' is recommended.
        reduction:  {str} Reduction you want.
    By dshong
    '''
    if len(p.shape) == 1:
        p = p.unsqueeze(1)
    if len(q.shape) == 1:
        q = q.unsqueeze(1)

    assert p.size() == q.size()
    B, C = p.size()

    # 1. Expand the dimension:
    #    p: (B,C) →[unsqueeze]→ (B,1,C) →[repeat]→ (B,B,C) →[reshape]→ (B*B,C)
    #    q: (B,C) →[unsqueeze]→ (1,B,C) →[repeat]→ (B,B,C) →[reshape]→ (B*B,C)
    p_exp = p.unsqueeze(1).repeat(1, B, 1).reshape(-1, C)
    q_exp = q.unsqueeze(0).repeat(B, 1, 1).reshape(-1, C)

    # 2. Compute with criterion:
    matrix = criterion(q_exp, p_exp)

    # 3. Do reduction
    if reduction == 'none':
        matrix = matrix.reshape(B, B, -1)
    elif reduction == 'mean':
        matrix = torch.mean(matrix, dim=-1)
        matrix = matrix.reshape(B, B)
    elif reduction == 'sum':
        matrix = torch.sum(matrix, dim=-1)
        matrix = matrix.reshape(B, B)
    else:
        raise ValueError(f'unexpected reduction type {reduction}.')

    return matrix


def analyze_representations_1input(logits, labels=None, lambda_weight=12, temper=1.0, reduction='batchmean'):
    '''
    logging representations by jsdv4 and L2 distance
    1 inputs
    '''

    device = logits.device
    batch_size = logits.size()[0]
    targets = labels

    pred = logits.data.max(1)[1]
    logits = logits.detach()

    # logging
    batch_size = logits.size()[0]
    targets = targets.contiguous().view(-1, 1)  # [B, 1]
    temper = 1.0

    # mask
    mask_identical = torch.ones([batch_size, batch_size], dtype=torch.float32).to(device)
    mask_triu = torch.triu(mask_identical.clone().detach())
    mask_same_instance = torch.eye(batch_size, dtype=torch.float32).to(device)  # [B, B]
    mask_triuu = mask_triu - mask_same_instance
    mask_same_class = torch.eq(targets, targets.T).float()  # [B, B]
    mask_same_triuu = mask_same_class * mask_triuu
    mask_diff_class = 1 - mask_same_class  # [B, B]
    mask_diff_triuu = mask_diff_class * mask_triuu

    # softmax
    p_clean = F.softmax(logits / temper, dim=1)

    # # JSD matrix
    # jsd_matrix = make_matrix(p_clean, p_clean, criterion=nn.KLDivLoss(reduction='none'), reduction='sum')
    #
    # jsd_matrix_diff_class = jsd_matrix * mask_diff_triuu
    # jsd_distance_diff_class = jsd_matrix_diff_class.sum() / mask_diff_triuu.sum().detach()
    #
    # jsd_matrix_same_class = jsd_matrix * mask_same_triuu
    # jsd_distance_same_class = jsd_matrix_same_class.sum() / mask_same_triuu.sum().detach()

    # MSE matrix
    mse_matrix = make_matrix(logits, logits, criterion=nn.MSELoss(reduction='none'), reduction='mean')

    mse_matrix_diff_class = mse_matrix * mask_diff_triuu
    mse_distance_diff_class = mse_matrix_diff_class.sum() # / mask_diff_triuu.sum().detach()

    mse_matrix_same_class = mse_matrix * mask_same_triuu
    mse_distance_same_class = mse_matrix_same_class.sum() # / mask_same_triuu.sum().detach()

    # Cosine Similarity matrix
    cs_matrix = make_matrix(logits, logits, criterion=nn.CosineSimilarity(dim=1), reduction='none')
    cs_matrix = cs_matrix.squeeze(dim=-1)

    cs_matrix_diff_class = cs_matrix * mask_diff_triuu
    cs_distance_diff_class = cs_matrix_diff_class.sum() # / mask_diff_triuu.sum().detach()

    cs_matrix_same_class = cs_matrix * mask_same_triuu
    cs_distance_same_class = cs_matrix_same_class.sum() # / mask_same_triuu.sum().detach()

    # class-wise distance
    # confusion_matrix_jsd = torch.zeros(9, 9)
    confusion_matrix_l2 = torch.zeros(9, 9)
    confusion_matrix_cs = torch.zeros(9, 9)
    confusion_matrix_sample_number = torch.zeros(9, 9)

    B, _ = targets.size()
    targets1 = targets.repeat(1, B).unsqueeze(0)
    targets2 = targets.T.repeat(B, 1).unsqueeze(0)
    target_matrix = torch.cat([targets1, targets2], dim=0) # class index of batch sampe (2, 512, 512) (target, target) tuple
    target_matrix_np = target_matrix.detach().cpu().numpy()


    for i in range(9):
        for j in range(9):
            a = target_matrix[0, :, :] == i
            b = target_matrix[1, :, :] == j
            class_mask = a & b

            # class_jsd_matrix = jsd_matrix * class_mask
            class_mse_matrix = mse_matrix * class_mask
            class_cs_matrix = cs_matrix * class_mask

            # confusion_matrix_jsd[i, j] = class_jsd_matrix.sum()
            confusion_matrix_l2[i, j] = torch.sqrt(class_mse_matrix).sum()
            confusion_matrix_cs[i, j] = class_cs_matrix.sum()
            confusion_matrix_sample_number[i, j] = class_mask.sum()


    features = {
                # 'jsd_distance_diff_class': jsd_distance_diff_class.detach(),
                # 'jsd_distance_same_class': jsd_distance_same_class.detach(),
                'mse_distance_diff_class': mse_distance_diff_class.detach().cpu().numpy(),
                'mse_distance_same_class': mse_distance_same_class.detach().cpu().numpy(),
                'cs_distance_diff_class': cs_distance_diff_class.detach().cpu().numpy(),
                'cs_distance_same_class': cs_distance_same_class.detach().cpu().numpy(),
                # 'confusion_matrix_jsd': confusion_matrix_jsd.detach(),
                'confusion_matrix_l2': confusion_matrix_l2.detach().cpu().numpy(),
                'confusion_matrix_cs': confusion_matrix_cs.detach().cpu().numpy(),
                'matrix_sample_number': confusion_matrix_sample_number.detach().cpu().numpy(),
                }

    return features


def analyze_representations_2input(logits_clean, logits_aug1, labels=None, lambda_weight=12, temper=1.0, reduction='batchmean'):
    '''
    logging representations by jsdv4 and L2 distance
    3 inputs
    '''

    device = logits_clean.device
    targets = labels

    pred_clean = logits_clean.data.max(1)[1]
    pred_aug1 = logits_aug1.data.max(1)[1]

    logits_clean = logits_clean.detach()
    logits_aug1 = logits_aug1.detach()

    # logging
    batch_size = logits_clean.size()[0]
    targets = targets.contiguous().view(-1, 1)  # [B, 1]
    temper = 1.0

    # mask
    mask_identical = torch.ones([batch_size, batch_size], dtype=torch.float32).to(device)
    mask_triu = torch.triu(mask_identical.clone().detach())
    mask_same_instance = torch.eye(batch_size, dtype=torch.float32).to(device)  # [B, B]
    mask_triuu = mask_triu - mask_same_instance
    mask_same_class = torch.eq(targets, targets.T).float()  # [B, B]
    mask_same_triuu = mask_same_class * mask_triuu
    mask_diff_class = 1 - mask_same_class  # [B, B]
    mask_diff_triuu = mask_diff_class * mask_triuu

    # softmax
    p_clean, p_aug1  = F.softmax(logits_clean / temper, dim=1), \
                       F.softmax(logits_aug1 / temper, dim=1)

    # Clamp mixture distribution to avoid exploding KL divergence
    p_mixture = torch.clamp((p_clean + p_aug1) / 2., 1e-7, 1).log()

    # JSD matrix
    jsd_matrix = (make_matrix(p_clean, p_mixture, criterion=nn.KLDivLoss(reduction='none'), reduction='sum') + \
                  make_matrix(p_aug1, p_mixture, criterion=nn.KLDivLoss(reduction='none'), reduction='sum')) / 2

    jsd_matrix_same_instance = jsd_matrix * mask_same_instance
    jsd_distance = jsd_matrix_same_instance.sum()

    jsd_matrix_diff_class = jsd_matrix * mask_diff_triuu
    jsd_distance_diff_class = jsd_matrix_diff_class.sum()

    jsd_matrix_same_class = jsd_matrix * mask_same_triuu
    jsd_distance_same_class = jsd_matrix_same_class.sum()

    # MSE matrix
    mse_matrix = make_matrix(logits_clean, logits_aug1, criterion=nn.MSELoss(reduction='none'), reduction='mean')

    mse_matrix_same_instance = mse_matrix * mask_same_instance
    mse_distance = mse_matrix_same_instance.sum()

    mse_matrix_diff_class = mse_matrix * mask_diff_triuu
    mse_distance_diff_class = mse_matrix_diff_class.sum()

    mse_matrix_same_class = mse_matrix * mask_same_triuu
    mse_distance_same_class = mse_matrix_same_class.sum()

    # Cosine Similarity matrix
    cs_matrix = make_matrix(logits_clean, logits_aug1, criterion=nn.CosineSimilarity(dim=1), reduction='none')
    cs_matrix = cs_matrix.squeeze(dim=-1)

    cs_matrix_same_instance = cs_matrix * mask_same_instance
    cs_distance = cs_matrix_same_instance.sum()

    cs_matrix_diff_class = cs_matrix * mask_diff_triuu
    cs_distance_diff_class = cs_matrix_diff_class.sum()

    cs_matrix_same_class = cs_matrix * mask_same_triuu
    cs_distance_same_class = cs_matrix_same_class.sum()

    # class-wise distance
    confusion_matrix_jsd = torch.zeros(9, 9)
    confusion_matrix_l2 = torch.zeros(9, 9)
    confusion_matrix_cs = torch.zeros(9, 9)
    confusion_matrix_sample_number = torch.zeros(9, 9)

    B, _ = targets.size()
    targets1 = targets.repeat(1, B).unsqueeze(0)
    targets2 = targets.T.repeat(B, 1).unsqueeze(0)
    target_matrix = torch.cat([targets1, targets2], dim=0) # class index of batch sampe (2, 512, 512) (target, target) tuple
    target_matrix_np = target_matrix.detach().cpu().numpy()


    for i in range(9):
        for j in range(9):
            a = target_matrix[0, :, :] == i
            b = target_matrix[1, :, :] == j
            class_mask = a & b

            class_jsd_matrix = jsd_matrix * class_mask
            class_mse_matrix = mse_matrix * class_mask
            class_cs_matrix = cs_matrix * class_mask

            confusion_matrix_jsd[i, j] = class_jsd_matrix.sum()
            confusion_matrix_l2[i, j] = torch.sqrt(class_mse_matrix).sum()
            confusion_matrix_cs[i, j] = class_cs_matrix.sum()
            confusion_matrix_sample_number[i, j] = class_mask.sum()


    features = {'jsd_distance': jsd_distance.detach().cpu().numpy(),
                'jsd_distance_diff_class': jsd_distance_diff_class.detach().cpu().numpy(),
                'jsd_distance_same_class': jsd_distance_same_class.detach().cpu().numpy(),
                'mse_distance': mse_distance.detach().cpu().numpy(),
                'mse_distance_diff_class': mse_distance_diff_class.detach().cpu().numpy(),
                'mse_distance_same_class': mse_distance_same_class.detach().cpu().numpy(),
                'confusion_matrix_jsd': confusion_matrix_jsd.detach().cpu().numpy(),
                'confusion_matrix_l2': confusion_matrix_l2.detach().cpu().numpy(),
                'confusion_matrix_cs': confusion_matrix_cs.detach().cpu().numpy(),
                'matrix_sample_number': confusion_matrix_sample_number.detach().cpu().numpy(),
                }

    return features

def analyze_representations_3input(logits_clean, logits_aug1, logits_aug2, labels=None, lambda_weight=12, temper=1.0, reduction='batchmean'):
    '''
    logging representations by jsdv4 and L2 distance
    3 inputs
    '''

    device = logits_clean.device
    batch_size = logits_clean.size()[0]
    targets = labels

    pred_clean = logits_clean.data.max(1)[1]
    pred_aug1 = logits_aug1.data.max(1)[1]
    pred_aug2 = logits_aug2.data.max(1)[1]

    logits_clean = logits_clean.detach()
    logits_aug1 = logits_aug1.detach()
    logits_aug2 = logits_aug2.detach()

    # ntxent_loss = supcontrast(logits_clean, logits_aug1, logits_aug2, targets, lambda_weight, temper, reduction='batchmean')

    # logging
    batch_size = logits_clean.size()[0]
    targets = targets.contiguous().view(-1, 1)  # [B, 1]
    temper = 1.0

    # mask
    mask_identical = torch.ones([batch_size, batch_size], dtype=torch.float32).to(device)
    mask_triu = torch.triu(mask_identical.clone().detach())
    mask_same_instance = torch.eye(batch_size, dtype=torch.float32).to(device)  # [B, B]
    mask_triuu = mask_triu - mask_same_instance
    mask_same_class = torch.eq(targets, targets.T).float()  # [B, B]
    mask_same_triuu = mask_same_class * mask_triuu
    mask_diff_class = 1 - mask_same_class  # [B, B]
    mask_diff_triuu = mask_diff_class * mask_triuu

    # softmax
    p_clean, p_aug1, p_aug2 = F.softmax(logits_clean / temper, dim=1), \
                              F.softmax(logits_aug1 / temper, dim=1), \
                              F.softmax(logits_aug2 / temper, dim=1)

    # Clamp mixture distribution to avoid exploding KL divergence
    p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()

    # JSD matrix
    jsd_matrix = (make_matrix(p_clean, p_mixture, criterion=nn.KLDivLoss(reduction='none'), reduction='sum') + \
                  make_matrix(p_aug1, p_mixture, criterion=nn.KLDivLoss(reduction='none'), reduction='sum') + \
                  make_matrix(p_aug2, p_mixture, criterion=nn.KLDivLoss(reduction='none'), reduction='sum')) / 3.

    jsd_matrix_same_instance = jsd_matrix * mask_same_instance
    jsd_distance = jsd_matrix_same_instance.sum() / mask_same_instance.sum().detach()

    jsd_matrix_diff_class = jsd_matrix * mask_diff_triuu
    jsd_distance_diff_class = jsd_matrix_diff_class.sum() / mask_diff_triuu.sum().detach()

    jsd_matrix_same_class = jsd_matrix * mask_same_triuu
    jsd_distance_same_class = jsd_matrix_same_class.sum() / mask_same_triuu.sum().detach()

    # MSE matrix
    mse_matrix = (make_matrix(p_clean, p_aug1, criterion=nn.MSELoss(reduction='none'), reduction='mean') + \
                  make_matrix(p_aug1, p_aug2, criterion=nn.MSELoss(reduction='none'), reduction='mean') + \
                  make_matrix(p_aug2, p_clean, criterion=nn.MSELoss(reduction='none'), reduction='mean')) / 3.

    mse_matrix_same_instance = mse_matrix * mask_same_instance
    mse_distance = mse_matrix_same_instance.sum() / mask_same_instance.sum().detach()

    mse_matrix_diff_class = mse_matrix * mask_diff_triuu
    mse_distance_diff_class = mse_matrix_diff_class.sum() / mask_diff_triuu.sum().detach()

    mse_matrix_same_class = mse_matrix * mask_same_triuu
    mse_distance_same_class = mse_matrix_same_class.sum() / mask_same_triuu.sum().detach()

    # Cosine Similarity matrix
    cs_matrix = (make_matrix(p_clean, p_aug1, criterion=nn.CosineSimilarity(dim=1), reduction='none') + \
                 make_matrix(p_aug1, p_aug2, criterion=nn.CosineSimilarity(dim=1), reduction='none') + \
                 make_matrix(p_aug2, p_clean, criterion=nn.CosineSimilarity(dim=1), reduction='none')) / 3.
    cs_matrix = cs_matrix.squeeze(dim=-1)

    cs_matrix_same_instance = cs_matrix * mask_same_instance
    cs_distance = cs_matrix_same_instance.sum() / mask_same_instance.sum().detach()

    cs_matrix_diff_class = cs_matrix * mask_diff_triuu
    cs_distance_diff_class = cs_matrix_diff_class.sum() / mask_diff_triuu.sum().detach()

    cs_matrix_same_class = cs_matrix * mask_same_triuu
    cs_distance_same_class = cs_matrix_same_class.sum() / mask_same_triuu.sum().detach()

    # class-wise distance
    confusion_matrix_jsd = torch.zeros(9, 9)
    confusion_matrix_l2 = torch.zeros(9, 9)
    confusion_matrix_cs = torch.zeros(9, 9)
    confusion_matrix_sample_number = torch.zeros(9, 9)

    B, _ = targets.size()
    targets1 = targets.repeat(1, B).unsqueeze(0)
    a = targets.T
    targets2 = targets.T.repeat(B, 1).unsqueeze(0)
    target_matrix = torch.cat([targets1, targets2], dim=0) # class index of batch sampe (2, 512, 512) (target, target) tuple
    target_matrix_np = target_matrix.detach().cpu().numpy()


    for i in range(9):
        for j in range(9):
            a = target_matrix[0, :, :] == i
            b = target_matrix[1, :, :] == j
            class_mask = a & b

            class_jsd_matrix = jsd_matrix * class_mask
            class_mse_matrix = mse_matrix * class_mask
            class_cs_matrix = cs_matrix * class_mask

            confusion_matrix_jsd[i, j] = class_jsd_matrix.sum()
            confusion_matrix_l2[i, j] = torch.sqrt(class_mse_matrix).sum()
            confusion_matrix_cs[i, j] = class_cs_matrix.sum()
            confusion_matrix_sample_number[i, j] = class_mask.sum()

    # confusion_matrix_jsd = confusion_matrix_jsd / (confusion_matrix_sample_number + 1e-8)
    # confusion_matrix_l2 = confusion_matrix_l2 / (confusion_matrix_sample_number + 1e-8)
    # confusion_matrix_cs = confusion_matrix_cs / (confusion_matrix_sample_number + 1e-8)
    #
    # f = confusion_matrix_l2.detach().cpu().numpy()
    # g = confusion_matrix_sample_number.detach().cpu().numpy()



    features = {'jsd_distance': jsd_distance.detach(),
                'jsd_distance_diff_class': jsd_distance_diff_class.detach(),
                'jsd_distance_same_class': jsd_distance_same_class.detach(),
                'mse_distance': mse_distance.detach(),
                'mse_distance_diff_class': mse_distance_diff_class.detach(),
                'mse_distance_same_class': mse_distance_same_class.detach(),
                'confusion_matrix_jsd': confusion_matrix_jsd.detach(),
                'confusion_matrix_l2': confusion_matrix_l2.detach(),
                'confusion_matrix_cs': confusion_matrix_cs.detach(),
                'matrix_sample_number': confusion_matrix_sample_number.detach(),
                }

    return features


def supcontrast_maskv0_01(logits_anchor, logits_contrast, targets, mask_anchor, mask_contrast, lambda_weight=0.1, temper=0.07):

    base_temper = temper

    logits_anchor, logits_contrast = F.normalize(logits_anchor, dim=1), F.normalize(logits_contrast, dim=1)

    anchor_dot_contrast = torch.div(torch.matmul(logits_anchor, logits_contrast.T), temper)
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    exp_logits = torch.exp(logits) * mask_contrast
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
    mean_log_prob_pos = (mask_anchor * log_prob).sum(1) / (mask_anchor.sum(1) + 1e-8)
    loss = - (temper / base_temper) * mean_log_prob_pos
    loss = loss.mean()

    return loss


def supcontrast_clean(logits_clean, labels=None, lambda_weight=0.1, temper=0.07, reduction='batchmean'):

    """
        supcontrast loss
        input: only clean logit
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

    mask_eye = torch.eye(batch_size, dtype=torch.float32).to(device)
    mask_anchor = torch.eq(targets, targets.T).float()  # [B, B]
    mask_anchor_except_eye = mask_anchor - mask_eye
    # mask_anchor_np = mask_anchor.detach().cpu().numpy()
    mask_contrast = torch.ones([batch_size, batch_size], dtype=torch.float32).to(device)
    mask_contrast_except_eye = mask_contrast - mask_eye
    # mask_contrast_np = mask_contrast.detach().cpu().numpy()

    # mask_same_instance_diff_class_np = mask_same_instance_diff_class.detach().cpu().numpy()

    loss1 = supcontrast_maskv0_01(logits_clean, logits_clean, targets,
                                  mask_anchor_except_eye, mask_contrast_except_eye, lambda_weight, temper)

    loss = loss1

    return loss


def supcontrast_clean_kpositive(logits_clean, labels=None, k=3, classes=9, lambda_weight=0.1, temper=0.07, reduction='batchmean'):

    """
        supcontrast loss
        input: only clean logit
        mask (mask anchor): augmented instance, original same class, augmented same class [3*B, 3*B]
        logits_mask (mask contrast): Self-instance case was excluded already, so we don't have to exclude it explicitly.
    """

    mask = None
    contrast_mode = 'all'
    base_temper = temper
    device = logits_clean.device
    batch_size = logits_clean.size()[0]
    targets = labels

    kpositive_class_targets = torch.zeros_like(targets, dtype=torch.float32)
    for i in range(classes):
        class_targets, _ = (targets==i).nonzero(as_tuple=True)
        if class_targets.size(0) == 0:
            pass
        elif class_targets.size(0) < k:
            kpositive_class_targets[class_targets, 0] = 1
        else:
            sample_choice = np.random.choice(class_targets.detach().cpu().numpy(), k, replace=False)
            kpositive_class_targets[sample_choice, 0] = 1
    kpositive_class_targets = kpositive_class_targets.contiguous().view(-1, 1)
    kpositive_class_anchor = torch.matmul(kpositive_class_targets, kpositive_class_targets.T).float()
    # kpositive_class_anchor_npy = kpositive_class_anchor.detach().cpu().numpy()
    targets = targets.contiguous().view(-1, 1)


    mask_eye = torch.eye(batch_size, dtype=torch.float32).to(device)
    mask_anchor = torch.eq(targets, targets.T).float()  # [B, B]
    mask_anchor_except_eye = mask_anchor - mask_eye
    # mask_anchor_np = mask_anchor.detach().cpu().numpy()
    mask_contrast = torch.ones([batch_size, batch_size], dtype=torch.float32).to(device)
    mask_contrast_except_eye = mask_contrast - mask_eye
    # mask_contrast_np = mask_contrast.detach().cpu().numpy()

    kmask_anchor = mask_anchor_except_eye * kpositive_class_anchor
    # kamsk_anchor_npy = kmask_anchor.detach().cpu().numpy()

    # mask_same_instance_diff_class_np = mask_same_instance_diff_class.detach().cpu().numpy()

    loss1 = supcontrast_maskv0_01(logits_clean, logits_clean, targets,
                                  kmask_anchor, mask_contrast_except_eye, lambda_weight, temper)

    loss = loss1

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