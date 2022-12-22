import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# REF: https://github.com/ErikEnglesson/GJS/blob/main/losses.py

def kl_div(pred_log, target, eps=1e-7, reduction='sum'):
    '''
    Args:
        pred_log: (tensor) log_softmaxed logits with shape of (batch_size, -1)
        target  : (tensor) with shape of (batch_size, -1)
    '''
    assert pred_log.shape == target.shape

    output_pos = target * (target.clamp(min=eps).log() - pred_log)
    zeros = torch.zeros_like(output_pos)
    output = torch.where(target > 0, output_pos, zeros)

    if reduction == 'none':
        return output
    elif reduction == 'batchmean':
        return torch.sum(output) / len(output)
    elif reduction == 'sum':
        return torch.sum(output)
    elif reduction == 'mean':
        return torch.mean(output)
    else:
        raise NotImplementedError('')


class WeightedGeneralizedJSD(nn.Module):
    def __init__(self, weights, scale=True, tolerance=0.001):
        super(WeightedGeneralizedJSD, self).__init__()
        self.weights = weights      # with length of M
        if scale:
            self.scale = -1 / ((1.0-weights[0]) * np.log(1.0-weights[0]))
        else:
            self.scale = 1.0
        assert abs(1.0 - torch.sum(weights)) < tolerance

    def forward(self, prob_list, targets, reduction='mean', eps=1e-7):
        '''
        Args:
            prob_list   : (list) which contains [p2, p3, ..., pM] with length of (M-1)
            targets     : (tensor) which is p1
        '''
        # Pre-processing: Reshape prob_list and targets
        if len(prob_list[0].shape) > 2:
            batch_size = prob_list[0].shape[0]
            prob_list = [prob.reshape(batch_size, -1) for prob in prob_list]
        if len(targets.shape) > 2:
            targets = targets.reshape(targets.shape[0], -1)

        # Distributions contain [p1, p2, ..., pM] for pi has shape of (batch_size, dim)
        assert prob_list[0].shape == targets.shape
        distribs = [targets] + prob_list
        distribs = torch.stack(distribs, dim=0)     # (M, batch_size, dim)

        # Weighted distributions: sum of {πj*pj} for j = 1, ..., M, where πj is j-th weight and pj is j-th prob_list
        # a.k.a. mixture
        assert len(self.weights) == len(distribs)
        weighted_distrib_list = []
        for i in range(len(self.weights)):
            weighted_distrib_list.append(self.weights[i] * distribs[i])
        weighted_distrib = torch.sum(torch.stack(weighted_distrib_list, dim=0), dim=0)
        weighted_distrib_log = weighted_distrib.clamp(eps, 1.0).log()       # (batch_size, dim)

        # Weighted KLD: sum of {πi*KLD(pi||sum(πj*pj))} for i = 1, ..., M
        weighted_kld_list = []
        for i in range(len(self.weights)):
            # NOTE: weighted_jsd has same shape with weighted_kld
            weighted_kld = self.weights[i] * kl_div(weighted_distrib_log, distribs[i], reduction=reduction)
            weighted_kld_list.append(weighted_kld)
        weighted_jsd = torch.stack(weighted_kld_list, dim=0)
        weighted_jsd = torch.sum(weighted_jsd, dim=0)

        return self.scale * weighted_jsd



