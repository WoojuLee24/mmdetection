import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.core import (multi_apply)
from mmdet.models.builder import build_loss

from mmdet.utils.visualize import multi_imsave


class RpnTail():
    def __init__(self,
                 channel_wise=None,
                 out_channels=1):
        self.out_channels = out_channels
        self.channel_wise = channel_wise

    def __call__(self, x):
        N, num_anchors, H, W = x.shape # e.g., 3, 3, 224, 440

        if self.channel_wise is not None:
            x = x.permute(0, 2, 3, 1)   # (N, H, W, num_anchors)
            if self.channel_wise == 'avgpool':
                cwise_criterion = nn.AvgPool2d((1, num_anchors))
            elif self.channel_wise == 'maxpool':
                cwise_criterion = nn.MaxPool2d((1, num_anchors))
            else:
                raise TypeError("channel_wise must be ['avgpool', 'maxpool']. "
                                f"but got {self.channel_wise}")
            x = cwise_criterion(x)      # (N, H, W, 1)
            x = x.permute(0, 3, 1, 2)   # (N, 1, H, W)

        maxpool = nn.MaxPool2d(2)
        x = maxpool(x)
        # multi_imsave(x[0], 3, 1, f"maxpool({H},{W})")

        if H > 50:
            x = F.interpolate(x, size=[10, 20])
        else:
            x = F.interpolate(x, size=[5, 10])
        # multi_imsave(x[0], 3, 1, f"interpolate=({H},{W})")

        x = x.reshape(N, -1)
        return x

    def loss_single(self, features):
        features = features.reshape(features.shape[0], -1)
        assert features.shape[0] == 3

        feats_orig, feats_aug1, feats_aug2 = torch.chunk(features, 3)
        p_clean, p_aug1, p_aug2 = F.softmax(feats_orig, dim=1), \
                                  F.softmax(feats_aug1, dim=1), \
                                  F.softmax(feats_aug2, dim=1)
        p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()

        kld1 = F.kl_div(p_mixture, p_clean, reduction='batchmean')
        kld2 = F.kl_div(p_mixture, p_aug1, reduction='batchmean')
        kld3 = F.kl_div(p_mixture, p_aug2, reduction='batchmean')

        loss = (kld1 + kld2 + kld3) / 3.

        return loss

