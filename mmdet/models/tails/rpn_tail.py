import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.utils.visualize import multi_imsave


class RpnTail():
    def __init__(self,
                 channel_wise=None,
                 maxpool_criterion=dict(type='v1.1', kernel_size=2),
                 interpolate=dict(type='v1.1', thr_h=50, sizes=[[10,20], [5,10]]),
                 out_channels=1):
        self.out_channels = out_channels
        self.channel_wise = channel_wise
        self.maxpool_criterion = maxpool_criterion
        self.interpolate = interpolate

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

        if self.maxpool_criterion is not None:
            if self.maxpool_criterion['type'] == 'v1.1':
                maxpool = nn.MaxPool2d(self.maxpool_criterion['kernel_size'])
            elif self.maxpool_criterion['type'] == 'v1.2':
                index = 0 if H > self.maxpool_criterion['thr_h'] else 1
                maxpool = nn.MaxPool2d(self.maxpool_criterion['kernel_size'][index])
            elif self.maxpool_criterion['type'] == 'v1.3':
                maxpool = nn.AvgPool2d(self.maxpool_criterion['kernel_size'])
            else:
                raise TypeError("maxpool criterion must be ['v1.1', 'v1.2']. "
                                f"but got {self.maxpool_criterion['type']}")
            x = maxpool(x)

        # multi_imsave(x[0], 3, 1, f"maxpool({H},{W})")

        if self.interpolate is not None:
            if self.interpolate['type'] == 'v1.1':
                if H > self.interpolate['thr_h']:
                    x = F.interpolate(x, size=self.interpolate['sizes'][0])
                else:
                    x = F.interpolate(x, size=self.interpolate['sizes'][1])
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

