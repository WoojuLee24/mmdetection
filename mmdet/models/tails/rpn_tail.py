import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.utils.visualize import multi_imsave


class RpnTail():
    def __init__(self,
                 channel_wise=None,
                 maxpool_criterion=dict(type='v1.1', kernel_size=2),
                 interpolate=dict(type='v1.1', thr_h=50, sizes=[[10,20], [5,10]]),
                 linear_criterion=None, # dict(type='v1.1', out_channels=[200, 50])
                 out_channels=1):
        self.out_channels = out_channels
        self.channel_wise = channel_wise
        self.maxpool_criterion = maxpool_criterion
        self.interpolate = interpolate
        self.linear_criterion = linear_criterion

        if self.linear_criterion is not None:
            num_anchors = 3 if channel_wise is None else 1
            if self.interpolate is not None:
                if 'thr_h' in self.interpolate:
                    h1, w1 = self.interpolate['sizes'][0]
                    h2, w2 = self.interpolate['sizes'][1]
                else:
                    h1, w1 = self.interpolate['size']
            else:
                raise ValueError("linear_criterion can be used when interpolate is not None.")
            if 'out_channels' in linear_criterion:
                if isinstance(linear_criterion['out_channels'], list):
                    assert len(linear_criterion['out_channels']) == len(interpolate['sizes'])
                    self.linear1 = nn.Linear(num_anchors * h1 * w1, linear_criterion['out_channels'][0])
                    self.linear2 = nn.Linear(num_anchors * h2 * w2, linear_criterion['out_channels'][1])
                elif isinstance(self.linear_criterion['out_channels'], int):
                    assert isinstance(interpolate['sizes'], int)
                    self.linear1 = nn.Linear(num_anchors * h1 * w1, linear_criterion['out_channels'])
                    self.linear2 = None
                else:
                    raise TypeError("'out_channels' in linear_criterion must be list or int. "
                                    f"but got {type(linear_criterion['out_channels'])}.")
            else:
                raise ValueError("linear_criterion must have 'out_channels'. "
                                 f"but it doesn't have.")

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

        if self.linear_criterion is not None:
            if self.linear2 is None:
                x = self.linear1(x)
            else:
                if H > self.interpolate['thr_h']:
                    x = self.linear1(x)
                else:
                    x = self.linear2(x)

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

