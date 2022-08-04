import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.core import (multi_apply)
from mmdet.models.builder import build_loss

class RpnTail2():
    def __init__(self,
                 in_channels,
                 hidden_channels=3,
                 out_channels=1,
                 loss=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)):
        self.in_channels = in_channels
        # self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        # self.loss = build_loss(loss)
        #
        # self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=7, stride=1, padding=3)
        # self.prelu1 = nn.PReLU()
        #
        # self.fc1 = nn.Linear(128 * 3 * 3, 2)
        # self.prelu_fc1 = nn.PReLU()
        # self.fc2 = nn.Linear(2, num_classes)

        # self.extract_feature = nn.Linear(self.in_channels, self.out_channels)

        self.channel_wise_conv = nn.Conv2d(in_channels, 1, kernel_size=7, stride=1)
        self.maxpool = nn.MaxPool2d(kernel_size = 2)

    def forward_single(self, x):
        # N, num_anchors, H, W = x.shape # e.g., 3, 3, 224, 440
        #
        # x = x.view(N, -1)
        # feature = self.extract_feature(x)  #
        # feature_normed = feature.div(
        #     torch.norm(feature, p=2, dim=1, keepdim=True).expand_as(feature))
        x = self.channel_wise_conv(x)
        x = self.maxpool(x)

        return x

        # x = self.conv1(x)
        # x = self.prelu1(x)
        # x = F.max_pool2d(x, 2)
        #
        # x = x.view(-1, 128 * 3 * 3)
        # x = self.prelu_fc1(self.fc1(x))
        # y = self.fc2(x)
        #
        # loss = self.loss_single(x)

        return

    def forward(self, feats):
        return multi_apply(self.forward_single, feats)

    def loss_single(self, features):

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

    def loss(self, features):
        loss = multi_apply(self.loss_single, features)

        return loss

