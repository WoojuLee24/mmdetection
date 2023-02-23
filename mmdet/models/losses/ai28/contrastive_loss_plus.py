# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...builder import LOSSES
from mmdet.models.losses.ai28.contrastive_loss import supcontrast_clean_fg_bg, supcontrastv0_2, \
    supcontrastv1_0, supcontrastv1_1, supcontrastv1_2, supcontrastv1_3


@LOSSES.register_module()
class ContrastiveLossPlus(nn.Module):

    def __init__(self,
                 version,
                 loss_weight=1,
                 temperature=0.07,
                 memory=0,
                 num_classes=None,
                 dim=0,
                 num_views=1,
                 max_views=-1,
                 normalized_input=True,
                 iou_act='x',
                 iou_th=0.7,
                 **kwargs):
        """ContrastiveLossPlus."""
        super(ContrastiveLossPlus, self).__init__()
        self.version = version
        self.loss_weight = loss_weight
        self.temperature = temperature
        self.base_temperature = 1 # required?
        self.memory = memory
        self.num_views = num_views
        self.max_samples = 1024 # to do
        self.max_views = max_views  # -1: no sampling, 1: default
        self.normalized_input = normalized_input
        self.iou_act = iou_act
        self.iou_th = iou_th
        self.kwargs = kwargs

        if self.memory > 0:
            assert type(num_classes) == int
            assert dim > 0
            queue = torch.randn(num_classes-1, memory, dim) # exclude background classes
            self.queue = F.normalize(queue, p=2, dim=2)
            self.queue_ptr = torch.zeros(num_classes-1, dtype=torch.long)

        if self.version in ['0.1']:
            self.loss = supcontrast_clean_fg_bg
        elif self.version in ['0.2']:
            self.loss = supcontrastv0_2
        elif self.version in ['1.0']:
            self.loss = supcontrastv1_0
        elif self.version in ['1.1']:
            self.loss = supcontrastv1_1
        elif self.version in ['1.2']:
            self.loss = supcontrastv1_2
        elif self.version in ['1.3']:
            self.loss = supcontrastv1_3

        else:
            raise NotImplementedError(f'does not support version=={version}')

    def forward(self,
                cont_feats, pred_cls, labels, label_weights, fg_iou,
                reduction='mean', **kwargs):
        """Forward function.

        Args:
        Returns:
            torch.Tensor: The calculated loss.
        """
        if len(cont_feats) == 0:
            return torch.zeros(1)
        if self.normalized_input:
            cont_feats = F.normalize(cont_feats, dim=1)
        # Default: contrastive loss v1.0 all, fg and bg
        # feats_: [2048, 256], lables_: [2048, ]
        loss = self.loss(cont_feats, labels, temper=self.temperature, min_samples=10,
                         fg_iou=fg_iou, iou_act=self.iou_act, iou_th=self.iou_th)
        # hard anchor sampling: deprecated
        # # just split with fixed dim 512
        # total_views = cont_feats.size(0) // 512
        # if total_views != 4:
        #     print("check")
        # cont_feats = cont_feats.contiguous().view(total_views, 512, -1)
        # labels = labels.contiguous().view(total_views, -1)
        # pred_cls = pred_cls.contiguous().view(total_views, -1)
        # feats_, labels_ = self._hard_anchor_sampling(cont_feats, labels, pred_cls)
        # label_weights = None # TODO: is label_weights necessary?
        #
        # feats_ = feats_.squeeze(dim=1)
        # if self.memory > 0:
        #     q_feats, q_labels = self._sample_negative(self.queue)   # exclude background class (background mean is useless)
        #     q_feats = q_feats.contiguous().detach()
        #     q_labels = q_labels.contiguous().view(-1,)
        #     feats_ = torch.cat([feats_, q_feats], dim=0)
        #     labels_ = torch.cat([labels_, q_labels], dim=0)
        #
        # # contrastive loss v0.1 all.fg.bg
        # # feats_: [2048, 256], lables_: [2048, ]
        # loss = self.loss(feats_, labels_, temper=self.temperature, min_samples=10,
        #                  fg_iou=fg_iou, iou_act=self.iou_act, iou_th=self.iou_th)

        # Enqueue and dequeue
        if self.memory > 0: # TODO: Exception handling
            self.queue, self.queue_ptr = self._dequeue_and_enqueue(cont_feats.detach(), labels, self.queue, self.queue_ptr)

        return self.loss_weight * loss


    def _get_feature(self, feats, queue, labels):
        class_num, cache_size, feat_size = queue.shape

        X_ = torch.zeros((class_num * cache_size, feat_size)).float().cuda()    # [19, 10000, 256] -> [190000, 256]
        y_ = torch.zeros((class_num * cache_size, 1)).float().cuda() # [190000, 1]
        sample_ptr = 0
        for ii in range(class_num):
            # if ii == 0: continue # all classes will be included
            this_q = queue[ii, :cache_size, :]

            X_[sample_ptr:sample_ptr + cache_size, ...] = this_q
            y_[sample_ptr:sample_ptr + cache_size, ...] = ii
            sample_ptr += cache_size

        return X_, y_

    def _hard_anchor_sampling(self, feats, labels, preds):
        batch_size, feat_dim = feats.size(0), feats.size(-1)  # batch_size=2

        classes = []
        total_classes = 0

        if self.max_views == -1:
            # no sampling option
            feats = feats.contiguous().view(-1, 1, feat_dim)
            labels = labels.contiguous().view(-1)
            return feats, labels

        for ii in range(batch_size):
            this_labels = labels[ii]
            this_classes = torch.unique(this_labels)
            # filtering class
            this_classes = [x for x in this_classes if x >= 0]

            classes.append(this_classes)
            total_classes += len(this_classes)  # total_classes: chosen classes

        if total_classes == 0:
            return None, None

        n_view = self.max_samples // total_classes  # self.max_samples=1024
        n_view = min(n_view, self.max_views)  # self.max_views < n_view < self.max_samples // total_classes -> usually n_view = 1

        feats_ = torch.zeros((total_classes, n_view, feat_dim), dtype=torch.float).cuda()
        labels_ = torch.zeros(total_classes, dtype=torch.float).cuda()

        feats_ptr = 0
        for ii in range(batch_size):
            this_labels = labels[ii]
            this_preds = preds[ii]
            this_classes = classes[ii]

            for cls_id in this_classes:
                hard_indices = ((this_labels == cls_id) & (this_preds != cls_id)).nonzero()
                easy_indices = ((this_labels == cls_id) & (this_preds == cls_id)).nonzero()

                num_hard = hard_indices.shape[0]
                num_easy = easy_indices.shape[0]

                if num_hard >= n_view / 2 and num_easy >= n_view / 2:
                    num_hard_keep = n_view // 2
                    num_easy_keep = n_view - num_hard_keep
                elif num_hard >= n_view / 2:
                    num_easy_keep = num_easy
                    num_hard_keep = n_view - num_easy_keep
                elif num_easy >= n_view / 2:
                    num_hard_keep = num_hard
                    num_easy_keep = n_view - num_hard_keep
                else:
                    print('this shoud be never touched! {} {} {}'.format(num_hard, num_easy, n_view))
                    raise Exception

                perm = torch.randperm(num_hard)
                hard_indices = hard_indices[perm[:num_hard_keep]]
                perm = torch.randperm(num_easy)
                easy_indices = easy_indices[perm[:num_easy_keep]]
                indices = torch.cat((hard_indices, easy_indices), dim=0)

                feats_[feats_ptr, :, :] = feats[ii, indices, :].squeeze(1)  # feats: [total_classes, n_view, feat_dim)]
                labels_[feats_ptr] = cls_id  # y_: [11, ] [total_classes,)]
                feats_ptr += 1

        return feats_, labels_


    def _sample_negative(self, Q):
        class_num, cache_size, feat_size = Q.shape

        X_ = torch.zeros((class_num * cache_size, feat_size)).float().cuda()
        y_ = torch.zeros((class_num * cache_size, 1)).float().cuda()
        sample_ptr = 0
        for ii in range(class_num):
            # if ii == class_num-1: continue # exclude background
            this_q = Q[ii, :cache_size, :]

            X_[sample_ptr:sample_ptr + cache_size, ...] = this_q
            y_[sample_ptr:sample_ptr + cache_size, ...] = ii
            sample_ptr += cache_size

        return X_, y_


    def _contrastive(self, X_anchor, y_anchor, queue=None): # TODO: need to remove this function.
        anchor_num, n_view = X_anchor.shape[0], X_anchor.shape[1]

        y_anchor = y_anchor.contiguous().view(-1, 1)    # [11] -> [11, 1]
        anchor_count = n_view   # [1]
        anchor_feature = torch.cat(torch.unbind(X_anchor, dim=1), dim=0)    # [11, 256] -> [11, 256]

        if queue is not None:
            X_contrast, y_contrast = self._sample_negative(queue)   # exclude zero class    # X_contrast: [190000, 256], y_contrast: [190000, 1]
            y_contrast = y_contrast.contiguous().view(-1, 1)
            contrast_count = 1
            contrast_feature = X_contrast
        else:
            y_contrast = y_anchor
            contrast_count = n_view
            contrast_feature = torch.cat(torch.unbind(X_anchor, dim=1), dim=0)

        mask = torch.eq(y_anchor, y_contrast.T).float().cuda()  # [11, 190000]
        mask_np = mask.detach().cpu().numpy()
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T),   # [11, 256], [190000, 256] -> [11, 190000]
                                        self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        neg_mask = 1 - mask # diff class
        mask_neg_np = neg_mask.detach().cpu().numpy()

        logits_mask = torch.ones_like(mask).scatter_(1,
                                                     torch.arange(anchor_num * anchor_count).view(-1, 1).cuda(),
                                                     0) # exclude self-case only

        mask = mask * logits_mask   # same class, exclude self-case only
        mask_np2 = mask.detach().cpu().numpy()

        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True)

        exp_logits = torch.exp(logits)

        log_prob = logits - torch.log(exp_logits + neg_logits)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss


    def _dequeue_and_enqueue(self, keys, labels,
                             queue, queue_ptr):
        """
        keys: [2, 256, 128, 256]
        labels:
        segment_queue: [19, 5000, 256] : I guess [label, memory_size, feat_dims]
        segment_queue_ptr: [19]
        """
        keys = keys.permute((0, 2, 1))
        batch_size = keys.shape[0]
        feat_dim = keys.shape[1]

        for bs in range(batch_size):
            this_feat = keys[bs].contiguous().view(feat_dim, -1)
            this_label = labels[bs].contiguous().view(-1)
            this_label_ids = torch.unique(this_label)
            this_label_ids = [x for x in this_label_ids if x < self.num_classes]    # exclude background. self.num_classes=8

            for lb in this_label_ids:
                idxs = (this_label == lb).nonzero()
                # enqueue and dequeue
                feat = torch.mean(this_feat[:, idxs], dim=1).squeeze(1) # [256, 32768] -> .. -> [256]
                ptr = int(queue_ptr[lb])
                queue[lb, ptr, :] = nn.functional.normalize(feat.view(-1), p=2, dim=0)
                queue_ptr[lb] = (queue_ptr[lb] + 1) % self.memory

        return queue, queue_ptr



