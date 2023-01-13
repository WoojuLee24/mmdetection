# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...builder import LOSSES
from mmdet.models.losses.ai28.contrastive_loss import supcontrast_clean_fg_bg


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
        self.kwargs = kwargs

        if self.memory > 0:
            assert type(num_classes) == int
            assert dim > 0
            queue = torch.randn(num_classes-1, memory, dim) # exclude background classes
            self.queue = F.normalize(queue, p=2, dim=2)
            self.queue_ptr = torch.zeros(num_classes-1, dtype=torch.long)

        if self.version in ['1.1']: # supcontrast_clean_fg
            anchor_target_view, anchor_target_class = 'orig', 'fg'
            contrast_target_view, contrast_target_class = 'orig', 'fg'
            target_mask_pos = 'same_class_different_instance'
            target_mask_neg = 'different_instance'
        elif self.version in ['1.2']: # supcontrast_all_fg_bg
            anchor_target_view, anchor_target_class = 'all', 'fg'
            contrast_target_view, contrast_target_class = 'all', 'all'
            target_mask_pos = 'same_fg_class_different_instance' # TODO
            target_mask_neg = 'different_instance'
        elif self.version in ['0.1']:
            print('temporary code. supcontrast.clean.fg.bg function is used')
            # v0.1: all (orig + aug1) + queue => (2048 + mem * 8) x (2048 + mem * 8)
            # queue: only fg mean feature
            # loss: ntxent.all.fg.bg
            # pos: fg, neg: fg and bg
            anchor_target_view, anchor_target_class = 'all', 'fg'
            contrast_target_view, contrast_target_class = 'all', 'all'
            target_mask_pos = 'same_fg_class_different_instance'  # TODO
            target_mask_neg = 'different_instance'
        else:
            raise NotImplementedError(f'does not support version=={version}')

        # anchor and contrast
        self.anchor_target_view, self.anchor_target_class = anchor_target_view, anchor_target_class
        self.contrast_target_view, self.contrast_target_class = contrast_target_view, contrast_target_class

        # mask
        valid_target_mask = ['same_instance', 'different_instance',
                             'same_class', 'different_class',
                             'same_class_different_instance',
                             'same_fg_class_different_instance']
        self.target_mask_pos = target_mask_pos
        self.target_mask_neg = target_mask_neg
        if (not self.target_mask_pos in valid_target_mask) or (not self.target_mask_neg in valid_target_mask):
            raise ValueError(f'only support for [{",".join([str(target) for target in valid_target_mask])}],')

    def make_mask(self, type, device='cuda',
                   mask_size=None,
                   labels_anchor=None, labels_contrast=None,):
        if isinstance(mask_size, tuple):
            N, M = mask_size
        else:
            N = M = mask_size

        if type == 'all':
            return torch.ones(N, M, dtype=torch.float32).to(device)
        elif type == 'none':
            return (self.make_mask('all', device=device, mask_size=mask_size) == 0)
        elif type == 'same_instance':
            return torch.eye(N, M, dtype=torch.float32).to(device)
        elif type == 'different_instance':
            return (self.make_mask('same_instance', device=device, mask_size=mask_size)) == 0
        elif type == 'same_class':
            return torch.eq(labels_anchor, labels_contrast.T).float()
        elif type == 'different_class':
            return (self.make_mask('same_class', device=device,
                                    labels_anchor=labels_anchor,
                                    labels_contrast=labels_contrast) == 0)
        elif type == 'same_class_different_instance':
            mask_same_instance = self.make_mask('same_instance', device=device,
                                                 mask_size=mask_size)
            mask_same_class = self.make_mask('same_class', device=device,
                                              labels_anchor=labels_anchor,
                                              labels_contrast=labels_contrast)
            return mask_same_class * (mask_same_instance == 0)
        elif type == 'same_fg_class_different_instance': # TODO: need to validate
            mask_same_class_different_instance = self.make_mask('same_class_different_instance',
                                                                device=device, mask_size=mask_size,
                                                                labels_anchor=labels_anchor,
                                                                labels_contrast=labels_contrast)
            anchor_bg_inds = (labels_anchor == self.num_classes).squeeze()
            contrast_bg_inds = (labels_contrast == self.num_classes).squeeze()
            mask_same_class_different_instance[anchor_bg_inds, :] = 0
            mask_same_class_different_instance[: contrast_bg_inds] = 0
            return mask_same_class_different_instance
        else:
            raise TypeError('')

    def get_cont_target(self, cont_feats, labels, label_weights,
                        target_view, target_class):
        '''
        Args:
            cont_feats: (num_views, batch_size, dim_feat)
            labels: (num_views * batch_size, 1)
        '''
        dim_feat = cont_feats.shape[-1]

        if target_view == 'orig':
            _cont_feats = cont_feats[0]
            _labels = labels[0]
            _label_weights = label_weights[0] if label_weights is not None else label_weights
        elif target_view == 'aug2':
            _cont_feats = cont_feats[1]
            _labels = labels[1]
            _label_weights = label_weights[1] if label_weights is not None else label_weights
        elif target_view == 'all':
            _cont_feats = cont_feats
            _labels = labels
            _label_weights = label_weights
        else:
            raise TypeError
        _cont_feats = _cont_feats.reshape(-1, dim_feat)
        _labels = _labels.reshape(-1, 1)

        if target_class == 'all':
            inds = ((_labels >= 0) & (_labels <= self.num_classes)).squeeze()
        elif target_class == 'fg':
            inds = ((_labels >= 0) & (_labels < self.num_classes)).squeeze()
        elif target_class == 'bg':
            inds = (_labels == self.num_classes).squeeze()
        else:
            raise TypeError

        _cont_feats = _cont_feats[inds]
        _labels = _labels[inds]
        _label_weights = _label_weights[inds] if label_weights is not None else label_weights

        return _cont_feats, _labels, _label_weights

    def loss(self, cont_feats, labels, label_weights, reduction='mean', **kwargs):
        # Settings
        base_temper = temper = self.temperature
        device = cont_feats.get_device()

        dim_feat = cont_feats.shape[-1]
        batch_size, _r = divmod(len(cont_feats), self.num_views)
        assert _r == 0

        # Pre-Processing: Reshape
        cont_feats = cont_feats.reshape(self.num_views, -1, dim_feat)
        labels = labels.reshape(self.num_views, -1, 1)
        label_weights = label_weights.reshape(self.num_views, -1) if label_weights is not None else label_weights

        # anchor, contrast
        anchor, labels_anchor, label_weights_anchor = self.get_cont_target(
            cont_feats, labels, label_weights, self.anchor_target_view, self.anchor_target_class)
        contrast, labels_contrast, label_weights_contrast = self.get_cont_target(
            cont_feats, labels, label_weights, self.contrast_target_view, self.contrast_target_class)

        if self.memory > 0:
            # queue
            q_feats, q_labels = self._sample_negative(self.queue)
            q_feats = q_feats.contiguous()
            q_labels = q_labels.contiguous()
            anchor = torch.cat([anchor, q_feats], dim=0)
            labels_anchor = torch.cat([labels_anchor, q_labels], dim=0)
            contrast = torch.cat([contrast, q_feats], dim=0)
            labels_contrast = torch.cat([labels_contrast, q_labels], dim=0)

        # Generate mask
        mask_size = (len(anchor), len(contrast))
        mask_pos = self.make_mask(self.target_mask_pos, device=device, mask_size=mask_size,
                                  labels_anchor=labels_anchor, labels_contrast=labels_contrast)
        mask_neg = self.make_mask(self.target_mask_neg, device=device, mask_size=mask_size,
                                  labels_anchor=labels_anchor, labels_contrast=labels_contrast)

        # Compute contrastive loss
        anchor_dot_contrast = torch.div(torch.matmul(anchor, contrast.T), temper)  # (Na, Nc)
        max_similarity, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        similarity = anchor_dot_contrast - max_similarity.detach()

        neg_similarity = torch.exp(similarity) * mask_neg
        neg_similarity = torch.log(neg_similarity.sum(1, keepdim=True))
        log_prob = mask_pos * (similarity - neg_similarity)
        mean_log_prob = log_prob.sum(1) / (mask_pos.sum(1) + 1e-8)
        loss = - (temper / base_temper) * mean_log_prob

        # Reduction
        if reduction == 'sum':
            loss = loss.sum()
        elif reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'none':
            loss = loss
        else:
            raise TypeError

        return loss

    def forward(self,
                cont_feats, pred_cls, labels, label_weights,
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
        # hard anchor sampling
        # just split with fixed dim 512
        total_views = cont_feats.size(0) // 512
        cont_feats = cont_feats.contiguous().view(total_views, 512, -1)
        labels = labels.contiguous().view(total_views, -1)
        pred_cls = pred_cls.contiguous().view(total_views, -1)
        feats_, labels_ = self._hard_anchor_sampling(cont_feats, labels, pred_cls)
        label_weights = None # TODO: is label_weights necessary?

        # v0.1: all + queue (orig + aug1 + queue, fg.bg), queue: only fg
        feats_ = feats_.squeeze(dim=1)
        if self.memory > 0:
            q_feats, q_labels = self._sample_negative(self.queue)   # exclude background class (background mean is useless)
            q_feats = q_feats.contiguous().detach()
            q_labels = q_labels.contiguous().view(-1,)
            feats_ = torch.cat([feats_, q_feats], dim=0)
            labels_ = torch.cat([labels_, q_labels], dim=0)

        # contrastive loss v0.1 all.fg.bg
        loss = supcontrast_clean_fg_bg(feats_, labels_, temper=0.07, min_samples=10)

        # loss = self.loss(feats_, labels_, label_weights, reduction=reduction) # TODO: need to validate

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



