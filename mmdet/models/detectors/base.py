# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod

import mmcv
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from mmcv.runner import BaseModule, auto_fp16

from mmdet.core.visualization import imshow_det_bboxes

# import wandb
import matplotlib.pyplot as plt
import os
import pdb
from collections import OrderedDict
from mmdet.models.losses.ai28.frame_loss import fpn_loss
from mmdet.models.trackers.sort_tracker import Sort, associate_detections_to_trackers
import cv2 # debug

# use_wandb = True # False True

def images_to_levels(target, num_levels):
    """Convert targets by image to targets by feature level.

    [target_img0, target_img1] -> [target_level0, target_level1, ...]
    """
    target = torch.stack(target, 0)
    level_targets = []
    start = 0
    for n in num_levels:
        end = start + n
        # level_targets.append(target[:, start:end].squeeze(0))
        level_targets.append(target[:, start:end])
        start = end
    return level_targets

class BaseDetector(BaseModule, metaclass=ABCMeta):
    """Base class for detectors."""

    def __init__(self, init_cfg=None):
        super(BaseDetector, self).__init__(init_cfg)
        self.fp16_enabled = False
        self.prev_data = None
        self.features = dict() # self.features are values of hook_layer_list
        self.wandb_data = dict()
        self.wandb_features=dict() # self.wandb_features has values of wandb.layer_list, accuracy, and values
        self.index = 0 # check current iteration for save log image
        self.loss_type_list = dict()

    @property
    def with_neck(self):
        """bool: whether the detector has a neck"""
        return hasattr(self, 'neck') and self.neck is not None

    # TODO: these properties need to be carefully handled
    # for both single stage & two stage detectors
    @property
    def with_shared_head(self):
        """bool: whether the detector has a shared head in the RoI Head"""
        return hasattr(self, 'roi_head') and self.roi_head.with_shared_head

    @property
    def with_bbox(self):
        """bool: whether the detector has a bbox head"""
        return ((hasattr(self, 'roi_head') and self.roi_head.with_bbox)
                or (hasattr(self, 'bbox_head') and self.bbox_head is not None))

    @property
    def with_mask(self):
        """bool: whether the detector has a mask head"""
        return ((hasattr(self, 'roi_head') and self.roi_head.with_mask)
                or (hasattr(self, 'mask_head') and self.mask_head is not None))

    @abstractmethod
    def extract_feat(self, imgs):
        """Extract features from images."""
        pass

    def extract_feats(self, imgs):
        """Extract features from multiple images.

        Args:
            imgs (list[torch.Tensor]): A list of images. The images are
                augmented from the same image but in different ways.

        Returns:
            list[torch.Tensor]: Features of different images
        """
        assert isinstance(imgs, list)
        return [self.extract_feat(img) for img in imgs]

    def forward_train(self, imgs, img_metas, **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys, see
                :class:`mmdet.datasets.pipelines.Collect`.
            kwargs (keyword arguments): Specific to concrete implementation.
        """
        # NOTE the batched image size information may be useful, e.g.
        # in DETR, this is needed for the construction of masks, which is
        # then used for the transformer_head.
        batch_input_shape = tuple(imgs[0].size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape

    async def async_simple_test(self, img, img_metas, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def simple_test(self, img, img_metas, **kwargs):
        pass

    @abstractmethod
    def aug_test(self, imgs, img_metas, **kwargs):
        """Test function with test time augmentation."""
        pass

    async def aforward_test(self, *, img, img_metas, **kwargs):
        for var, name in [(img, 'img'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')

        num_augs = len(img)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(img)}) '
                             f'!= num of image metas ({len(img_metas)})')
        # TODO: remove the restriction of samples_per_gpu == 1 when prepared
        samples_per_gpu = img[0].size(0)
        assert samples_per_gpu == 1

        if num_augs == 1:
            return await self.async_simple_test(img[0], img_metas[0], **kwargs)
        else:
            raise NotImplementedError

    def forward_test(self, imgs, img_metas, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch.
        """
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(imgs)}) '
                             f'!= num of image meta ({len(img_metas)})')

        # NOTE the batched image size information may be useful, e.g.
        # in DETR, this is needed for the construction of masks, which is
        # then used for the transformer_head.
        for img, img_meta in zip(imgs, img_metas):
            batch_size = len(img_meta)
            for img_id in range(batch_size):
                img_meta[img_id]['batch_input_shape'] = tuple(img.size()[-2:])

        if num_augs == 1:
            # proposals (List[List[Tensor]]): the outer list indicates
            # test-time augs (multiscale, flip, etc.) and the inner list
            # indicates images in a batch.
            # The Tensor should have a shape Px4, where P is the number of
            # proposals.
            if 'proposals' in kwargs:
                kwargs['proposals'] = kwargs['proposals'][0]
            return self.simple_test(imgs[0], img_metas[0], **kwargs)
        else:
            assert imgs[0].size(0) == 1, 'aug test does not support ' \
                                         'inference with batch size ' \
                                         f'{imgs[0].size(0)}'
            # TODO: support test augmentation for predefined proposals
            assert 'proposals' not in kwargs
            return self.aug_test(imgs, img_metas, **kwargs)

    @auto_fp16(apply_to=('img', ))
    def forward(self, img, img_metas, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if torch.onnx.is_in_onnx_export():
            assert len(img_metas) == 1
            return self.onnx_export(img[0], img_metas[0])

        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)
        else:
            return self.forward_test(img, img_metas, **kwargs)

    def _parse_losses(self, losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor \
                which may be a weighted sum of all losses, log_vars contains \
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        # If the loss_vars has different length, GPUs will wait infinitely
        if dist.is_available() and dist.is_initialized():
            log_var_length = torch.tensor(len(log_vars), device=loss.device)
            dist.all_reduce(log_var_length)
            message = (f'rank {dist.get_rank()}' +
                       f' len(log_vars): {len(log_vars)}' + ' keys: ' +
                       ','.join(log_vars.keys()))
            assert log_var_length == len(log_vars) * dist.get_world_size(), \
                'loss log variables are different across GPUs!\n' + message

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def save_the_result_img(self, data):
        cls_scores_all = self.features['rpn_head.rpn_cls']  # {list:5}  (3, 3, H', W')

        # Get gt_scores
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores_all]
        anchor_list, valid_flag_list = self.rpn_head.get_anchors(
            featmap_sizes, data['img_metas'])
        label_channels = 1
        cls_reg_targets = self.rpn_head.get_targets(
            anchor_list,
            valid_flag_list,
            data['gt_bboxes'],
            data['img_metas'],
            # gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=data['gt_labels'],
            label_channels=label_channels)
        del anchor_list, valid_flag_list
        # edited by dnwn24
        labels_flatten, labels_flatten_weight = cls_reg_targets[0], cls_reg_targets[1] # {list:5}  (num_types, H' * W' * num_priors)
        labels_flatten = cls_reg_targets[0]

        num_priors = 3
        num_type = data['img'].size()[0]
        num_lev = 5

        # print(labels_flatten)
        labels_all = []
        for i in range(5):
            label = labels_flatten[i] # (num_types, H' * W' * num_priors)
            label = label.reshape(num_type, featmap_sizes[i][0], featmap_sizes[i][1], -1).contiguous()  # (num_types, H', W', num_priors)
            # edited by dnwn24
            label_weight = labels_flatten_weight[i]
            label_weight = label_weight.reshape(num_type, featmap_sizes[i][0], featmap_sizes[i][1], -1).contiguous()
            label = torch.ones_like(label) - label
            label = label.type(torch.cuda.FloatTensor)
            label = torch.ones_like(label) - label
            label *= label_weight
            label = label.permute(0, 3, 1, 2).contiguous()  # (num_types, num_priors, H', W')
            labels_all.append(label)
        del featmap_sizes, cls_reg_targets, labels_flatten, label
        # labels_all : {list:5}  (num_types, num_priors, H', W')

        # Let's visualize
        H, W = int(cls_scores_all[0].size()[2]), int(cls_scores_all[0].size()[3])
        for i in range(1, 5):
            cls_scores_all[i] = F.interpolate(cls_scores_all[i], size=(H, W), mode='nearest')
            labels_all[i] = F.interpolate(labels_all[i].type(torch.cuda.FloatTensor), size=(H, W), mode='nearest') # labels_all : (num_types, num_priors, H, W)
        cls_scores_all = torch.stack([cls_scores for cls_scores in cls_scores_all], dim=0)  # (num_lev, num_type, num_priors, H, W)
        cls_scores_all = cls_scores_all.permute(2, 1, 0, 3, 4).contiguous()   # (num_priors, num_type, num_lev, H, W)
        labels_all = torch.stack([labels for labels in labels_all], dim=0)                  # (num_lev, num_type, num_priors, H, W)
        labels_all = labels_all.permute(2, 1, 0, 3, 4).contiguous()   # (num_priors, num_type, num_lev, H, W)

        num_priors = 3
        num_type = data['img'].size()[0]
        num_lev = 5
        plt.figure(figsize=(5 * (num_type + 1), 4 * num_priors))
        for p in range(num_priors):         # [0, 1, 2]
            labels = labels_all[p]          # (num_type, num_lev, H, W)
            # if torch.all(torch.all(torch.all(torch.eq(labels[0], labels[1]), dim=0), dim=0), dim=0):  # if equal
            #     print('Right!')
            # if torch.all(torch.all(torch.all(torch.eq(labels[0], labels[1]), dim=1), dim=0), dim=0):  # if equal
            #     print('Right!')

            label = labels[0]  # (num_lev, H, W)
            # label = torch.clamp(label.sum(axis=0) / num_lev, min=0)  # (H,W), range=[0,1]
            label = label.sum(axis=0) / num_lev  # (H,W), range=[0,1]
            label = label.to('cpu').detach().numpy()
            #
            label_min, label_max = np.min(label), np.max(label)
            if not label_max==label_min:
                label = (label-label_min)/(label_max-label_min)
            #
            label = (label * 255).astype(np.uint8)
            plt.subplot(num_priors, num_type + 1, p * (num_type + 1) + 1)
            plt.imshow(label, interpolation='nearest')
            plt.axis("off")

            cls_scores = cls_scores_all[p]  # (num_type, num_lev, H, W)
            for t in range(num_type):        # [clean, aug1, aug2]
                cls_score = cls_scores[t]    # (num_lev, H, W)
                cls_score = cls_score.sum(axis=0) / num_lev  # (H,W), range=[0,1]
                cls_score = cls_score.to('cpu').detach().numpy()
                # normalize
                cls_score_min, cls_score_max = np.min(cls_score), np.max(cls_score)
                if not cls_score_min == cls_score_max:
                    cls_score = (cls_score - cls_score_min)/(cls_score_max-cls_score_min)
                cls_score = (cls_score*255).astype(np.uint8)

                plt.subplot(num_priors, num_type+1, p * (num_type+1) + t + 2)
                plt.imshow(cls_score, interpolation='nearest')
                plt.axis("off")

        return plt

    def save_the_fpn_img(self):
        fpn_features = {k: v[0].detach().cpu() for k, v in self.features.items() if 'neck.fpn' in k}
        fpn_level = len(fpn_features)
        fpn_sizes = {k: v.size()[:] for k, v in fpn_features.items()}
        B, C, H, W = fpn_sizes[list(fpn_sizes.keys())[0]]
        plt.figure(figsize=(fpn_level, B))
        i = 1
        for key, feats in fpn_features.items():
            # feats: [B, C, H, W], single level features
            if B > 1:
                with torch.no_grad():
                    loss = F.mse_loss(feats[1], feats[0], reduction='mean')
                    self.wandb_features[key + ".p_aug1.mse_loss"] = loss
                    loss = F.mse_loss(feats[2], feats[0], reduction='mean')
                    self.wandb_features[key + ".p_aug2.mse_loss"] = loss
            # plt show
            feats_mean = feats.mean(dim=1, keepdim=True)
            feats_mean_inp = F.interpolate(feats_mean, size=(H, W), mode='nearest')
            for feat_mean_inp in feats_mean_inp:
                feat_mean_inp = torch.squeeze(feat_mean_inp)
                feat_mean_inp = feat_mean_inp.detach().cpu().numpy()
                plt.subplot(fpn_level, B, i)
                plt.imshow(feat_mean_inp, interpolation='bilinear')
                plt.axis("off")
                i += 1
        # plt.savefig("/ws/data/debug_test/test.jpg")

        pdb.set_trace()
        return plt

    def process_results(self, result, show_score_thr=0.3):
        ## get results
        import numpy as np
        if isinstance(result, tuple):
            bbox_result, segm_result = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]  # ms rcnn
        else:
            bbox_result, segm_result = result, None
        bboxes = np.vstack(bbox_result)

        labels = [np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(bbox_result)]
        labels = np.concatenate(labels)
        # draw segmentation masks
        segms = None
        if segm_result is not None and len(labels) > 0:  # non empty
            segms = mmcv.concat_list(segm_result)
            if isinstance(segms[0], torch.Tensor):
                segms = torch.stack(segms, dim=0).detach().cpu().numpy()
            else:
                segms = np.stack(segms, axis=0)

        if show_score_thr > 0:
            assert bboxes.shape[1] == 5
            scores = bboxes[:, -1]
            inds = scores > show_score_thr
            bboxes = bboxes[inds, :]
            labels = labels[inds]
            if segms is not None:
                segms = segms[inds, ...]
        mask = segms.astype(np.int32)

        processed_result = {"bboxes": bboxes, "mask": mask, "labels": labels}

        return processed_result


    def train_step(self, data, optimizer):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``, \
                ``num_samples``.

                - ``loss`` is a tensor for back propagation, which can be a
                  weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the
                  logger.
                - ``num_samples`` indicates the batch size (when the model is
                  DDP, it means the batch size on each GPU), which is used for
                  averaging the logs.
        """
        self.features.clear()
        self.wandb_features.clear()

        # pdb.set_trace()
        # with torch.no_grad():
        #     data2 = dict()
        #     data2['img'] = [data['img']]
        #     data2['img_metas'] = [data['img_metas']]
        #     result = self(return_loss=False, rescale=True, **data2)

        # pdb.set_trace()

        if self.prev_data != None:
            # concatenate the current data and previous data.
            consecutive_data = dict()
            data_test = dict()
            prev_data_test = dict()
            for key in data.keys():
                if key == 'img':
                    consecutive_data['img'] = torch.cat((data['img'], self.prev_data['img']), dim=0)
                    data_test['img'] = [data['img']]
                    prev_data_test['img'] = [self.prev_data['img']]
                elif key == 'img_metas':
                    data_test[key] = [data[key]]
                    prev_data_test[key] = [self.prev_data[key]]
                    consecutive_data[key] = data[key] + self.prev_data[key]
                else:
                    consecutive_data[key] = data[key] + self.prev_data[key]


            losses = self(**consecutive_data)

            with torch.no_grad():
                # get results
                pres_data_result = self(return_loss=False, rescale=True, **data_test)
                prev_data_result = self(return_loss=False, rescale=True, **prev_data_test)
                # process_results
                pres = self.process_results(pres_data_result[0])
                prev = self.process_results(prev_data_result[0])

                # pres_mask = pres['mask']
                # pres_bbox = pres['bboxes']

                # for key, feature in self.features.items():
                #     pres_mask_feature, pres_bbox_resized = self.mask_feature(feature[1], pres)
                #     prev_mask_feature, prev_bbox_resized = self.mask_feature(feature[2], prev)


                # for i, mask in enumerate(pres_mask):
                #     # draw bbox
                #     left_top = int(pres_bbox[i, 0]), int(pres_bbox[i, 1])
                #     right_bottom = int(pres_bbox[i, 2]), int(pres_bbox[i, 3])
                #     mask = mask * 255
                #     cv2.rectangle(mask, left_top, right_bottom, color=255, thickness=3)
                #     cv2.imwrite("/ws/data/cityscapes/mask/debug/pres/{}_pres.png".format(i), mask)

                pres_prev_matched, unmatched_pres_dets, unmatched_prev_dets = \
                    associate_detections_to_trackers(pres['bboxes'], prev['bboxes'], iou_threshold=0.3)

                if np.shape(pres_prev_matched)[0] != 0:
                    pres_matched_mask = pres["mask"][pres_prev_matched[:, 0]]
                    pres_matched_bbox = pres["bboxes"][pres_prev_matched[:, 0]]
                    # pres_matched_feat = pres["x1_npy"] * pres_matched_mask

                    prev_matched_mask = prev["mask"][pres_prev_matched[:, 1]]
                    prev_matched_bbox = prev["bboxes"][pres_prev_matched[:, 1]]
                    # prev_matched_feat = prev["x1_npy"] * prev_matched_mask

                    pres["mask"] = pres["mask"][pres_prev_matched[:, 0]]
                    pres["bboxes"] = pres["bboxes"][pres_prev_matched[:, 0]]
                    pres["labels"] = pres["labels"][pres_prev_matched[:, 0]]

                    prev["mask"] = prev["mask"][pres_prev_matched[:, 1]]
                    prev["bboxes"] = prev["bboxes"][pres_prev_matched[:, 1]]
                    prev["labels"] = prev["labels"][pres_prev_matched[:, 1]]

                    for key, feature in self.features.items():
                        pres_mask_feature, pres_bbox_resized = self.mask_feature(feature[1], pres)
                        prev_mask_feature, prev_bbox_resized = self.mask_feature(feature[2], prev)
                        prev_mask_feature_resized = self.interpolate_features(prev_mask_feature, pres_bbox_resized)
                        a = prev_mask_feature_resized

                    # for i, mask in enumerate(pres_matched_mask):
                    #     # draw bbox
                    #     left_top = int(pres_matched_bbox[i, 0]), int(pres_matched_bbox[i, 1])
                    #     right_bottom = int(pres_matched_bbox[i, 2]), int(pres_matched_bbox[i, 3])
                    #     mask = mask * 255
                    #     cv2.rectangle(mask, left_top, right_bottom, color=255, thickness=3)
                    #     cv2.imwrite("/ws/data/cityscapes/mask/debug/matched/{}_pres.png".format(i), mask)
                    #
                    #     left_top = int(prev_matched_bbox[i, 0]), int(prev_matched_bbox[i, 1])
                    #     right_bottom = int(prev_matched_bbox[i, 2]), int(prev_matched_bbox[i, 3])
                    #     prev_mask = prev_matched_mask[i] * 255
                    #     cv2.rectangle(prev_mask, left_top, right_bottom, color=255, thickness=3)
                    #     cv2.imwrite("/ws/data/cityscapes/mask/debug/matched/{}_prev.png".format(i), prev_mask)


                    # for key, feature in self.features.items():
                    #     _, f, h, w = feature[0].size()
                    #     F, H, W = np.shape(pres_matched_mask)
                    #     scale = (h / H, w / W)
                    #     for i in np.shape(pres_matched_mask):
                    #     # pres_matched_mask_resized = torch.nn.functional.interpolate(pres_matched_mask[0], size=(h, w), mode='nearest')
                    #     pres_matched_mask_resized = cv2.resize(pres_matched_mask[i], dsize=(h, w), interpolation=cv2.INTER_NEAREST)
                    #     pres_matched_bbox_resized = pres_matched_bbox[:, 0] * w / W, \
                    #                                 pres_matched_bbox[:, 1] * h / H, \
                    #                                 pres_matched_bbox[:, 2] * w / W, \
                    #                                 pres_matched_bbox[:, 3] * h / H
                    #     left_top = int(pres_matched_bbox_resized[i, 0]), int(pres_matched_bbox_resized[i, 1])
                    #     right_bottom = int(pres_matched_bbox_resized[i, 2]), int(pres_matched_bbox_resized[i, 3])
                    #     cv2.rectangle(pres_matched_mask_resized, left_top, right_bottom, color=255, thickness=3)
                    #     cv2.imwrite("/ws/data/cityscapes/mask/debug/prev_{}.png".format(key), pres_matched_mask_resized)


                    # domain generalization in the fpn layer
                    # if train_cfg.wandb.log.features_list = [], pass

                    if "loss" in self.neck.train_cfg:
                        # pdb.set_trace()
                        dict_kwargs = dict()
                        neck_train_cfg_loss = self.neck.train_cfg["loss"]
                        for key, value in neck_train_cfg_loss.items():
                            dict_kwargs[key] = value
                        for key, pred in self.features.items():
                            loss_, p_dist = fpn_loss(self.features[key], **dict_kwargs)
                            losses[f"fpn_loss.{key}"] = loss_

            loss, log_vars = self._parse_losses(losses)

            # wandb
            for layer_name in self.train_cfg.wandb.log.features_list:
                self.wandb_features[layer_name] = self.features[layer_name]
            if 'log_vars' in self.train_cfg.wandb.log.vars:
                for name, value in log_vars.items():
                    self.wandb_features[name] = np.mean(value)

            self.prev_data = data

        else:
            losses = self(**data)

            loss, log_vars = self._parse_losses(losses)

            # wandb
            # for layer_name in self.train_cfg.wandb.log.features_list:
            #     self.wandb_features[layer_name] = self.features[layer_name]
            if 'log_vars' in self.train_cfg.wandb.log.vars:
                for name, value in log_vars.items():
                    self.wandb_features[name] = np.mean(value)

            self.prev_data = data

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs

    def mask_feature(self, feature, results):
        """

        Args:
            feature: feature maps of feature pyramid network. (1, f, h, w)
            mask: mask of instance objects. (D, H, W)
            bbox: bbox of instance objects. (D, 4)

        Returns:

        """
        mask_features = []
        mask = results['mask']
        mask = torch.from_numpy(mask).float().to(feature.device)    # to implement F.interpolate
        mask = torch.unsqueeze(mask, dim=0)
        bbox = results['bboxes']
        _, f, h, w  = feature.size()
        _, D, H, W = np.shape(mask)
        scale = (h / H, w / W)
        # resize mask with scale
        # mask_resized = cv2.resize(mask, dsize=scale, interpolation=cv2.INTER_NEAREST)
        mask_resized = F.interpolate(mask, size=(h, w), mode='nearest')
        # debug
        # output, inverse_indices = torch.unique(mask_resized, sorted=True, return_inverse=True)
        bbox_resized = bbox * (w/ W, h / H, w/ W, h / H, 1)
        bbox_resized = bbox_resized.astype(np.int)
        for i in range(D):
            mask_feature = feature * mask_resized[:, i, :, :]
            # bbox slice interpolation is required later. Exception is required to treat under 1 pixel
            mask_feature_cropped = mask_feature[:, :, bbox_resized[i, 1]:bbox_resized[i, 3], bbox_resized[i, 0]:bbox_resized[i, 2]]
            mask_features.append(mask_feature_cropped)
            # bbox_feature = feature[:, :, bbox_resized[i, 1]:bbox_resized[i, 3], bbox_resized[i, 0]:bbox_resized[i, 2]]
            # mask_features.append(mask_feature)
            # bbox_features.append(bbox_feature)

        return mask_features, bbox_resized


    def interpolate_features(self, prev_features, pres_bboxes):
        """

        Args:
            pres_feature, prev_feature: {list: {Tensor: f, h, w}}
            pres_bbox, prev_bbox: {list: {tuple: (D, 5)}}

        Returns:

        """
        assert len(pres_bboxes) == len(prev_features)
        prev_features_resized = []
        for i in range(len(prev_features)):
            # pres_feature = pres_features[i]
            prev_feature = prev_features[i]
            pres_w = pres_bboxes[i, 2] - pres_bboxes[i, 0]
            pres_h = pres_bboxes[i, 3] - pres_bboxes[i, 1]
            prev_feature_resized = F.interpolate(prev_feature, size=(pres_h, pres_w), mode='nearest')
            prev_features_resized.append(prev_feature_resized)

        return prev_features_resized


    def val_step(self, data, optimizer=None):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs

    def show_result(self,
                    img,
                    result,
                    score_thr=0.3,
                    bbox_color=(72, 101, 241),
                    text_color=(72, 101, 241),
                    mask_color=None,
                    thickness=2,
                    font_size=13,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (Tensor or tuple): The results to draw over `img`
                bbox_result or (bbox_result, segm_result).
            score_thr (float, optional): Minimum score of bboxes to be shown.
                Default: 0.3.
            bbox_color (str or tuple(int) or :obj:`Color`):Color of bbox lines.
               The tuple of color should be in BGR order. Default: 'green'
            text_color (str or tuple(int) or :obj:`Color`):Color of texts.
               The tuple of color should be in BGR order. Default: 'green'
            mask_color (None or str or tuple(int) or :obj:`Color`):
               Color of masks. The tuple of color should be in BGR order.
               Default: None
            thickness (int): Thickness of lines. Default: 2
            font_size (int): Font size of texts. Default: 13
            win_name (str): The window name. Default: ''
            wait_time (float): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            img (Tensor): Only if not `show` or `out_file`
        """
        img = mmcv.imread(img)
        img = img.copy()
        if isinstance(result, tuple):
            bbox_result, segm_result = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]  # ms rcnn
        else:
            bbox_result, segm_result = result, None
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        # draw segmentation masks
        segms = None
        if segm_result is not None and len(labels) > 0:  # non empty
            segms = mmcv.concat_list(segm_result)
            if isinstance(segms[0], torch.Tensor):
                segms = torch.stack(segms, dim=0).detach().cpu().numpy()
            else:
                segms = np.stack(segms, axis=0)
        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False
        # draw bounding boxes
        img = imshow_det_bboxes(
            img,
            bboxes,
            labels,
            segms,
            class_names=self.CLASSES,
            score_thr=score_thr,
            bbox_color=bbox_color,
            text_color=text_color,
            mask_color=mask_color,
            thickness=thickness,
            font_size=font_size,
            win_name=win_name,
            show=show,
            wait_time=wait_time,
            out_file=out_file)

        if not (show or out_file):
            return img

    def onnx_export(self, img, img_metas):
        raise NotImplementedError(f'{self.__class__.__name__} does '
                                  f'not support ONNX EXPORT')