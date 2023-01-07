# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F

from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from ..builder import HEADS, build_head, build_roi_extractor
from .base_roi_head import BaseRoIHead
from .test_mixins import BBoxTestMixin, MaskTestMixin
from mmdet.models.losses.ai28.contrastive_loss import analyze_representations_2input_sample, analyze_representations_2input

import matplotlib.pyplot as plt
from thirdparty.dscv.utils.detection_utils import pixel2inch, visualize_bbox_xy
from thirdparty.dscv.utils.image_utils import denormalize, tensor_img_type_to
import numpy as np
import os


@HEADS.register_module()
class StandardRoIHead(BaseRoIHead, BBoxTestMixin, MaskTestMixin):
    """Simplest base roi head including one bbox head and one mask head."""

    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        self.bbox_assigner = None
        self.bbox_sampler = None
        if self.train_cfg:
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)
            self.bbox_sampler = build_sampler(
                self.train_cfg.sampler, context=self)

    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        """Initialize ``bbox_head``"""
        self.bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor)
        self.bbox_head = build_head(bbox_head)

    def init_mask_head(self, mask_roi_extractor, mask_head):
        """Initialize ``mask_head``"""
        if mask_roi_extractor is not None:
            self.mask_roi_extractor = build_roi_extractor(mask_roi_extractor)
            self.share_roi_extractor = False
        else:
            self.share_roi_extractor = True
            self.mask_roi_extractor = self.bbox_roi_extractor
        self.mask_head = build_head(mask_head)

    def forward_dummy(self, x, proposals):
        """Dummy forward function."""
        # bbox head
        outs = ()
        rois = bbox2roi([proposals])
        if self.with_bbox:
            bbox_results = self._bbox_forward(x, rois)
            outs = outs + (bbox_results['cls_score'],
                           bbox_results['bbox_pred'])
        # mask head
        if self.with_mask:
            mask_rois = rois[:100]
            mask_results = self._mask_forward(x, mask_rois)
            outs = outs + (mask_results['mask_pred'], )
        return outs

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []

            if not 'num_views' in kwargs:
                for i in range(num_imgs):
                    assign_result = self.bbox_assigner.assign(
                        proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                        gt_labels[i])
                    sampling_result = self.bbox_sampler.sample(
                        assign_result,
                        proposal_list[i],
                        gt_bboxes[i],
                        gt_labels[i],
                        feats=[lvl_feat[i][None] for lvl_feat in x])
                    sampling_results.append(sampling_result)
            else:
                sampling_results_tmp = []
                for i in range(int(num_imgs / kwargs['num_views'])):
                    assign_result = self.bbox_assigner.assign(
                        proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                        gt_labels[i])
                    sampling_result = self.bbox_sampler.sample(
                        assign_result,
                        proposal_list[i],
                        gt_bboxes[i],
                        gt_labels[i],
                        feats=[lvl_feat[i][None] for lvl_feat in x])
                    sampling_results_tmp.append(sampling_result)
                for j in range(kwargs['num_views']):
                    sampling_results.extend(sampling_results_tmp)
                _ = kwargs.pop('num_views')

        losses = dict()

        if self.train_cfg['dropout'] and self.with_bbox:
            bbox_results = self._bbox_forward_train_dropout(x, sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    img_metas, **kwargs)
            losses.update(bbox_results['loss_bbox'])
            if 'loss_feat' in bbox_results:  # DEV[CODE=102]: Contrastive loss with GenAutoAugment
                losses.update({'loss_feat': bbox_results['loss_feat']})


        # bbox head forward and loss
        elif self.with_bbox:
            bbox_results = self._bbox_forward_train(x, sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    img_metas, **kwargs)
            losses.update(bbox_results['loss_bbox'])
            if 'loss_feat' in bbox_results: # DEV[CODE=102]: Contrastive loss with GenAutoAugment
                losses.update({'loss_feat': bbox_results['loss_feat']})

        # mask head forward and loss
        if self.with_mask:
            mask_results = self._mask_forward_train(x, sampling_results,
                                                    bbox_results['bbox_feats'],
                                                    gt_masks, img_metas)
            losses.update(mask_results['loss_mask'])

        return losses


    def forward_analysis(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            # edited by dnwn24
            # assert divmod(num_imgs, 3)[1] == 0
            # same proposal list -> different sampling results
            # if divmod(num_imgs, 3)[1] !=0 :
            #     for i in range(num_imgs):
            #         assign_result = self.bbox_assigner.assign(
            #             proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
            #             gt_labels[i])
            #         sampling_result = self.bbox_sampler.sample(
            #             assign_result,
            #             proposal_list[i],
            #             gt_bboxes[i],
            #             gt_labels[i],
            #             feats=[lvl_feat[i][None] for lvl_feat in x])
            #         sampling_results.append(sampling_result)
            assign_result = self.bbox_assigner.assign(
                proposal_list[0], gt_bboxes[0], gt_bboxes_ignore[0],
                gt_labels[0])
            sampling_result = self.bbox_sampler.sample(
                assign_result,
                proposal_list[0],
                gt_bboxes[0],
                gt_labels[0],
                feats=[lvl_feat[0][None] for lvl_feat in x])
            # duplicate sampling results
            for i in range(num_imgs):
                sampling_results.append(sampling_result)

        features = self._bbox_forward_feature_analysis(x, sampling_results, gt_bboxes, gt_labels, img_metas, **kwargs)


        return features


    # ANALYSIS[CODE=002]: analysis loss region
    def _bbox_forward_analysis_loss_region(self, x, rois, bbox_targets, gt_bboxes, img_metas, save_dir):
        labels, label_weights, bbox_targets, bbox_weights = bbox_targets
        img = img_metas[0]['img']
        img_height, img_width = img.shape[-2:]

        fig, axes = plt.subplots(4, 3, figsize=(pixel2inch(img_width)/2*3, pixel2inch(img_height)/2*4))
        for i in range(3):
            mean, std = img_metas[i]['img_norm_cfg']['mean'], img_metas[i]['img_norm_cfg']['std']
            _img = denormalize(img[i], mean, std)
            _img = tensor_img_type_to(_img, np.ndarray, dtype=np.uint8)
            axes[0, i].imshow(_img); axes[1, i].imshow(_img); axes[2, i].imshow(_img); axes[3, i].imshow(_img)
        for row in range(3):
            for i in range(3):
                for j in range(64):
                    visualize_bbox_xy(rois[row * 64 + j, 1:], fig=fig, ax=axes[row+1, i], color_idx=labels[row * 64 + j], num_colors=9)

        # compute iou
        overlaps = self.bbox_assigner.iou_calculator(gt_bboxes[0], rois[:int(len(rois)/3), 1:])

        # for each anchor, which gt best overlaps with it
        # for each anchor, the max iou of all gts
        max_overlaps, argmax_overlaps = overlaps.max(dim=0)

        def bincount(data, min_val=None, max_val=None, num_bins=10):
            if min_val == None:
                min_val = float(data.min())
            if max_val == None:
                max_val = float(data.max())
            return torch.histc(data, bins=num_bins, min=min_val, max=max_val)
        hist_overlaps = bincount(max_overlaps, min_val=0.0, max_val=1.0, num_bins=10)
        title ='  '
        for i in range(len(hist_overlaps)):
            title += f"{int(hist_overlaps[i]):3d} "
        fig.suptitle(title)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fig.savefig(f'{save_dir}/{img_metas[0]["ori_filename"].split("/")[-1].split(".png")[0]}_{self.num_save_img:06d}.png')
        self.num_save_img += 1

        return

    # ANALYSIS[CODE=001]: analysis background
    def _bbox_forward_analysis_background(self, x, rois, img_metas):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats = self.bbox_roi_extractor.forward_analysis_background(
            x[:self.bbox_roi_extractor.num_inputs], rois, img_metas)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred = self.bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results


    def _bbox_forward_feature_analysis(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas, gt_instance_inds=None):
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)

        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        # DEV[CODE=100]: Analyze feature with sample-to-sample method
        global feature_cls_feats
        feature_cls_feats = self.bbox_head.cls_feats

        # To do
        from mmdet.models.losses.ai28.contrastive_loss import analyze_representations_2input
        num_imgs = len(sampling_results)
        feature_analysis = dict()
        if num_imgs == 2:
            feature_clean, feature_corr = torch.chunk(feature_cls_feats, num_imgs)
            label = bbox_targets[0]
            label, _  = torch.chunk(label, num_imgs)
            feature_analysis1 = analyze_representations_2input(feature_clean, feature_clean, label, )
            feature_analysis2 = analyze_representations_2input(feature_clean, feature_corr, label, )
            feature_analysis3 = analyze_representations_2input(feature_corr, feature_corr, label, )

            for key, value in feature_analysis1.items():
                feature_analysis[f"clean_clean_{key}"] = value
                feature_analysis[f"clean_corr_{key}"] = feature_analysis2[key]
                feature_analysis[f"corr_corr_{key}"] = feature_analysis3[key]

        elif num_imgs == 3:
            feature_clean, feature_corr, feature_aug = torch.chunk(feature_cls_feats, num_imgs)
            label = bbox_targets[0]
            label, _, _ = torch.chunk(label, num_imgs)
            feature_analysis1 = analyze_representations_2input(feature_clean, feature_clean, label, )
            feature_analysis2 = analyze_representations_2input(feature_clean, feature_corr, label, )
            feature_analysis3 = analyze_representations_2input(feature_clean, feature_aug, label, )
            feature_analysis4 = analyze_representations_2input(feature_corr, feature_corr, label, )
            feature_analysis5 = analyze_representations_2input(feature_corr, feature_aug, label, )
            feature_analysis6 = analyze_representations_2input(feature_aug, feature_aug, label, )


            for key, value in feature_analysis1.items():
                feature_analysis[f"clean_clean_{key}"] = feature_analysis1[key]
                feature_analysis[f"clean_corr_{key}"] = feature_analysis2[key]
                feature_analysis[f"clean_aug_{key}"] = feature_analysis3[key]
                feature_analysis[f"corr_corr_{key}"] = feature_analysis4[key]
                feature_analysis[f"corr_aug_{key}"] = feature_analysis5[key]
                feature_analysis[f"aug_aug_{key}"] = feature_analysis6[key]

        self.bbox_targets = bbox_targets

        return feature_analysis


    def _bbox_forward(self, x, rois):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred = self.bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats, cls_feats=self.bbox_head.cls_feats)
        return bbox_results


    def _bbox_forward_train_dropout(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas, gt_instance_inds=None, log_loss_region=None):
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)

        global count_fg
        bbox_target_label = bbox_targets[0]

        count_total = int(bbox_target_label.size(0) / 3)
        count_bg = int((bbox_target_label == bbox_target_label.max()).float().sum() / 3)
        count_fg = count_total - count_bg

        self.bbox_targets = bbox_targets

        bbox_results = self._bbox_forward(x, rois)


        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], rois,
                                        *bbox_targets)

        bbox_results.update(loss_bbox=loss_bbox)

        ### ANALYSIS CODE from here ###
        if hasattr(self, 'analysis_list'):
            type_list = [analysis['type'] for analysis in self.analysis_list]
            if 'log_loss_region' in type_list:
                analysis_cfg = self.analysis_list[type_list.index('log_loss_region')]
                if img_metas[0]["ori_filename"] in analysis_cfg.filename_list:
                    self._bbox_forward_analysis_loss_region(x, rois, bbox_targets, gt_bboxes, img_metas,
                                                            analysis_cfg.save_dir)
        ### ANALYSIS CODE to here ###

        # DEV[CODE=102]: Contrastive loss with GenAutoAugment
        if self.loss_feat != None:
            # Ground-truth
            valid_gt_bboxes, valid_gt_instance_inds, valid_gt_rois = \
                self.loss_feat.filter_and_collect(gt_bboxes, gt_instance_inds)

            gt_bbox_results = []
            for i in range(len(valid_gt_rois)):
                gt_bbox_results.append(self._bbox_forward(x, valid_gt_rois[i]))

            loss_feat = self.loss_feat(gt_bbox_results, gt_labels, valid_gt_instance_inds)

            bbox_results.update(loss_feat=loss_feat)

        return bbox_results


    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas, gt_instance_inds=None, **kwargs):
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)

        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)

        self.bbox_targets = bbox_targets
        self.bbox_head.num_samples = self.train_cfg.sampler.num
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], rois,
                                        *bbox_targets)

        bbox_results.update(loss_bbox=loss_bbox)

        ### ANALYSIS CODE from here ###
        if hasattr(self, 'analysis_list'):
            type_list = [analysis['type'] for analysis in self.analysis_list]
            if 'log_loss_region' in type_list:
                analysis_cfg = self.analysis_list[type_list.index('log_loss_region')]
                if img_metas[0]["ori_filename"] in analysis_cfg.filename_list:
                    self._bbox_forward_analysis_loss_region(x, rois, bbox_targets, gt_bboxes, img_metas,
                                                            analysis_cfg.save_dir)
        ### ANALYSIS CODE to here ###

        # DEV[CODE=102]: Contrastive loss with GenAutoAugment
        if self.loss_feat != None:
            # Ground-truth
            valid_gt_bboxes, valid_gt_instance_inds, valid_gt_rois = \
                self.loss_feat.filter_and_collect(gt_bboxes, gt_instance_inds)

            gt_bbox_results = []
            for i in range(len(valid_gt_rois)):
                gt_bbox_results.append(self._bbox_forward(x, valid_gt_rois[i]))

            loss_feat = self.loss_feat(gt_bbox_results, gt_labels, valid_gt_instance_inds)

            bbox_results.update(loss_feat=loss_feat)

        return bbox_results

    def _mask_forward_train(self, x, sampling_results, bbox_feats, gt_masks,
                            img_metas):
        """Run forward function and calculate loss for mask head in
        training."""
        if not self.share_roi_extractor:
            pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
            mask_results = self._mask_forward(x, pos_rois)
        else:
            pos_inds = []
            device = bbox_feats.device
            for res in sampling_results:
                pos_inds.append(
                    torch.ones(
                        res.pos_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
                pos_inds.append(
                    torch.zeros(
                        res.neg_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
            pos_inds = torch.cat(pos_inds)

            mask_results = self._mask_forward(
                x, pos_inds=pos_inds, bbox_feats=bbox_feats)

        mask_targets = self.mask_head.get_targets(sampling_results, gt_masks,
                                                  self.train_cfg)
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        loss_mask = self.mask_head.loss(mask_results['mask_pred'],
                                        mask_targets, pos_labels)

        mask_results.update(loss_mask=loss_mask, mask_targets=mask_targets)
        return mask_results

    def _mask_forward(self, x, rois=None, pos_inds=None, bbox_feats=None):
        """Mask head forward function used in both training and testing."""
        assert ((rois is not None) ^
                (pos_inds is not None and bbox_feats is not None))
        if rois is not None:
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
        else:
            assert bbox_feats is not None
            mask_feats = bbox_feats[pos_inds]

        mask_pred = self.mask_head(mask_feats)
        mask_results = dict(mask_pred=mask_pred, mask_feats=mask_feats)
        return mask_results

    async def async_simple_test(self,
                                x,
                                proposal_list,
                                img_metas,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = await self.async_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)
        if not self.with_mask:
            return bbox_results
        else:
            segm_results = await self.async_test_mask(
                x,
                img_metas,
                det_bboxes,
                det_labels,
                rescale=rescale,
                mask_test_cfg=self.test_cfg.get('mask'))
            return bbox_results, segm_results

    # ANALYSIS[CODE=001]: analysis background
    def simple_test_analysis_background(self,
                                        x,
                                        proposal_list,
                                        img_metas,
                                        proposals=None,
                                        rescale=False):
        assert self.with_bbox, 'Bbox head must be implemented.'

        ''' self.simple_test_bboxes starts from here '''
        proposals = proposal_list
        rcnn_test_cfg = self.test_cfg

        rois = bbox2roi(proposals)

        if not rois.shape[0] == 0:
            only_gt = True
            if only_gt:
                gt_bboxes = torch.tensor(img_metas[0]['annotations'][0]['bboxes']).to(rois.get_device())
                gt_labels = torch.tensor(img_metas[0]['annotations'][0]['labels']).to(rois.get_device())
                batch_inds = gt_bboxes.new_full((gt_bboxes.size(0), 1), 0)
                rois = torch.cat([batch_inds, gt_bboxes], dim=1)

            bbox_results = self._bbox_forward_analysis_background(x, rois, img_metas)
            img_shapes = tuple(meta['img_shape'] for meta in img_metas)
            scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

            # split batch bbox prediction back to each image
            cls_score = bbox_results['cls_score']
            bbox_pred = bbox_results['bbox_pred']

            if only_gt:
                num_proposals_per_img = len(rois)
            else:
                num_proposals_per_img = tuple(len(p) for p in proposals)
            rois = rois.split(num_proposals_per_img, 0)
            cls_score = cls_score.split(num_proposals_per_img, 0)

            # some detector with_reg is False, bbox_pred will be None
            if bbox_pred is not None:
                # TODO move this to a sabl_roi_head
                # the bbox prediction of some detectors like SABL is not Tensor
                if isinstance(bbox_pred, torch.Tensor):
                    bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
                else:
                    bbox_pred = self.bbox_head.bbox_pred_split(
                        bbox_pred, num_proposals_per_img)
            else:
                bbox_pred = (None,) * len(proposals)

            # apply bbox post-processing to each image individually
            for i in range(len(proposals)):
                if not rois[i].shape[0] == 0:
                    self.bbox_head.analysis_regions(rois[i], cls_score[i], bbox_pred[i], img_shapes[i],
                                                    scale_factors[i], rescale=rescale, cfg=rcnn_test_cfg,
                                                    img_metas=img_metas)

        return 0


    # ANALYSIS[CODE=001]: analysis background
    def simple_test_analyze_feature(self,
                                        x,
                                        proposal_list,
                                        img_metas,
                                        proposals=None,
                                        rescale=False,
                                        max_samples=512):
        assert self.with_bbox, 'Bbox head must be implemented.'

        import numpy as np
        from mmdet.utils.visualize import plot_matrix
        import copy

        ''' self.simple_test_bboxes starts from here '''
        proposals = proposal_list
        rcnn_test_cfg = self.test_cfg

        rois = bbox2roi(proposals)

        x = list(x)
        # if img_metas[0]['proposal_type'] == 'perturb_cutout':
        #     for i in range(len(x)):
        #         mask = torch.rand_like(x[i][0, 0, :, :]) > 0.5
        #         mask = mask.float()
        #         mask = mask.repeat(1, 256, 1, 1)
        #         x_masked = x[i][0:1] * mask     # original image masked
        #         x[i] = torch.cat([x[i], x_masked], dim=0)

        if not rois.shape[0] == 0:
            only_gt = False
            if only_gt:
                gt_bboxes = torch.tensor(img_metas[0]['annotations'][0]['bboxes']).to(rois.get_device())
                gt_labels = torch.tensor(img_metas[0]['annotations'][0]['labels']).to(rois.get_device())
                batch_inds = gt_bboxes.new_full((gt_bboxes.size(0), 1), 0)
                rois = torch.cat([batch_inds, gt_bboxes], dim=1)

            bbox_results = self._bbox_forward_analysis_background(x, rois, img_metas)

            img_shapes = tuple(meta['img_shape'] for meta in img_metas)
            scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

            # split batch bbox prediction back to each image
            cls_score = bbox_results['cls_score']
            bbox_pred = bbox_results['bbox_pred']

            if only_gt:
                num_proposals_per_img = len(rois)
            else:
                num_proposals_per_img = tuple(len(p) for p in proposals)
            rois = rois.split(num_proposals_per_img, 0)
            cls_score = cls_score.split(num_proposals_per_img, 0)

            # global feature_cls_feats
            feature_cls_feats = self.bbox_head.cls_feats
            feature_cls_feats = feature_cls_feats.split(num_proposals_per_img, 0)

            # some detector with_reg is False, bbox_pred will be None
            if bbox_pred is not None:
                # TODO move this to a sabl_roi_head
                # the bbox prediction of some detectors like SABL is not Tensor
                if isinstance(bbox_pred, torch.Tensor):
                    bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
                else:
                    bbox_pred = self.bbox_head.bbox_pred_split(
                        bbox_pred, num_proposals_per_img)
            else:
                bbox_pred = (None,) * len(proposals)

            # max_samples = 30

            feature_class = True
            if feature_class:
                pass

            one_plot = True
            if one_plot:
                # 1D plot features
                from mmdet.utils.visualize import plot_bar
                for i in range(feature_cls_feats):
                    for j, value in enumerate(feature_cls_feats[i]):
                        plt = plot_bar(value.detach().cpu().numpy())
                        plt.savefig(f"{img_metas[0]['work_dir']}feature1d_{i}_{j}.png")

            feature_sample = True
            if feature_sample:

                features = analyze_representations_2input_sample(feature_cls_feats[0], feature_cls_feats[1])
                name = 'orig_corr_'
                for key, value in features.items():
                    np.savetxt(f"{img_metas[0]['work_dir']}{name}{key}.txt", value, fmt='%1.3f')
                    plt = plot_matrix(value, classes=proposals[0].size(0), txt=False)
                    plt.savefig(f"{img_metas[0]['work_dir']}{name}{key}.png")

                features = analyze_representations_2input_sample(feature_cls_feats[0], feature_cls_feats[0])
                name = 'orig_oig_'
                for key, value in features.items():
                    np.savetxt(f"{img_metas[0]['work_dir']}{name}{key}.txt", value, fmt='%1.3f')
                    plt = plot_matrix(value, classes=proposals[0].size(0), txt=False)
                    plt.savefig(f"{img_metas[0]['work_dir']}{name}{key}.png")

                features = analyze_representations_2input_sample(feature_cls_feats[1], feature_cls_feats[1])
                name = 'corr_corr_'
                for key, value in features.items():
                    np.savetxt(f"{img_metas[0]['work_dir']}{name}{key}.txt", value, fmt='%1.3f')
                    plt = plot_matrix(value, classes=proposals[0].size(0), txt=False)
                    plt.savefig(f"{img_metas[0]['work_dir']}{name}{key}.png")

            vis_image = True
            if vis_image:
                # apply bbox post-processing to each image individually
                for i in range(len(proposals)):
                    img_metas_i = copy.deepcopy(img_metas)
                    img_metas_i[0]['work_dir'] += str(i) + '_'

                    if not rois[i].shape[0] == 0:
                        self.bbox_head.analysis_regions(rois[i], cls_score[i], bbox_pred[i], img_shapes[0],
                                                        scale_factors[0], rescale=rescale, cfg=rcnn_test_cfg,
                                                        img_metas=img_metas_i, each_bbox=True, each_class=False)


        return 0


    # ANALYSIS[CODE=001]: analysis background
    def simple_test_analyze_feature_class(self,
                                        x,
                                        proposal_list,
                                        img_metas,
                                        proposals=None,
                                        rescale=False,
                                        max_samples=512):
        assert self.with_bbox, 'Bbox head must be implemented.'

        import numpy as np
        from mmdet.utils.visualize import plot_matrix
        import copy

        ''' self.simple_test_bboxes starts from here '''
        proposals = proposal_list
        rcnn_test_cfg = self.test_cfg

        rois = bbox2roi(proposals)

        x = list(x)
        # if img_metas[0]['proposal_type'] == 'perturb_cutout':
        #     for i in range(len(x)):
        #         mask = torch.rand_like(x[i][0, 0, :, :]) > 0.5
        #         mask = mask.float()
        #         mask = mask.repeat(1, 256, 1, 1)
        #         x_masked = x[i][0:1] * mask     # original image masked
        #         x[i] = torch.cat([x[i], x_masked], dim=0)

        if not rois.shape[0] == 0:
            only_gt = False
            if only_gt:
                gt_bboxes = torch.tensor(img_metas[0]['annotations'][0]['bboxes']).to(rois.get_device())
                gt_labels = torch.tensor(img_metas[0]['annotations'][0]['labels']).to(rois.get_device())
                batch_inds = gt_bboxes.new_full((gt_bboxes.size(0), 1), 0)
                rois = torch.cat([batch_inds, gt_bboxes], dim=1)

            bbox_results = self._bbox_forward_analysis_background(x, rois, img_metas)

            img_shapes = tuple(meta['img_shape'] for meta in img_metas)
            scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

            # split batch bbox prediction back to each image
            cls_score = bbox_results['cls_score']
            bbox_pred = bbox_results['bbox_pred']

            if only_gt:
                num_proposals_per_img = len(rois)
            else:
                num_proposals_per_img = tuple(len(p) for p in proposals)
            rois = rois.split(num_proposals_per_img, 0)
            cls_score = cls_score.split(num_proposals_per_img, 0)

            # global feature_cls_feats
            feature_cls_feats = self.bbox_head.cls_feats
            feature_cls_feats = feature_cls_feats.split(num_proposals_per_img, 0)

            # to do. how to get gt labels on the test mode???
            features = analyze_representations_2input(feature_cls_feats[0], feature_cls_feats[1], )


            # some detector with_reg is False, bbox_pred will be None
            if bbox_pred is not None:
                # TODO move this to a sabl_roi_head
                # the bbox prediction of some detectors like SABL is not Tensor
                if isinstance(bbox_pred, torch.Tensor):
                    bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
                else:
                    bbox_pred = self.bbox_head.bbox_pred_split(
                        bbox_pred, num_proposals_per_img)
            else:
                bbox_pred = (None,) * len(proposals)


        return 0


    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    proposals=None,
                    rescale=False):
        """Test without augmentation.

        Args:
            x (tuple[Tensor]): Features from upstream network. Each
                has shape (batch_size, c, h, w).
            proposal_list (list(Tensor)): Proposals from rpn head.
                Each has shape (num_proposals, 5), last dimension
                5 represent (x1, y1, x2, y2, score).
            img_metas (list[dict]): Meta information of images.
            rescale (bool): Whether to rescale the results to
                the original image. Default: True.

        Returns:
            list[list[np.ndarray]] or list[tuple]: When no mask branch,
            it is bbox results of each image and classes with type
            `list[list[np.ndarray]]`. The outer list
            corresponds to each image. The inner list
            corresponds to each class. When the model has mask branch,
            it contains bbox results and mask results.
            The outer list corresponds to each image, and first element
            of tuple is bbox results, second element is mask results.
        """
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)

        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
            return list(zip(bbox_results, segm_results))

    def aug_test(self, x, proposal_list, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        det_bboxes, det_labels = self.aug_test_bboxes(x, img_metas,
                                                      proposal_list,
                                                      self.test_cfg)
        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= det_bboxes.new_tensor(
                img_metas[0][0]['scale_factor'])
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        # det_bboxes always keep the original scale
        if self.with_mask:
            segm_results = self.aug_test_mask(x, img_metas, det_bboxes,
                                              det_labels)
            return [(bbox_results, segm_results)]
        else:
            return [bbox_results]

    def onnx_export(self, x, proposals, img_metas, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        det_bboxes, det_labels = self.bbox_onnx_export(
            x, img_metas, proposals, self.test_cfg, rescale=rescale)

        if not self.with_mask:
            return det_bboxes, det_labels
        else:
            segm_results = self.mask_onnx_export(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
            return det_bboxes, det_labels, segm_results

    def mask_onnx_export(self, x, img_metas, det_bboxes, det_labels, **kwargs):
        """Export mask branch to onnx which supports batch inference.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            img_metas (list[dict]): Image meta info.
            det_bboxes (Tensor): Bboxes and corresponding scores.
                has shape [N, num_bboxes, 5].
            det_labels (Tensor): class labels of
                shape [N, num_bboxes].

        Returns:
            Tensor: The segmentation results of shape [N, num_bboxes,
                image_height, image_width].
        """
        # image shapes of images in the batch

        if all(det_bbox.shape[0] == 0 for det_bbox in det_bboxes):
            raise RuntimeError('[ONNX Error] Can not record MaskHead '
                               'as it has not been executed this time')
        batch_size = det_bboxes.size(0)
        # if det_bboxes is rescaled to the original image size, we need to
        # rescale it back to the testing scale to obtain RoIs.
        det_bboxes = det_bboxes[..., :4]
        batch_index = torch.arange(
            det_bboxes.size(0), device=det_bboxes.device).float().view(
                -1, 1, 1).expand(det_bboxes.size(0), det_bboxes.size(1), 1)
        mask_rois = torch.cat([batch_index, det_bboxes], dim=-1)
        mask_rois = mask_rois.view(-1, 5)
        mask_results = self._mask_forward(x, mask_rois)
        mask_pred = mask_results['mask_pred']
        max_shape = img_metas[0]['img_shape_for_onnx']
        num_det = det_bboxes.shape[1]
        det_bboxes = det_bboxes.reshape(-1, 4)
        det_labels = det_labels.reshape(-1)
        segm_results = self.mask_head.onnx_export(mask_pred, det_bboxes,
                                                  det_labels, self.test_cfg,
                                                  max_shape)
        segm_results = segm_results.reshape(batch_size, num_det, max_shape[0],
                                            max_shape[1])
        return segm_results

    def bbox_onnx_export(self, x, img_metas, proposals, rcnn_test_cfg,
                         **kwargs):
        """Export bbox branch to onnx which supports batch inference.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            img_metas (list[dict]): Image meta info.
            proposals (Tensor): Region proposals with
                batch dimension, has shape [N, num_bboxes, 5].
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.

        Returns:
            tuple[Tensor, Tensor]: bboxes of shape [N, num_bboxes, 5]
                and class labels of shape [N, num_bboxes].
        """
        # get origin input shape to support onnx dynamic input shape
        assert len(
            img_metas
        ) == 1, 'Only support one input image while in exporting to ONNX'
        img_shapes = img_metas[0]['img_shape_for_onnx']

        rois = proposals

        batch_index = torch.arange(
            rois.size(0), device=rois.device).float().view(-1, 1, 1).expand(
                rois.size(0), rois.size(1), 1)

        rois = torch.cat([batch_index, rois[..., :4]], dim=-1)
        batch_size = rois.shape[0]
        num_proposals_per_img = rois.shape[1]

        # Eliminate the batch dimension
        rois = rois.view(-1, 5)
        bbox_results = self._bbox_forward(x, rois)
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']

        # Recover the batch dimension
        rois = rois.reshape(batch_size, num_proposals_per_img, rois.size(-1))
        cls_score = cls_score.reshape(batch_size, num_proposals_per_img,
                                      cls_score.size(-1))

        bbox_pred = bbox_pred.reshape(batch_size, num_proposals_per_img,
                                      bbox_pred.size(-1))
        det_bboxes, det_labels = self.bbox_head.onnx_export(
            rois, cls_score, bbox_pred, img_shapes, cfg=rcnn_test_cfg)

        return det_bboxes, det_labels
