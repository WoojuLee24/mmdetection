# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import numpy as np
import torch

from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector
from mmdet.utils.visualize import visualize_score_distribution, visualize_score_density, visualize_image, get_file_name


@DETECTORS.register_module()
class TwoStageDetector(BaseDetector):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(TwoStageDetector, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)

        if neck is not None:
            self.neck = build_neck(neck)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = build_head(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            roi_head.pretrained = pretrained
            self.roi_head = build_head(roi_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    @property
    def with_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        if hasattr(self, 'grad_cam'):
            if self.grad_cam:
                x_backbone = self.backbone(img)
                if self.with_neck:
                    x_fpn = self.neck(x_backbone)
                return x_backbone, x_fpn

        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        outs = ()
        # backbone
        x = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 4).to(img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs, )
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x = self.extract_feat(img)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        ### ANALYSIS CODE from here ###
        if 'analysis_list' in self.train_cfg:
            type_list = [analysis['type'] for analysis in self.train_cfg.analysis_list]
            if 'analysis_num_pos_and_neg' in type_list:
                analysis_cfg = self.train_cfg.analysis_list[type_list.index('analysis_num_pos_and_neg')]
                if not hasattr(self.roi_head.bbox_sampler, 'analysis_list'):
                    self.roi_head.bbox_sampler.analysis_list = []
                self.roi_head.bbox_sampler.analysis_list.append(analysis_cfg)
            if 'log_loss_region' in type_list:
                analysis_cfg = self.train_cfg.analysis_list[type_list.index('log_loss_region')]
                img_metas[0]['img'] = img
                if not hasattr(self.roi_head, 'analysis_list'):
                    self.roi_head.analysis_list = []
                self.roi_head.analysis_list.append(analysis_cfg)
            if 'bbox_head_loss' in type_list:
                analysis_cfg = self.train_cfg.analysis_list[type_list.index('bbox_head_loss')]
                analysis_cfg.num_samples = self.roi_head.train_cfg.sampler.num
                if not hasattr(self.roi_head.bbox_head, 'analysis_list'):
                    self.roi_head.bbox_head.analysis_list = []
                self.roi_head.bbox_head.analysis_list.append(analysis_cfg)
        ### ANALYSIS CODE to here ###

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)

        ### ANALYSIS CODE from here ###
        if 'analysis_list' in self.train_cfg:
            type_list = [analysis['type'] for analysis in self.train_cfg.analysis_list]
            if 'loss_weight' in type_list:
                analysis_cfg = self.train_cfg.analysis_list[type_list.index('loss_weight')]
                # rpn_cls
                for key, value in self.rpn_head.loss_cls.outputs.items():
                    analysis_cfg.outputs[f"loss_weight({key.replace('loss_', '')}__per__rpn_cls)"] = float(sum(value) / sum(rpn_losses['loss_rpn_cls']))
                    self.rpn_head.loss_cls.outputs[key].clear()
                # rpn_reg
                for key, value in self.rpn_head.loss_bbox.outputs.items():
                    analysis_cfg.outputs[f"loss_weight({key.replace('loss_', '')}__per__rpn_bbox)"] = float(sum(value) / sum(rpn_losses['loss_rpn_bbox']))
                    self.rpn_head.loss_bbox.outputs[key].clear()
                # roi_cls
                for key, value in self.roi_head.bbox_head.loss_cls.outputs.items():
                    analysis_cfg.outputs[f"loss_weight({key.replace('loss_', '')}__per__roi_cls)"] = float(sum(value) / roi_losses['loss_cls'])
                    self.roi_head.bbox_head.loss_cls.outputs[key].clear()
                # roi_cls
                for key, value in self.roi_head.bbox_head.loss_bbox.outputs.items():
                    analysis_cfg.outputs[f"loss_weight({key.replace('loss_', '')}__per__roi_bbox)"] = float(sum(value) / roi_losses['loss_bbox'])
                    self.roi_head.bbox_head.loss_bbox.outputs[key].clear()
            if 'additional_loss_ratio' in type_list:
                analysis_cfg = self.train_cfg.analysis_list[type_list.index('additional_loss_ratio')]
                if not hasattr(analysis_cfg, 'outputs'):
                    analysis_cfg.outputs = dict()
                if 'rpn_cls' in analysis_cfg.log_list:
                    _outputs = self.rpn_head.loss_cls.outputs
                    analysis_cfg.outputs[f"additional_loss_ratio({self.rpn_head.loss_cls.wandb_name})"] = \
                        float(self.rpn_head.loss_cls.lambda_weight * sum(_outputs['loss_additional']) / sum(_outputs['loss_cls']))
                if 'roi_cls' in analysis_cfg.log_list:
                    _outputs = self.roi_head.bbox_head.loss_cls.outputs
                    analysis_cfg.outputs[f"additional_loss_ratio({self.roi_head.bbox_head.loss_cls.wandb_name})"] = \
                        float(self.rpn_head.loss_cls.lambda_weight * sum(_outputs['loss_additional']) / sum(_outputs['loss_cls']))
        ### ANALYSIS CODE to here ###

        return losses


    def forward_analysis(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        if 'img2' in kwargs:
            # img2 = kwargs.pop('img2')   # orig data
            img2 = kwargs['img2']   # orig data
            # img = torch.cat([img2, img, img], dim=0)
            img = torch.cat([img2, img], dim=0)
            gt_bboxes.append(gt_bboxes[0])
            # gt_bboxes.append(gt_bboxes[0])
            gt_labels.append(gt_labels[0])
            # gt_labels.append(gt_labels[0])
            img_metas.append(img_metas[0])
            # img_metas.append(img_metas[0])

        if 'img3' in kwargs:    # augmented data. img = ['img1', 'img2', 'img3'] = [original, corrupted, augmented]
            img3 = kwargs['img3']
            img = torch.cat([img, img3], dim=0)
            gt_bboxes.append(gt_bboxes[0])
            gt_labels.append(gt_labels[0])
            img_metas.append(img_metas[0])


        x = self.extract_feat(img)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            losses.update(rpn_losses)

            # proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)


        else:
            proposal_list = proposals

        if 'img2' in kwargs:
            # proposal list of original image
            proposal_list[1] = proposal_list[0]
            # proposal_list[2] = proposal_list[0]
            _ = kwargs.pop('img2')

        if 'img3' in kwargs:
            proposal_list[2] = proposal_list[0]
            _ = kwargs.pop('img3')

        # roi_losses, features = self.roi_head.forward_analysis(x, img_metas, proposal_list,
        #                                          gt_bboxes, gt_labels,
        #                                          gt_bboxes_ignore, gt_masks,
        #                                          **kwargs)
        features = self.roi_head.forward_analysis(x, img_metas, proposal_list,
                                                  gt_bboxes, gt_labels,
                                                  gt_bboxes_ignore, gt_masks,
                                                  **kwargs)

        return losses, features


    async def async_simple_test(self,
                                img,
                                img_meta,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = await self.rpn_head.async_simple_test_rpn(
                x, img_meta)
        else:
            proposal_list = proposals

        return await self.roi_head.async_simple_test(
            x, proposal_list, img_meta, rescale=rescale)

    def simple_test(self, img, img_metas, proposals=None, rescale=False, **kwargs):
        """Test without augmentation."""
        debug_cfg = kwargs['debug_cfg'] if 'debug_cfg' in kwargs else None

        assert self.with_bbox, 'Bbox head must be implemented.'
        if hasattr(self, 'grad_cam'):
            if self.grad_cam:
                x_backbone, x_fpn = self.extract_feat(img)
                if proposals is None:
                    proposal_list = self.rpn_head.simple_test_rpn(x_fpn, img_metas)
                else:
                    proposal_list = proposals
                return self.roi_head.simple_test(x_fpn, proposal_list, img_metas, rescale=rescale), x_backbone, x_fpn

        x = self.extract_feat(img)
        # hook the fpn features
        self.fpn_features = x
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals
        if debug_cfg:
            if 'given_proposal_list' in debug_cfg:
                if debug_cfg['given_proposal_list']:
                    out_dir = debug_cfg['out_dir']
                    out_dir = out_dir.replace('given', 'augmix.wotrans_plus_rpn.tailv2.1.none_roi.none.none__e2_lw.12')
                    out_dir = out_dir.replace('gaussian_noise/1', 'gaussian_noise/0')
                    out_dir = out_dir.replace('gaussian_noise/2', 'gaussian_noise/0')
                    name = f"{img_metas[0]['ori_filename'].split('.png')[0]}_proposal_list"
                    proposal_list = torch.load(f"{out_dir}/{name}.pt")
            if 'given_proposal_list2' in debug_cfg:
                if debug_cfg['given_proposal_list2']:
                    out_dir = debug_cfg['out_dir']
                    out_dir = out_dir.replace('given2', 'augmix.wotrans_plus_rpn.tailv2.1.none_roi.none.none__e2_lw.12')
                    out_dir = out_dir.replace('gaussian_noise/0', 'gaussian_noise/2')
                    name = f"{img_metas[0]['ori_filename'].split('.png')[0]}_proposal_list"
                    proposal_list = torch.load(f"{out_dir}/{name}.pt")


        bbox_results = self.roi_head.simple_test(x, proposal_list, img_metas, rescale=rescale)

        if debug_cfg:
            if debug_cfg and ('proposal_list' in debug_cfg['save_list']):
                fn = get_file_name(debug_cfg, 'proposal_list', extension='pt', img_meta=img_metas[0])
                torch.save(proposal_list, fn)
            visualize_image(img_meta=img_metas[0], name='original_image', debug_cfg=debug_cfg)
            visualize_score_distribution(proposal_list[0][:, 4], name='proposal_list_score_distribution', bins=50, img_meta=img_metas[0], debug_cfg=debug_cfg)
            visualize_score_density(proposal_list[0], name='proposal_list_score_density', img_meta=img_metas[0], topk=300, debug_cfg=debug_cfg)
            visualize_score_distribution(np.concatenate(bbox_results[0], 0)[:, 4], name='bbox_results_score_distribution', bins=50, img_meta=img_metas[0], debug_cfg=debug_cfg)
            visualize_score_density(bbox_results[0], name='bbox_results_score_density', img_meta=img_metas[0], debug_cfg=debug_cfg)

        return bbox_results

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        x = self.extract_feats(imgs)
        proposal_list = self.rpn_head.aug_test_rpn(x, img_metas)
        return self.roi_head.aug_test(
            x, proposal_list, img_metas, rescale=rescale)

    def onnx_export(self, img, img_metas):

        img_shape = torch._shape_as_tensor(img)[2:]
        img_metas[0]['img_shape_for_onnx'] = img_shape
        x = self.extract_feat(img)
        proposals = self.rpn_head.onnx_export(x, img_metas)
        if hasattr(self.roi_head, 'onnx_export'):
            return self.roi_head.onnx_export(x, proposals, img_metas)
        else:
            raise NotImplementedError(
                f'{self.__class__.__name__} can not '
                f'be exported to ONNX. Please refer to the '
                f'list of supported models,'
                f'https://mmdetection.readthedocs.io/en/latest/tutorials/pytorch2onnx.html#list-of-supported-models-exportable-to-onnx'  # noqa E501
            )
