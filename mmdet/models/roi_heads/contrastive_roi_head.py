from mmdet.core import bbox2roi
from ..builder import HEADS

from .standard_roi_head import StandardRoIHead


@HEADS.register_module()
class ContrastiveRoIHead(StandardRoIHead):
    """Simplest base roi head including one bbox head and one mask head."""
    def __init__(self,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 shared_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 **kwargs):
        super(ContrastiveRoIHead, self).__init__(bbox_roi_extractor,
                                                 bbox_head,
                                                 mask_roi_extractor,
                                                 mask_head,
                                                 shared_head,
                                                 train_cfg,
                                                 test_cfg,
                                                 pretrained,
                                                 init_cfg)
        self.contrastive_head = None

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
                sampling_results_batch = []
                for i in range(kwargs['batch_size']):
                    assign_result = self.bbox_assigner.assign(
                        proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                        gt_labels[i])
                    sampling_result = self.bbox_sampler.sample(
                        assign_result,
                        proposal_list[i],
                        gt_bboxes[i],
                        gt_labels[i],
                        feats=[lvl_feat[i][None] for lvl_feat in x])
                    sampling_results_batch.append(sampling_result)
                for _ in range(kwargs['num_views']):
                    sampling_results.extend(sampling_results_batch)

        losses = dict()

        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(x, sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    img_metas, **kwargs)
            losses.update(bbox_results['loss_bbox'])

        if self.contrastive_head:
            pass


        # mask head forward and loss
        if self.with_mask:
            mask_results = self._mask_forward_train(x, sampling_results,
                                                    bbox_results['bbox_feats'],
                                                    gt_masks, img_metas)
            losses.update(mask_results['loss_mask'])

        return losses


    def _bbox_forward(self, x, rois):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred, cont_feats = self.bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, cont_feats=cont_feats,
            bbox_feats=bbox_feats, cls_feats=self.bbox_head.cls_feats)
        return bbox_results


    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas, gt_instance_inds=None, **kwargs):
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)

        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg) # labels, label_weights, bbox_targets, bbox_weights

        self.bbox_targets = bbox_targets
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'],
                                        bbox_results['cont_feats'],
                                        rois,
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

        return bbox_results
