# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...builder import LOSSES
from ..utils import weight_reduce_loss
import mmdet.models.detectors.base as base
from .contrastive_loss import supcontrast, supcontrastv0_01, supcontrastv0_02


@LOSSES.register_module()
class ContrastiveLossPlus(nn.Module):

    def __init__(self,
                 version='0.0.1',
                 loss_weight=0.01,
                 **kwargs):
        """ContrastiveLossPlus."""
        super(ContrastiveLossPlus, self).__init__()
        self.version = version
        self.loss_weight = loss_weight
        self.kwargs = kwargs

        if self.version in ['0.0.1', '0.0.2']:
            self.loss_criterion = supcontrast
        else:
            raise NotImplementedError(f'does not support version=={version}')

        if self.version in ['0.0.1']:
            self.target = 'bbox_feats'
        elif self.version in ['0.0.2']:
            self.target = 'cls_feats'
        else:
            raise NotImplementedError(f'does not support version=={version}')


    def filter_and_collect(self,
                           gt_bboxes,
                           gt_instance_inds):
        # Only gt_instance_inds shared by all views are classified as valid_inds.
        assert gt_instance_inds is not None
        if len(gt_instance_inds[0]) != len(gt_instance_inds[1]) or \
                len(gt_instance_inds[0]) != len(gt_instance_inds[2]):
            valid_gt_bboxes = []
            valid_gt_instance_inds = gt_instance_inds[0].tolist()
            for i in range(len(gt_bboxes)):
                valid_gt_bbox = torch.zeros_like(gt_bboxes[0])

                bbox_index = 0
                for instance_index in gt_instance_inds[0]:
                    if instance_index in gt_instance_inds[i]:
                        valid_gt_bbox[instance_index] = gt_bboxes[i][bbox_index]
                        bbox_index += 1
                    else:
                        valid_gt_bbox[instance_index] = -1 * torch.ones(4)
                        if instance_index in valid_gt_instance_inds:
                            valid_gt_instance_inds.remove(instance_index)
                valid_gt_bboxes.append(valid_gt_bbox)
            valid_gt_instance_inds = torch.tensor(valid_gt_instance_inds)
        else:
            valid_gt_bboxes = gt_bboxes
            valid_gt_instance_inds = gt_instance_inds[0]

        # Filter using valid_inds
        valid_gt_rois = []
        if len(valid_gt_instance_inds) != 0:
            for i in range(len(valid_gt_bboxes)):
                valid_gt_bbox_ = valid_gt_bboxes[i][valid_gt_instance_inds]
                batch_inds_ = valid_gt_bbox_.new_full((valid_gt_bbox_.size(0), 1), i)
                valid_roi_ = torch.cat([batch_inds_, valid_gt_bbox_], dim=1)
                valid_gt_rois.append(valid_roi_)
                del valid_gt_bbox_, valid_roi_

        return valid_gt_bboxes, valid_gt_instance_inds, valid_gt_rois


    def forward(self,
                bbox_results,
                labels,
                valid_instance_inds,
                **kwargs):
        """Forward function.

        Args:
        Returns:
            torch.Tensor: The calculated loss.
        """
        if len(bbox_results) == 0:
            return torch.zeros(1)

        loss_feat = self.loss_criterion(bbox_results[0][self.target],
                                        bbox_results[1][self.target],
                                        bbox_results[2][self.target],
                                        labels=labels[0][valid_instance_inds])

        return self.loss_weight * loss_feat

