# Copyright (c) OpenMMLab. All rights reserved.
import copy

from ..builder import PIPELINES
from .compose import Compose
from .auto_aug_det_utils import *


@PIPELINES.register_module()
class AutoAugDet:
    def __init__(self, policies, num_gen=2, no_jsd=False):
        assert isinstance(policies, list) and len(policies) > 0, \
            'Policies must be a non-empty list.'
        for policy in policies:
            assert isinstance(policy, list) and len(policy) > 0, \
                'Each policy in policies must be a non-empty list.'
            for augment in policy:
                assert isinstance(augment, dict) and 'type' in augment, \
                    'Each specific augmentation must be a dict with key' \
                    ' "type".'

        self.policies = copy.deepcopy(policies)
        self.transforms = [Compose(policy) for policy in self.policies]
        self.num_gen = num_gen
        self.no_jsd = no_jsd

    def __call__(self, results):
        # Define fields
        FIELDS = ['img_fields', 'bbox_fields']
        field_dict = dict()
        for field in FIELDS:
            field_dict[field] = []

        # Assign gt_instance_inds
        results['gt_instance_inds'] = np.arange(len(results['gt_bboxes']))
        results['custom_field'] = []
        results['custom_field'].append('gt_instance_inds')

        # Augments
        if self.no_jsd:
            transform = np.random.choice(self.transforms)
            transform(results)
        else:
            for i in range(2, self.num_gen+2, 1):
                transformed_result = copy.deepcopy(results)
                transform = np.random.choice(self.transforms)
                transform(transformed_result)
                for field in FIELDS:
                    for key in transformed_result.get(field, []):
                        results[f'{key}{i}'] = transformed_result[key].copy()
                        field_dict[field].append(f'{key}{i}')
                        results['custom_field'].append(f"{key}{i}")
                for key in ['gt_labels', 'gt_labels_ignore', 'gt_masks', 'gt_masks_ignore', 'gt_semantic_seg', 'gt_instance_inds']:
                    if key in transformed_result:
                        results[f'{key}{i}'] = transformed_result[key].copy()
                        results['custom_field'].append(f"{key}{i}")

        # Assign values into each field
        for field, value_list in field_dict.items():
            for value in value_list:
                results[field].append(value)

        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(policies={self.policies})'