# Copyright (c) OpenMMLab. All rights reserved.

import random
from mmdet.datasets.pipelines.gen_auto_augment import *

_MAX_LEVEL = 10

def get_policies(policy):
    aug_geometric = [Shear, Rotate, Translate]
    aug_color = [ColorTransform, EqualizeTransform, BrightnessTransform, ContrastTransform]
    # aug_bbox = [] # TODO
    aug_all = list(set(aug_geometric + aug_color))

    if policy == 'aug_geometric':
        return aug_geometric
    elif policy == 'aug_color':
        return aug_color
    elif policy == 'aug_all':
        return aug_all
    elif policy == '0.1':
        return [Shear, Rotate, ColorTransform, EqualizeTransform, BrightnessTransform, ContrastTransform]
    else:
        raise TypeError(f'The policy is not supported for {policy}')

@PIPELINES.register_module()
class RandAutoAugment:
    """Auto augmentation.

    This data augmentation is proposed in `Learning Data Augmentation
    Strategies for Object Detection <https://arxiv.org/pdf/1906.11172>`_.

    TODO: Implement 'Sharpness' transforms
    TODO: Implement 'posterize', 'solarize', 'zoom'

    Args:
        policies (str): The policies of auto augmentation.
    Examples:
        >>> replace = (104, 116, 124)
        >>> policies = [
        >>>     [
        >>>         dict(type='Sharpness', prob=0.0, level=8),
        >>>         dict(
        >>>             type='Shear',
        >>>             prob=0.4,
        >>>             level=0,
        >>>             replace=replace,
        >>>             axis='x')
        >>>     ],
        >>>     [
        >>>         dict(
        >>>             type='Rotate',
        >>>             prob=0.6,
        >>>             level=10,
        >>>             replace=replace),
        >>>         dict(type='Color', prob=1.0, level=6)
        >>>     ]
        >>> ]
        >>> augmentation = AutoAugment(policies)
        >>> img = np.ones(100, 100, 3)
        >>> gt_bboxes = np.ones(10, 4)
        >>> results = dict(img=img, gt_bboxes=gt_bboxes)
        >>> results = augmentation(results)
    """

    def __init__(self, policy, num_gen=2,
                 depth=-1, min_depth=1, max_depth=3,
                 level=-1, min_level=0, max_level=10):
        assert isinstance(policy, str), 'Policies must be a string.'

        self.policies = get_policies(policy)
        self.num_gen = num_gen

        assert min_depth > 0, 'min_depth should be larger than zero.'
        self.depth = depth # If depth is -1, then the depth will be chosen randomly from min_depth to max_depth.
        self.min_depth, self.max_depth = min_depth, max_depth

        assert (max_level <= 10 and max_level >=0) and \
               (min_level <= 10 and min_level >= 0) and (level <= 10), \
            'level should be in [0, 10]'
        assert min_level <= max_level, 'max_level should be larger than or equal to min_level'
        self.min_level, self.max_level = min_level, max_level
        self.level = level # If level is -1, then the level will be chosen randomly from min_level to max_level.


    def __call__(self, results):
        # Define fields
        FIELDS = ['img_fields', 'bbox_fields', 'mask_fields', 'seg_fields']
        field_dict = dict()
        for field in FIELDS:
            field_dict[field] = []

        # Assign gt_instance_inds
        results['gt_instance_inds'] = np.arange(len(results['gt_bboxes']))
        results['custom_field'] = []
        results['custom_field'].append('gt_instance_inds')

        # Augments
        for i in range(2, self.num_gen+2, 1):
            transformed_result = copy.deepcopy(results)
            depth = self.depth if self.depth > 0 else \
                np.random.randint(low=self.min_depth, high=self.max_depth+1)

            transforms = random.sample(self.policies, depth)

            for _op in transforms:
                if _op in [EqualizeTransform]:
                    transform = _op(prob=1.0)
                else:
                    level_list = [i for i in range(self.min_level, self.max_level + 1, 1)]
                    if _op in [ColorTransform, BrightnessTransform, ContrastTransform]:
                        if 5 in level_list:
                            level_list.remove(5)
                    level = random.choice(level_list)
                    transform = _op(prob=1.0, level=level)

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
