import copy

import numpy as np
from numpy import random

from ..builder import PIPELINES

try:
    from imagecorruptions import corrupt
except ImportError:
    corrupt = None

try:
    import albumentations
    from albumentations import Compose
except ImportError:
    albumentations = None
    Compose = None

from PIL import Image
from mmdet.datasets.pipelines.augmix import (autocontrast, equalize, posterize, solarize, color,
                                             contrast, brightness, sharpness,
                                             rotate, shear_x, shear_y, translate_x, translate_y,)


# BBoxOnlyAugmentation
# REF: https://github.com/poodarchu/learn_aug_for_object_detection.numpy/
def _apply_bbox_only_augmentation(img, bbox_xy, aug_func, **kwargs):
    '''
    Args:
        img     : (np.array) (img_width, img_height, channel)
        bbox_xy : (tensor) [x1, y1, x2, y2]
        aug_func: (func) can be contain 'level', 'img_size', etc.
    '''
    if not isinstance(img, np.ndarray):
        img = np.asarray(img)

    # Get bbox_content from image
    img_height, img_width = img.shape[0], img.shape[1]
    x1, y1, x2, y2 = int(bbox_xy[0]), int(bbox_xy[1]), int(bbox_xy[2]), int(bbox_xy[3])
    bbox_content = img[y1:y2+1, x1:x2+1, :]

    # Augment
    kwargs['img_size'] = Image.fromarray(bbox_content).size
    augmented_bbox_content = aug_func(Image.fromarray(bbox_content), **kwargs)
    augmented_bbox_content = np.asarray(augmented_bbox_content)

    # Pad with pad_width: [[before_1, after_1], [before_2, after_2], ..., [before_N, after_N]]
    pad_width = [[y1, img_height - 1 - y2], [x1, img_width - 1 - x2], [0, 0]]
    augmented_bbox_content = np.pad(augmented_bbox_content, pad_width, 'constant', constant_values=0)

    mask = np.zeros_like(bbox_content)
    mask = np.pad(mask, pad_width, 'constant', constant_values=1)

    # Overwrite augmented_bbox_content into img
    img = img * mask + augmented_bbox_content

    return img


def _apply_bboxes_only_augmentation(img, bboxes_xy, aug_func, **kwargs):
    '''
    Args:
        img         : (np.array) (img_width, img_height, channel)
        bboxes_xy   : (tensor) has shape of (num_bboxes, 4) with [x1, y1, x2, y2]
        aug_func    : (func) The argument is bbox_content # TODO: severity?
    '''
    for i in range(len(bboxes_xy)):
        img = _apply_bbox_only_augmentation(img, bboxes_xy[i], aug_func, **kwargs)
    return Image.fromarray(img)


def bboxes_only_rotate(pil_img, bboxes_xy, level, img_size):
    return _apply_bboxes_only_augmentation(pil_img, bboxes_xy, rotate, level=level, img_size=img_size)


def bboxes_only_shear_x(pil_img, bboxes_xy, level, img_size):
    return _apply_bboxes_only_augmentation(pil_img, bboxes_xy, shear_x, level=level, img_size=img_size)


def bboxes_only_shear_y(pil_img, bboxes_xy, level, img_size):
    return _apply_bboxes_only_augmentation(pil_img, bboxes_xy, shear_y, level=level, img_size=img_size)


def bboxes_only_translate_x(pil_img, bboxes_xy, level, img_size, **kwargs):
    return _apply_bboxes_only_augmentation(pil_img, bboxes_xy, translate_x, level=level, img_size=img_size)


def bboxes_only_translate_y(pil_img, bboxes_xy, level, img_size):
    return _apply_bboxes_only_augmentation(pil_img, bboxes_xy, translate_y, level=level, img_size=img_size)


def get_aug_list(version):
    if version == '0.1':
        aug_list = [
            # color
            autocontrast, equalize, posterize, solarize, color, contrast, brightness, sharpness,

            # geometric
            # rotate, shear_x, shear_y, translate_x, translate_y,

            # bounding box only
            bboxes_only_rotate, bboxes_only_shear_x, bboxes_only_shear_y,
            bboxes_only_translate_x, bboxes_only_translate_y,
        ]
    else:
        raise NotImplementedError
    return aug_list


@PIPELINES.register_module()
class AugMixDetection:
    def __init__(self, mean, std,
                 num_views=3,
                 version='0.1',
                 aug_severity=6,
                 to_rgb=True,):
        super(AugMixDetection, self).__init__()
        self.mixture_width = 3
        self.mixture_depth = -1
        self.aug_prob_coeff = 1.

        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.num_views = num_views
        self.aug_list = get_aug_list(version)
        self.aug_severity = aug_severity
        self.to_rgb = to_rgb

    def __call__(self, results, *args, **kwargs):
        if self.num_views == 1:
            return self.aug_and_mix(results['img'], results['gt_bboxes'])

        for i in range(2, self.num_views+1):
            img_augmix = self.aug_and_mix(results['img'].copy(), results['gt_bboxes'])
            results[f'img{i}'] = np.array(img_augmix, dtype=results['img'].dtype)
            results[f'gt_bboxes{i}'] = copy.deepcopy(results['gt_bboxes']) # TODO: allow to geometric operations containing bbox
            results[f'gt_labels{i}'] = copy.deepcopy(results['gt_labels']) # TODO: allow to geometric operations containing bbox
            results['img_fields'].append(f'img{i}')
            results['bbox_fields'].append(f'gt_bboxes{i}')

        return results


    def aug_and_mix(self, img_orig, gt_bboxes):
        # TODO: change library to albumentation. It will make it faster.
        img_height, img_width, _ = img_orig.shape
        img_size = (img_width, img_height)

        # Sample
        #   > mixing_weights: [w1, w2, ..., wk] ~ Dirichlet(alpha, alpha, ..., alpha)
        #   > sample_weight: m ~ Beta(alpha, alpha)
        mixing_weights = np.float32(np.random.dirichlet([self.aug_prob_coeff] * self.mixture_width))
        sample_weight = np.float32(np.random.beta(self.aug_prob_coeff, self.aug_prob_coeff))

        # Fill x_aug with zeros
        img_mix = np.zeros_like(img_orig.copy(), dtype=np.float32)
        for i in range(self.mixture_width):
            # Sample operations : [op1, op2, op3] ~ O
            depth = self.mixture_depth if self.mixture_depth > 0 else np.random.randint(1, 4)
            op_chain = np.random.choice(self.aug_list, depth, replace=False) # not allow same aug if replace is False.

            # Augment
            img_aug = self.chain(img_orig.copy(), op_chain, img_size, gt_bboxes=gt_bboxes)

            # Mixing
            img_aug = np.asarray(img_aug, dtype=np.float32)
            img_mix += mixing_weights[i] * img_aug

        img_augmix = (1-sample_weight) * img_orig + sample_weight * img_mix
        img_augmix = np.array(img_augmix, dtype=np.float32)
        return img_augmix

    def chain(self, img, op_chain, img_size, gt_bboxes=None):
        '''
        img: np.array
        '''
        img_aug = Image.fromarray(img, 'RGB') # pil_img

        for op in op_chain:
            img_aug = op(img_aug, level=self.aug_severity,
                         img_size=img_size, bboxes_xy=gt_bboxes)

        return img_aug

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str



