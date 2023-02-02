import copy

import numpy as np
# from numpy import random
import random

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
def _apply_bbox_only_augmentation(img, bbox_xy, aug_func, fillmode=None, fillcolor=None, **kwargs):
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
    if fillmode is None:
        bbox_content = img[y1:y2+1, x1:x2+1, :]
        mask = img[y1:y2+1, x1:x2+1, :]
        center = None
    elif fillmode == 'img':
        bbox_content = img
        mask = img[y1:y2 + 1, x1:x2 + 1, :]
        center = ((x1 + x2) / 2., (y1 + y2) / 2.)
        kwargs['img_size_for_level'] = Image.fromarray(mask).size
    else:
        raise TypeError

    # Augment
    kwargs['img_size'] = Image.fromarray(bbox_content).size
    augmented_bbox_content = aug_func(Image.fromarray(bbox_content), **kwargs, fillcolor=fillcolor, center=center)
    augmented_bbox_content = np.asarray(augmented_bbox_content)
    if fillmode == 'img':
        augmented_bbox_content = augmented_bbox_content[y1:y2+1, x1:x2+1, :]

    # Pad with pad_width: [[before_1, after_1], [before_2, after_2], ..., [before_N, after_N]]
    pad_width = [[y1, max(0, img_height - y2 - 1)], [x1, max(0, img_width - x2 - 1)], [0, 0]]
    if fillcolor is None:
        augmented_bbox_content = np.pad(augmented_bbox_content, pad_width, 'constant', constant_values=0)
    else:
        augmented_bbox_content = np.pad(augmented_bbox_content, pad_width, 'constant', constant_values=fillcolor[0])

    mask = np.zeros_like(mask)
    mask = np.pad(mask, pad_width, 'constant', constant_values=1)

    # Overwrite augmented_bbox_content into img
    img = img * mask + augmented_bbox_content * (mask==0)

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


def bboxes_only_rotate(pil_img, bboxes_xy, level, img_size, **kwargs):
    return _apply_bboxes_only_augmentation(pil_img, bboxes_xy, rotate, level=level, img_size=img_size, **kwargs)


def bboxes_only_shear_x(pil_img, bboxes_xy, level, img_size, **kwargs):
    return _apply_bboxes_only_augmentation(pil_img, bboxes_xy, shear_x, level=level, img_size=img_size, **kwargs)


def bboxes_only_shear_y(pil_img, bboxes_xy, level, img_size, **kwargs):
    return _apply_bboxes_only_augmentation(pil_img, bboxes_xy, shear_y, level=level, img_size=img_size, **kwargs)


def bboxes_only_shear_xy(pil_img, bboxes_xy, level, img_size, **kwargs):
    func = bboxes_only_shear_x if np.random.rand() < 0.5 else bboxes_only_shear_y
    return func(pil_img, bboxes_xy, level, img_size, **kwargs)


def bboxes_only_translate_x(pil_img, bboxes_xy, level, img_size, **kwargs):
    return _apply_bboxes_only_augmentation(pil_img, bboxes_xy, translate_x, level=level, img_size=img_size, **kwargs)


def bboxes_only_translate_y(pil_img, bboxes_xy, level, img_size, **kwargs):
    return _apply_bboxes_only_augmentation(pil_img, bboxes_xy, translate_y, level=level, img_size=img_size, **kwargs)


def bboxes_only_translate_xy(pil_img, bboxes_xy, level, img_size, **kwargs):
    func = bboxes_only_translate_x if np.random.rand() < 0.5 else bboxes_only_translate_y
    return func(pil_img, bboxes_xy, level, img_size, **kwargs)


# Random bboxes only augmentation
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
def generate_random_bboxes_xy(img_size, num_bboxes, bboxes_xy=None,
                              scales=(0.01, 0.2), ratios=(0.3, 1/0.3),
                              max_iters=100, eps=1e-6, **kwargs):
    # REF: mmdetection/mmdet/datasets/pipelines/transforms.py Cutout
    if isinstance(num_bboxes, tuple) or isinstance(num_bboxes, list):
        num_bboxes = np.random.randint(num_bboxes[0], num_bboxes[1] + 1)
    (img_width, img_height) = img_size

    random_bboxes_xy = np.zeros((num_bboxes, 4))
    total_bboxes = 0
    for i in range(max_iters):
        if total_bboxes >= num_bboxes:
            break

        # Generate a random bbox.
        x1, y1 = np.random.randint(0, img_width), np.random.randint(0, img_height)
        scale = np.random.uniform(*scales) * img_height * img_width
        ratio = np.random.uniform(*ratios)
        bbox_w, bbox_h = int(np.sqrt(scale / ratio)), int(np.sqrt(scale * ratio))
        random_bbox = np.array([[x1, y1, min(x1 + bbox_w, img_width), min(y1 + bbox_h, img_height)]])
        if bboxes_xy is not None:
            ious = bbox_overlaps(random_bbox, bboxes_xy)
            if np.sum(ious) > eps:
                continue
        random_bboxes_xy[total_bboxes, :] = random_bbox[0]
        total_bboxes += 1
    if total_bboxes != num_bboxes:
        random_bboxes_xy = random_bboxes_xy[:total_bboxes, :]

    return random_bboxes_xy


def random_bboxes_only_rotate(pil_img, bboxes_xy, level, img_size, num_bboxes, **kwargs):
    random_bboxes_xy = generate_random_bboxes_xy(img_size, num_bboxes, bboxes_xy, **kwargs)
    return _apply_bboxes_only_augmentation(pil_img, random_bboxes_xy, rotate, level=level, img_size=img_size, **kwargs)


def random_bboxes_only_shear_xy(pil_img, bboxes_xy, level, img_size, num_bboxes, **kwargs):
    func = bboxes_only_shear_x if np.random.rand() < 0.5 else bboxes_only_shear_y
    random_bboxes_xy = generate_random_bboxes_xy(img_size, num_bboxes, bboxes_xy, **kwargs)
    return func(pil_img, random_bboxes_xy, level, img_size, **kwargs)


def random_bboxes_only_translate_xy(pil_img, bboxes_xy, level, img_size, num_bboxes, **kwargs):
    func = bboxes_only_translate_x if np.random.rand() < 0.5 else bboxes_only_translate_y
    random_bboxes_xy = generate_random_bboxes_xy(img_size, num_bboxes, bboxes_xy, **kwargs)
    return func(pil_img, random_bboxes_xy, level, img_size, **kwargs)


# Random bboxes + ground-truth bboxes only augmentation
def random_gt_only_rotate(pil_img, bboxes_xy, level, img_size, num_bboxes, sample_gt_ratio=1, **kwargs):
    random_bboxes_xy = generate_random_bboxes_xy(img_size, num_bboxes, bboxes_xy, **kwargs)
    num_gt_samples = int(len(bboxes_xy)*sample_gt_ratio) if len(bboxes_xy) > 1 else len(bboxes_xy)
    gt_bboxes_xy = np.stack(random.sample(list(bboxes_xy), num_gt_samples))
    random_gt_bboxes_xy = np.concatenate([random_bboxes_xy, gt_bboxes_xy], axis=0)
    return _apply_bboxes_only_augmentation(pil_img, random_gt_bboxes_xy, rotate, level=level, img_size=img_size, **kwargs)


def random_gt_only_shear_xy(pil_img, bboxes_xy, level, img_size, num_bboxes, sample_gt_ratio=1, **kwargs):
    func = bboxes_only_shear_x if np.random.rand() < 0.5 else bboxes_only_shear_y
    random_bboxes_xy = generate_random_bboxes_xy(img_size, num_bboxes, bboxes_xy, **kwargs)
    num_gt_samples = int(len(bboxes_xy) * sample_gt_ratio) if len(bboxes_xy) > 1 else len(bboxes_xy)
    gt_bboxes_xy = np.stack(random.sample(list(bboxes_xy), num_gt_samples))
    random_gt_bboxes_xy = np.concatenate([random_bboxes_xy, gt_bboxes_xy], axis=0)
    return func(pil_img, random_gt_bboxes_xy, level, img_size, **kwargs)


def random_gt_only_translate_xy(pil_img, bboxes_xy, level, img_size, num_bboxes, sample_gt_ratio=1, **kwargs):
    func = bboxes_only_translate_x if np.random.rand() < 0.5 else bboxes_only_translate_y
    random_bboxes_xy = generate_random_bboxes_xy(img_size, num_bboxes, bboxes_xy, **kwargs)
    num_gt_samples = int(len(bboxes_xy) * sample_gt_ratio) if len(bboxes_xy) > 1 else len(bboxes_xy)
    gt_bboxes_xy = np.stack(random.sample(list(bboxes_xy), num_gt_samples))
    random_gt_bboxes_xy = np.concatenate([random_bboxes_xy, gt_bboxes_xy], axis=0)
    return func(pil_img, random_gt_bboxes_xy, level, img_size, **kwargs)


# Background only augmentation
def _apply_bg_only_augmentation(img, bboxes_xy, aug_func, fillcolor=None, **kwargs):
    '''
    Args:
        img         : (np.array) (img_width, img_height, channel)
        bboxes_xy   : (tensor) has shape of (num_bboxes, 4) with [x1, y1, x2, y2]
        aug_func    : (func) The argument is bbox_content # TODO: severity?
    '''
    if not isinstance(img, np.ndarray):
        img = np.asarray(img)

    # Make the union of bboxes
    bbox_content = img.copy()
    mask = np.zeros_like(bbox_content)
    for i in range(len(bboxes_xy)):
        bbox_xy = bboxes_xy[i]
        x1, y1, x2, y2 = int(bbox_xy[0]), int(bbox_xy[1]), int(bbox_xy[2]), int(bbox_xy[3])
        if fillcolor is None:
            bbox_content[y1:y2 + 1, x1:x2 + 1, :] = 0.0
        elif isinstance(fillcolor, tuple):
            assert len(fillcolor) == bbox_content.shape[-1]
            for ch in range(len(fillcolor)):
                bbox_content[y1:y2 + 1, x1:x2 + 1, ch].fill(fillcolor[ch])
        else:
            bbox_content[y1:y2 + 1, x1:x2 + 1, :].fill(fillcolor)
        mask[y1:y2 + 1, x1:x2 + 1, :] = 1

    # Augment
    kwargs['img_size'] = Image.fromarray(bbox_content).size
    augmented_bbox_content = aug_func(Image.fromarray(bbox_content), **kwargs, fillcolor=fillcolor)
    augmented_bbox_content = np.asarray(augmented_bbox_content)

    # Overwrite augmented_bbox_content into img
    img = img * mask + augmented_bbox_content * (mask==0)

    return img


def bg_only_rotate(pil_img, bboxes_xy, level, img_size, **kwargs):
    return _apply_bg_only_augmentation(pil_img, bboxes_xy, rotate, level=level, img_size=img_size, **kwargs)


def bg_only_shear_x(pil_img, bboxes_xy, level, img_size, **kwargs):
    return _apply_bg_only_augmentation(pil_img, bboxes_xy, shear_x, level=level, img_size=img_size, **kwargs)


def bg_only_shear_y(pil_img, bboxes_xy, level, img_size, **kwargs):
    return _apply_bg_only_augmentation(pil_img, bboxes_xy, shear_y, level=level, img_size=img_size, **kwargs)


def bg_only_shear_xy(pil_img, bboxes_xy, level, img_size, **kwargs):
    func = bg_only_shear_x if np.random.rand() < 0.5 else bg_only_shear_y
    return func(pil_img, bboxes_xy, level, img_size, **kwargs)


def bg_only_translate_x(pil_img, bboxes_xy, level, img_size, **kwargs):
    return _apply_bg_only_augmentation(pil_img, bboxes_xy, translate_x, level=level, img_size=img_size, **kwargs)


def bg_only_translate_y(pil_img, bboxes_xy, level, img_size, **kwargs):
    return _apply_bg_only_augmentation(pil_img, bboxes_xy, translate_y, level=level, img_size=img_size, **kwargs)


def bg_only_translate_xy(pil_img, bboxes_xy, level, img_size, **kwargs):
    func = bg_only_translate_x if np.random.rand() < 0.5 else bg_only_translate_y
    return func(pil_img, bboxes_xy, level, img_size, **kwargs)


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
        return aug_list
    elif version in ['0.2', '0.3']:
        aug_color_list = [autocontrast, equalize, posterize, solarize, color, contrast, brightness, sharpness,]
        aug_geo_list = [
            bboxes_only_rotate, bboxes_only_shear_x, bboxes_only_shear_y,
            bboxes_only_translate_x, bboxes_only_translate_y,
        ]
        return aug_color_list, aug_geo_list
    elif version in ['1.1']:
        aug_color_list = [autocontrast, equalize, posterize, solarize]
        aug_geo_list = [bboxes_only_rotate, bboxes_only_shear_x, bboxes_only_shear_y,
                       bboxes_only_translate_x, bboxes_only_translate_y]
        return aug_color_list, aug_geo_list
    elif version in ['1.2']:
        aug_color_list = [autocontrast, equalize, posterize, solarize]
        aug_bg_only_list = [bg_only_rotate, bg_only_shear_x, bg_only_shear_y,
                                bg_only_translate_x, bg_only_translate_y]
        aug_bbox_only_list = [bboxes_only_rotate, bboxes_only_shear_x, bboxes_only_shear_y,
                              bboxes_only_translate_x, bboxes_only_translate_y]
        return aug_color_list, aug_bg_only_list, aug_bbox_only_list
    elif version in ['1.3', '1.3.1']:
        aug_list = [autocontrast, equalize, posterize, solarize, # color
                    bg_only_rotate, bg_only_shear_x, bg_only_shear_y,
                    bg_only_translate_x, bg_only_translate_y, # bg only transformation
                    bboxes_only_rotate, bboxes_only_shear_x, bboxes_only_shear_y,
                    bboxes_only_translate_x, bboxes_only_translate_y] # bbox only transformation
        return aug_list
    elif version in ['1.4', '1.4.1', '1.4.2', '1.4.3']:
        aug_list = [autocontrast, equalize, posterize, solarize,  # color
                    bg_only_rotate, bg_only_shear_xy, bg_only_translate_xy,  # bg only transformation
                    bboxes_only_rotate, bboxes_only_shear_xy, bboxes_only_translate_xy]  # bbox only transformation
        return aug_list
    elif version in ['1.5', '1.5.0', '1.5.1', '1.5.2', '1.5.3', '1.5.4', '1.5.5', '1.5.6',
                     '1.8', '1.8.1']:
        aug_list = [autocontrast, equalize, posterize, solarize,  # color
                    random_bboxes_only_rotate, random_bboxes_only_shear_xy, random_bboxes_only_translate_xy,  # random_bboxes only transformation
                    bboxes_only_rotate, bboxes_only_shear_xy, bboxes_only_translate_xy] # bbox only transformation
        return aug_list
    elif version in ['1.6']:
        aug_list = [autocontrast, equalize, posterize, solarize,  # color
                    random_gt_only_rotate, random_gt_only_shear_xy, random_gt_only_translate_xy]
        return aug_list
    elif version in ['1.7']:
        aug_list = [autocontrast, equalize, posterize, solarize,  # color
                    random_gt_only_rotate, random_gt_only_shear_xy, random_gt_only_translate_xy, # random bboxes and gt bboxes only transformation
                    bg_only_rotate, bg_only_shear_xy, bg_only_translate_xy]  # bg only transformation
        return aug_list
    else:
        raise NotImplementedError


GEO_OP_LIST = [bg_only_rotate, bg_only_shear_xy, bg_only_shear_x, bg_only_shear_y, # bg only transformation
               bg_only_translate_xy, bg_only_translate_x, bg_only_translate_y,
               bboxes_only_rotate, bboxes_only_shear_xy, bboxes_only_shear_x, bboxes_only_shear_y, # bboxes only transformation
               bboxes_only_translate_xy, bboxes_only_translate_x, bboxes_only_translate_y,
               random_bboxes_only_rotate, random_bboxes_only_shear_xy, random_bboxes_only_translate_xy, # random_bboxes only transformation
               random_gt_only_rotate, random_gt_only_shear_xy, random_gt_only_translate_xy # random bboxes and gt bboxes only transfromation
               ]

@PIPELINES.register_module()
class AugMixDetection:
    def __init__(self, mean, std,
                 num_views=3,
                 version='0.1',
                 aug_severity=6,
                 mixture_depth=-1,
                 geo_severity=None,
                 to_rgb=True,
                 **kwargs):
        super(AugMixDetection, self).__init__()
        self.mixture_width = 3
        self.aug_prob_coeff = 1.

        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.num_views = num_views
        self.version = version
        self.aug_list = get_aug_list(version)
        self.aug_severity = aug_severity
        self.mixture_depth = mixture_depth
        self.to_rgb = to_rgb

        self.geo_severity = geo_severity
        self.kwargs = kwargs

    def __call__(self, results, *args, **kwargs):
        if self.num_views == 1:
            results['img'] = np.asarray(self.aug_and_mix(results['img'], results['gt_bboxes']), dtype=results['img'].dtype)
            return results

        results['custom_field'] = []
        for i in range(2, self.num_views+1):
            if isinstance(self.aug_list, tuple):
                img_augmix = self.multiaug_and_mix(results['img'].copy(), results['gt_bboxes'])
            else:
                img_augmix = self.aug_and_mix(results['img'].copy(), results['gt_bboxes'])
            results[f'img{i}'] = np.array(img_augmix, dtype=results['img'].dtype)
            results[f'gt_bboxes{i}'] = copy.deepcopy(results['gt_bboxes']) # TODO: allow to geometric operations containing bbox
            results[f'gt_labels{i}'] = copy.deepcopy(results['gt_labels']) # TODO: allow to geometric operations containing bbox
            results['img_fields'].append(f'img{i}')
            results['bbox_fields'].append(f'gt_bboxes{i}')

            results['custom_field'].append(f'gt_bboxes{i}')
            results['custom_field'].append(f'gt_labels{i}')

        return results

    def multiaug_and_mix(self, img_orig, gt_bboxes):
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
            img_aug = img_orig.copy()
            for j in range(len(self.aug_list)):
                # Sample operations : [op1, op2, op3] ~ O
                if isinstance(self.mixture_depth, list):
                    depth = np.random.randint(*self.mixture_depth[j])
                else:
                    depth = self.mixture_depth if self.mixture_depth > 0 else np.random.randint(1, 3)
                op_chain = np.random.choice(self.aug_list[j], depth, replace=False)  # not allow same aug if replace is False.

                # Augment
                if isinstance(self.aug_severity, list):
                    aug_severity = self.aug_severity[j]
                else:
                    aug_severity = self.aug_severity
                img_aug = self.chain(img_aug, op_chain, img_size, aug_severity, gt_bboxes=gt_bboxes)

            # Mixing
            img_aug = np.asarray(img_aug, dtype=np.float32)
            img_mix += mixing_weights[i] * img_aug

        img_augmix = (1-sample_weight) * img_orig + sample_weight * img_mix
        img_augmix = np.array(img_augmix, dtype=np.float32)
        return img_augmix

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
            if isinstance(self.mixture_depth, tuple):
                depth = np.random.randint(*self.mixture_depth)
            else:
                depth = self.mixture_depth if self.mixture_depth > 0 else np.random.randint(1, 4)
            op_chain = np.random.choice(self.aug_list, depth, replace=False) # not allow same aug if replace is False.

            # Augment
            img_aug = self.chain(img_orig.copy(), op_chain, img_size, self.aug_severity, gt_bboxes=gt_bboxes)

            # Mixing
            img_aug = np.asarray(img_aug, dtype=np.float32)
            img_mix += mixing_weights[i] * img_aug

        img_augmix = (1-sample_weight) * img_orig + sample_weight * img_mix
        img_augmix = np.array(img_augmix, dtype=np.float32)
        return img_augmix

    def chain(self, img, op_chain, img_size, aug_severity, gt_bboxes=None):
        '''
        img: np.array
        '''
        if isinstance(img, np.ndarray):
            img_aug = Image.fromarray(img, 'RGB') # pil_img
        elif isinstance(img, Image.Image):
            img_aug = img
        else:
            raise TypeError

        for op in op_chain:
            if isinstance(img_aug, np.ndarray):
                img_aug = Image.fromarray(img_aug, 'RGB')
            if self.geo_severity is not None:
                if op in GEO_OP_LIST:
                    aug_severity = self.geo_severity
            img_aug = op(img_aug, level=aug_severity,
                         img_size=img_size, bboxes_xy=gt_bboxes, **self.kwargs)

        return img_aug

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str



