import copy

import numpy as np
# from numpy import random
import random

from ..builder import PIPELINES

import cv2

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

from PIL import Image, ImageDraw, ImageFilter
from mmdet.datasets.pipelines.augmix import (autocontrast, equalize, posterize, solarize, color,
                                             contrast, brightness, sharpness,
                                             rotate, shear_x, shear_y, translate_x, translate_y,)


# BBoxOnlyAugmentation
# REF: https://github.com/poodarchu/learn_aug_for_object_detection.numpy/
def _apply_bbox_only_augmentation(img, bbox_xy, aug_func, fillmode=None, fillcolor=None, return_bbox=False, radius=10,
                                  radius_ratio=None, margin=3, sigma_ratio=None, times=3, blur_bbox=None, **kwargs):
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
    if (x2-x1) < 1 or (y2-y1) < 1:
        return (np.asarray(img, dtype=np.uint8), bbox_xy) if return_bbox \
            else np.asarray(img, dtype=np.uint8)

    # bbox_content
    if fillmode is None:
        bbox_content = img[y1:y2 + 1, x1:x2 + 1, :]
        kwargs['img_size'] = (x2-x1+1, y2-y1+1)
    elif fillmode in ['img', 'blur', 'box_blur', 'box_blur_margin', 'gaussian_blur_margin', 'var_blur']:
        bbox_content = img
        kwargs['img_size'] = (img_width, img_height)
    else:
        raise TypeError

    center = None
    if fillmode in ['img', 'box_blur', 'box_blur_margin', 'blur_margin', 'blur', 'gaussian_blur_margin', 'var_blur']:
        center = ((x1 + x2) / 2., (y1 + y2) / 2.)
        kwargs['img_size_for_level'] = (x2-x1+1, y2-y1+1)

    # Augment
    outputs = aug_func(Image.fromarray(bbox_content), **kwargs, fillcolor=fillcolor, center=center,
                       bbox_xy=bbox_xy, return_bbox=return_bbox)

    if isinstance(outputs, dict):
        augmented_bbox_content = np.asarray(outputs['img'])
        augmented_gt_bbox = outputs['gt_bbox'] if 'gt_bbox' in outputs else bbox_xy
    else:
        augmented_bbox_content = np.asarray(outputs)
    if fillmode == 'img':
        augmented_bbox_content = augmented_bbox_content[y1:y2+1, x1:x2+1, :]

    # Pad with pad_width: [[before_1, after_1], [before_2, after_2], ..., [before_N, after_N]]
    if not fillmode in ['blur', 'blur_margin', 'box_blur', 'box_blur_margin', 'gaussian_blur_margin', 'var_blur']:
        pad_width = [[y1, max(0, img_height - y2 - 1)], [x1, max(0, img_width - x2 - 1)], [0, 0]]
        fill_value = 0 if fillcolor is None else fillcolor[0]
        augmented_bbox_content = np.pad(augmented_bbox_content, pad_width, 'constant', constant_values=fill_value)

    # get mask
    if blur_bbox is None:
        m = int(3*radius/2) if fillmode in ['box_blur_margin', 'blur_margin', 'gaussian_blur_margin'] else 0
        mask = np.zeros_like(img)
        mask[y1 + m: y2 - m + 1, x1 + m:x2 - m + 1, :] = 255

        # Blur
        if fillmode in ['blur', 'gaussian_blur_margin', 'var_blur']:
            if fillmode == 'var_blur':
                sigma_x, sigma_y = (x2 - x1) * sigma_ratio / 3 * 2, (y2 - y1) * sigma_ratio / 3 * 2
                mask = cv2.GaussianBlur(mask, (0, 0), sigma_x, sigmaY=sigma_y)
                mask = 255 - mask
            else:
                mask = cv2.GaussianBlur(mask, (0, 0), radius)
            if fillmode == 'gaussian_blur_margin':
                mask = 255 - mask
        elif fillmode in ['box_blur', 'box_blur_margin']:
            k = int(np.sqrt(12 * radius ** 2 / times + 1))
            for i in range(times):
                mask = cv2.blur(mask, (k, k))
            mask = 255 - mask
        elif fillmode == 'blur_margin':
            raise NotImplementedError
    else:
        mask = 255 - blur_bbox

    # Overwrite augmented_bbox_content into img
    img = img * (mask/255) + augmented_bbox_content * (1-mask/255)

    if return_bbox:
        return np.asarray(img, dtype=np.uint8), augmented_gt_bbox
    else:
        return np.asarray(img, dtype=np.uint8)


def _apply_bboxes_only_augmentation(img, bboxes_xy, aug_func, return_bbox=False, blur_bboxes=None, **kwargs):
    '''
    Args:
        img         : (np.array) (img_width, img_height, channel)
        bboxes_xy   : (tensor) has shape of (num_bboxes, 4) with [x1, y1, x2, y2]
        aug_func    : (func) The argument is bbox_content # TODO: severity?
    '''
    if return_bbox:
        new_bboxes_xy = np.zeros_like(bboxes_xy)
        for i in range(len(bboxes_xy)):
            img, new_bboxes_xy[i] = _apply_bbox_only_augmentation(img, bboxes_xy[i], aug_func, return_bbox=True, **kwargs)
        return Image.fromarray(img), new_bboxes_xy
    else:
        for i in range(len(bboxes_xy)):
            blur_bbox = None if blur_bboxes == None else blur_bboxes[i]
            img = _apply_bbox_only_augmentation(img, bboxes_xy[i], aug_func, blur_bbox=blur_bbox, **kwargs)
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


def random_bboxes_only_rotate(pil_img, bboxes_xy, level, img_size, num_bboxes, return_bbox=False, **kwargs):
    random_bboxes_xy = generate_random_bboxes_xy(img_size, num_bboxes, bboxes_xy, **kwargs)
    if 'blur_bboxes' in kwargs:
        kwargs['blur_bboxes'] = None
    return _apply_bboxes_only_augmentation(pil_img, random_bboxes_xy, rotate, level=level, img_size=img_size, return_bbox=False, **kwargs)


def random_bboxes_only_shear_x(pil_img, bboxes_xy, level, img_size, num_bboxes, return_bbox=False, **kwargs):
    random_bboxes_xy = generate_random_bboxes_xy(img_size, num_bboxes, bboxes_xy, return_bbox=False, **kwargs)
    if 'blur_bboxes' in kwargs:
        kwargs['blur_bboxes'] = None
    return _apply_bboxes_only_augmentation(pil_img, random_bboxes_xy, shear_x, level=level, img_size=img_size, return_bbox=False, **kwargs)


def random_bboxes_only_shear_y(pil_img, bboxes_xy, level, img_size, num_bboxes, return_bbox=False, **kwargs):
    random_bboxes_xy = generate_random_bboxes_xy(img_size, num_bboxes, bboxes_xy, return_bbox=False, **kwargs)
    if 'blur_bboxes' in kwargs:
        kwargs['blur_bboxes'] = None
    return _apply_bboxes_only_augmentation(pil_img, random_bboxes_xy, shear_y, level=level, img_size=img_size, return_bbox=False, **kwargs)


def random_bboxes_only_shear_xy(pil_img, bboxes_xy, level, img_size, num_bboxes, return_bbox=False, **kwargs):
    func = bboxes_only_shear_x if np.random.rand() < 0.5 else bboxes_only_shear_y
    if 'blur_bboxes' in kwargs:
        kwargs['blur_bboxes'] = None
    random_bboxes_xy = generate_random_bboxes_xy(img_size, num_bboxes, bboxes_xy, return_bbox=False, **kwargs)
    return func(pil_img, random_bboxes_xy, level, img_size, **kwargs)


def random_bboxes_only_translate_x(pil_img, bboxes_xy, level, img_size, num_bboxes, return_bbox=False, **kwargs):
    random_bboxes_xy = generate_random_bboxes_xy(img_size, num_bboxes, bboxes_xy, return_bbox=False, **kwargs)
    if 'blur_bboxes' in kwargs:
        kwargs['blur_bboxes'] = None
    return _apply_bboxes_only_augmentation(pil_img, random_bboxes_xy, translate_x, level=level, img_size=img_size, return_bbox=False, **kwargs)


def random_bboxes_only_translate_y(pil_img, bboxes_xy, level, img_size, num_bboxes, return_bbox=False, **kwargs):
    random_bboxes_xy = generate_random_bboxes_xy(img_size, num_bboxes, bboxes_xy, return_bbox=False, **kwargs)
    if 'blur_bboxes' in kwargs:
        kwargs['blur_bboxes'] = None
    return _apply_bboxes_only_augmentation(pil_img, random_bboxes_xy, translate_y, level=level, img_size=img_size, return_bbox=False, **kwargs)


def random_bboxes_only_translate_xy(pil_img, bboxes_xy, level, img_size, num_bboxes, return_bbox=False, **kwargs):
    func = bboxes_only_translate_x if np.random.rand() < 0.5 else bboxes_only_translate_y
    random_bboxes_xy = generate_random_bboxes_xy(img_size, num_bboxes, bboxes_xy, return_bbox=False, **kwargs)
    if 'blur_bboxes' in kwargs:
        kwargs['blur_bboxes'] = None
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
def _apply_bg_only_augmentation(img, bboxes_xy, aug_func, fillmode=None, fillcolor=0, return_bbox=False, radius=10,
                                radius_ratio=None, bg_margin=3, times=3, margin_bg=False, sigma_ratio=None, blur_bboxes=None, **kwargs):
    '''
    Args:
        img         : (np.array) (img_width, img_height, channel)
        bboxes_xy   : (tensor) has shape of (num_bboxes, 4) with [x1, y1, x2, y2]
        aug_func    : (func) The argument is bbox_content # TODO: severity?
    '''
    if not isinstance(img, np.ndarray):
        img = np.asarray(img)
    if margin_bg:
        if fillmode in ['blur_margin', 'box_blur_margin', 'gaussian_blur_margin']:
            pass
        elif fillmode in ['blur', 'box_blur', 'gaussian_blur']:
            fillmode = f"{fillmode}_margin"
        else:
            raise ValueError

    if fillmode in ['blur_margin', 'box_blur_margin', 'gaussian_blur_margin']:
        h, w = img.shape[0], img.shape[1]
        if fillmode == 'blur_margin':
            m = int(radius * bg_margin / 2)
        elif fillmode in ['box_blur_margin', 'gaussian_blur_margin']:
            m = int(3 * radius / 2)

    # Make the union of bboxes
    mask = np.zeros_like(img)
    fill_img = np.zeros_like(img, dtype=np.uint8)
    fill_img[:] = fillcolor
    expanded_mask = np.zeros_like(img)
    if blur_bboxes is None:
        for i in range(len(bboxes_xy)):
            bbox_xy = bboxes_xy[i]
            x1, y1, x2, y2 = int(bbox_xy[0]), int(bbox_xy[1]), int(bbox_xy[2]), int(bbox_xy[3])
            if fillmode == 'blur_margin':
                gaussian_box = kwargs['gaussian_box']
                if (x2-x1) < 1 or (y2-y1) < 1:
                    continue
                m_x, m_y = int((x2-x1)*radius_ratio), int((y2-y1)*radius_ratio)
                resize_w, resize_h = x2 - x1 + m_x * 2, y2 - y1 + m_y * 2
                resized_blur_box = cv2.resize(gaussian_box, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
                resized_blur_box = np.asarray(resized_blur_box, dtype=np.uint8)
                before_blur_mask = mask[max(0, y1-m_y): min(h, y2+m_y), max(0, x1-m_x): min(w, x2+m_x), :]
                resized_blur_box = resized_blur_box[max(m_y - y1, 0):min(h + m_y - y1, resize_h), max(m_x - x1, 0):min(w + m_x - x1, resize_w), :]
                mask[max(0, y1-m_y): min(h, y2+m_y), max(0, x1-m_x): min(w, x2+m_x), :] = np.maximum(before_blur_mask, resized_blur_box)
                # mask[max(0, y1-m_y): min(h, y2+m_y+1), max(0, x1-m_x): min(w, x2+m_x+1), :] = 255
                expanded_mask[max(0, y1-m): min(h, y2+m+1), max(0, x1-m): min(w, x2+m+1), :] = 255
            elif fillmode in ['box_blur_margin', 'gaussian_blur_margin']:
                if (x2 - x1) < 1 or (y2 - y1) < 1:
                    continue
                mask[max(0, y1-m): min(h, y2+m+1), max(0, x1-m): min(w, x2+m+1), :] = 255
                expanded_mask[max(0, y1-m): min(h, y2+m+1), max(0, x1-m): min(w, x2+m+1), :] = 255
            else:
                mask[y1:y2+1, x1:x2+1, :] = 255
                expanded_mask[y1:y2 + 1, x1:x2 + 1, :] = 255
    else:
        for i in range(len(blur_bboxes)):
            mask = np.maximum(mask, blur_bboxes[i])

    if fillmode is None:
        bbox_content = (mask / 255) * fill_img + (1.0 - mask / 255) * img
    elif fillmode == 'img':
        bbox_content = (expanded_mask / 255) * fill_img + (1.0 - expanded_mask / 255) * img
    elif fillmode in ['blur', 'blur_margin', 'box_blur', 'box_blur_margin', 'gaussian_blur_margin', 'var_blur']:
        bbox_content = img
    else:
        raise TypeError

    # Augment
    bbox_content = Image.fromarray(bbox_content)
    kwargs['img_size'] = bbox_content.size

    # Overwrite augmented_bbox_content into img
    if fillmode is None:
        outputs = aug_func(bbox_content, **kwargs, fillcolor=fillcolor)
        augmented_bbox_content = outputs['img'] if isinstance(outputs, dict) else outputs
        img = (mask/255) * img + (1.0 - mask/255) * augmented_bbox_content
    elif fillmode in ['img', 'blur', 'blur_margin', 'box_blur', 'box_blur_margin', 'gaussian_blur_margin', 'var_blur']:
        outputs = aug_func(bbox_content, return_bbox=False, **kwargs, fillcolor=fillcolor, mask=Image.fromarray(mask))
        if isinstance(outputs, dict):
            augmented_bbox_content = outputs['img']
            augmented_mask = outputs['mask']
        else:
            (augmented_bbox_content, augmented_mask) = outputs
        if fillmode == 'var_blur':
            maintained_mask = np.maximum(mask, augmented_mask)
        else:
            maintained_mask = (mask / 255) * mask + (1 - mask / 255) * augmented_mask

        if blur_bboxes is None:
            if fillmode in ['blur', 'blur_margin', 'gaussian_blur_margin', 'var_blur']:
                if fillmode == 'var_blur':
                    maintained_mask = cv2.GaussianBlur(maintained_mask, (0, 0), sigma)
                else:
                    maintained_mask = cv2.GaussianBlur(maintained_mask, (0,0), radius)
            elif fillmode in ['box_blur', 'box_blur_margin']:
                k = int(np.sqrt(12*radius**2/times+1))
                for i in range(times):
                    maintained_mask = cv2.blur(maintained_mask, (k, k))
        img = (maintained_mask/255) * img + (1 - maintained_mask/255) * augmented_bbox_content

    return np.asarray(img, dtype=np.uint8)


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
    elif version in ['1.4', '1.4.1', '1.4.2', '1.4.3', '1.4.4',
                     '1.4.4.1', '1.4.4.2', '1.4.4.3', '1.4.4.1.1', '1.4.4.1.2',
                     '2.1']:
        aug_list = [autocontrast, equalize, posterize, solarize,  # color
                    bg_only_rotate, bg_only_shear_xy, bg_only_translate_xy,  # bg only transformation
                    bboxes_only_rotate, bboxes_only_shear_xy, bboxes_only_translate_xy]  # bbox only transformation
        return aug_list
    elif version in ['1.4.5', '1.4.5.1', '1.4.5.2', '1.4.5.1.1', '1.4.5.1.2', '1.4.5.1.3', '1.4.5.1.4',
                     '2.2']:
        aug_list = [autocontrast, equalize, posterize, solarize,  # color
                    bg_only_rotate, bg_only_shear_xy, bg_only_translate_xy,  # bg only transformation
                    random_bboxes_only_rotate, random_bboxes_only_shear_xy, random_bboxes_only_translate_xy, # random_bboxes only transformation
                    bboxes_only_rotate, bboxes_only_shear_xy, bboxes_only_translate_xy]  # bbox only transformation
        return aug_list
    elif version in ['1.5', '1.5.0', '1.5.1', '1.5.2', '1.5.3', '1.5.4', '1.5.5', '1.5.6', '1.5.7',
                     '1.5.1.1', '1.5.7.1', '1.5.1.2', '1.5.1.3',
                     '1.5.1.2.1', '1.5.1.2.2', '1.5.1.2.3', '1.5.1.2.4',
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
    elif version in ['1.9', '1.9.1', '1.9.2']:
        policy1 = [
            autocontrast, equalize, posterize, solarize,
            bboxes_only_rotate, bboxes_only_shear_xy, bboxes_only_translate_xy]
        policy2 = [
            autocontrast, equalize, posterize, solarize,
            bg_only_rotate, bg_only_shear_xy, bg_only_translate_xy
        ]
        policy3 = [
            autocontrast, equalize, posterize, solarize,
            random_bboxes_only_rotate, random_bboxes_only_shear_xy, random_bboxes_only_translate_xy,
        ]
        aug_list = dict(policies=[policy1, policy2, policy3])
        if version in ['1.9.1', '1.9.2']:
            aug_list['return_bbox_list'] = [True, False, False]
        return aug_list
    elif version in ['1.10', '1.10.1']:
        policy1 = [
            autocontrast, equalize, posterize, solarize,
            bboxes_only_rotate, bboxes_only_shear_x, bboxes_only_shear_y,
            bboxes_only_translate_x, bboxes_only_translate_y]
        policy2 = [
            autocontrast, equalize, posterize, solarize,
            bg_only_rotate, bg_only_shear_x, bg_only_shear_y,
            bg_only_translate_x, bg_only_translate_y
        ]
        policy3 = [
            autocontrast, equalize, posterize, solarize,
            random_bboxes_only_rotate, random_bboxes_only_shear_x, random_bboxes_only_shear_y,
            random_bboxes_only_translate_x, random_bboxes_only_translate_y
        ]
        aug_list = dict(policies=[policy1, policy2, policy3])
        if version in ['1.10.1']:
            aug_list['return_bbox_list'] = [True, False, False]
        return aug_list
    elif version in ['1.11', '1.11.1', '1.11.2', '1.11.3']:
        policy1 = [
            autocontrast, equalize, posterize, solarize,
            bboxes_only_rotate, bboxes_only_shear_xy, bboxes_only_translate_xy]
        policy2 = [
            autocontrast, equalize, posterize, solarize,
            bg_only_rotate, bg_only_shear_xy, bg_only_translate_xy,
            bboxes_only_rotate, bboxes_only_shear_xy, bboxes_only_translate_xy
        ]
        policy3 = [
            autocontrast, equalize, posterize, solarize,
            random_bboxes_only_rotate, random_bboxes_only_shear_xy, random_bboxes_only_translate_xy,
            bboxes_only_rotate, bboxes_only_shear_xy, bboxes_only_translate_xy
        ]
        aug_list = dict(policies=[policy1, policy2, policy3])
        if version in ['1.11.1', '1.11.3']:
            aug_list['return_bbox_list'] = [True, True, True]
        return aug_list
    elif version in ['1.12']:
        policy1 = [
            autocontrast, equalize, posterize, solarize,
            bboxes_only_rotate, bboxes_only_shear_xy, bboxes_only_translate_xy]
        policy2 = [
            autocontrast, equalize, posterize, solarize,
            random_bboxes_only_rotate, random_bboxes_only_shear_xy, random_bboxes_only_translate_xy,
            bboxes_only_rotate, bboxes_only_shear_xy, bboxes_only_translate_xy
        ]
        policy3 = [
            autocontrast, equalize, posterize, solarize,
            random_bboxes_only_rotate, random_bboxes_only_shear_xy, random_bboxes_only_translate_xy,
            bboxes_only_rotate, bboxes_only_shear_xy, bboxes_only_translate_xy
        ]
        aug_list = dict(policies=[policy1, policy2, policy3])
        return aug_list
    elif version in ['999']:
        return [bboxes_only_rotate]
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
class AugMixDetectionFaster:
    def __init__(self, mean, std,
                 num_views=3,
                 version='0.1',
                 aug_severity=6,
                 mixture_depth=-1,
                 geo_severity=None,
                 to_rgb=True,
                 return_bbox=False,
                 pre_blur=False,
                 **kwargs):
        super(AugMixDetectionFaster, self).__init__()
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
        self.return_bbox = return_bbox
        self.pre_blur = pre_blur
        self.kwargs = kwargs


    def __call__(self, results, *args, **kwargs):
        if self.num_views == 1:
            if isinstance(self.aug_list, dict):
                return_bbox_list = self.aug_list['return_bbox_list'] if 'return_bbox_list' in self.aug_list else [False] * len(self.aug_list['policies'])
                outputs = self.aug_and_mix_with_policy(results['img'], results['gt_bboxes'], return_bbox_list=return_bbox_list)
            else:
                outputs = self.aug_and_mix(results['img'], results['gt_bboxes'], return_bbox=self.return_bbox, pre_blur=self.pre_blur)
            results['img'] = np.asarray(outputs['img'], dtype=results['img'].dtype)
            if 'gt_bboxes' in outputs:
                results['gt_bboxes'] = outputs['gt_bboxes']
            return results

        results['custom_field'] = []
        for i in range(2, self.num_views+1):
            gt_bboxes_augmix = None
            if isinstance(self.aug_list, tuple):
                img_augmix = self.multiaug_and_mix(results['img'].copy(), results['gt_bboxes'])
            elif isinstance(self.aug_list, dict):
                outputs = self.aug_and_mix_with_policy(results['img'].copy(), results['gt_bboxes'])
                img_augmix = outputs['img']
                gt_bboxes_augmix = outputs['gt_bboxes'] if 'gt_bboxes' in outputs else gt_bboxes_augmix
            else:
                outputs = self.aug_and_mix(results['img'].copy(), results['gt_bboxes'], pre_blur=self.pre_blur)
                img_augmix = outputs['img']
                gt_bboxes_augmix = outputs['gt_bboxes'] if 'gt_bboxes' in outputs else gt_bboxes_augmix
            results[f'img{i}'] = np.array(img_augmix, dtype=results['img'].dtype)
            results[f'gt_bboxes{i}'] = copy.deepcopy(results['gt_bboxes']) if gt_bboxes_augmix is None else gt_bboxes_augmix # TODO: allow to geometric operations containing bbox
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

    def aug_and_mix_with_policy(self, img_orig, gt_bboxes, return_bbox_list=False):
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

        policies = self.aug_list['policies']
        if return_bbox_list == False:
            return_bbox_list = [False] * len(policies)
        gt_bboxes_aug = np.zeros_like(gt_bboxes)
        assert len(policies) == self.mixture_width

        for i in range(self.mixture_width):
            # Sample operations : [op1, op2, op3] ~ O
            if isinstance(self.mixture_depth, tuple):
                depth = np.random.randint(*self.mixture_depth)
            else:
                depth = self.mixture_depth if self.mixture_depth > 0 else np.random.randint(1, 4)
            op_chain = np.random.choice(policies[i], depth, replace=False)  # not allow same aug if replace is False.

            # Augment
            if return_bbox_list[i]:
                img_aug, new_gt_bboxes = self.chain(img_orig.copy(), op_chain, img_size, self.aug_severity,
                                                    gt_bboxes=gt_bboxes.copy(), return_bbox=True)
                gt_bboxes_aug += mixing_weights[i] * new_gt_bboxes
            else:
                img_aug = self.chain(img_orig.copy(), op_chain, img_size, self.aug_severity, gt_bboxes=gt_bboxes)
                gt_bboxes_aug += mixing_weights[i] * gt_bboxes

            # Mixing
            img_aug = np.asarray(img_aug, dtype=np.float32)
            img_mix += mixing_weights[i] * img_aug

        img_augmix = (1 - sample_weight) * img_orig + sample_weight * img_mix
        img_augmix = np.array(img_augmix, dtype=np.float32)

        outputs = dict(img=img_augmix)
        if any(return_bbox_list):
            gt_bboxes_aug = (1 - sample_weight) * gt_bboxes + sample_weight * gt_bboxes_aug
            outputs['gt_bboxes'] = gt_bboxes_aug
        return outputs


    def aug_and_mix(self, img_orig, gt_bboxes, return_bbox=False, pre_blur=False):
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
        gt_bboxes_aug = np.zeros_like(gt_bboxes)

        blur_bboxes = None
        if pre_blur:
            blur_bboxes = []
            for i in range(len(gt_bboxes)):
                bbox_xy = gt_bboxes[i]
                x1, y1, x2, y2 = int(bbox_xy[0]), int(bbox_xy[1]), int(bbox_xy[2]), int(bbox_xy[3])
                mask = np.zeros_like(img_orig)
                mask[y1: y2 + 1, x1:x2 + 1, :] = 255
                sigma_ratio = self.kwargs['sigma_ratio']
                sigma_x, sigma_y = (x2-x1)*sigma_ratio/3*2, (y2-y1)*sigma_ratio/3*2
                mask = cv2.GaussianBlur(mask, (0, 0), sigma_x, sigmaY=sigma_y)
                blur_bboxes.append(mask)

        for i in range(self.mixture_width):
            # Sample operations : [op1, op2, op3] ~ O
            if isinstance(self.mixture_depth, tuple):
                depth = np.random.randint(*self.mixture_depth)
            else:
                depth = self.mixture_depth if self.mixture_depth > 0 else np.random.randint(1, 4)
            op_chain = np.random.choice(self.aug_list, depth, replace=False) # not allow same aug if replace is False.

            # Augment
            if return_bbox:
                img_aug, new_gt_bboxes = self.chain(img_orig.copy(), op_chain, img_size, self.aug_severity, gt_bboxes=gt_bboxes.copy(), return_bbox=True)
                gt_bboxes_aug += mixing_weights[i] * new_gt_bboxes
            else:
                img_aug = self.chain(img_orig.copy(), op_chain, img_size, self.aug_severity, gt_bboxes=gt_bboxes, return_bbox=False, blur_bboxes=blur_bboxes)
                gt_bboxes_aug += mixing_weights[i] * gt_bboxes

            # Mixing
            img_aug = np.asarray(img_aug, dtype=np.float32)
            img_mix += mixing_weights[i] * img_aug

        img_augmix = (1-sample_weight) * img_orig + sample_weight * img_mix
        img_augmix = np.array(img_augmix, dtype=np.float32)
        outputs = dict(img=img_augmix)
        if return_bbox:
            gt_bboxes_aug = (1 - sample_weight) * gt_bboxes + sample_weight * gt_bboxes_aug
            outputs['gt_bboxes'] = gt_bboxes_aug
        return outputs

    def chain(self, img, op_chain, img_size, aug_severity, gt_bboxes=None, return_bbox=False, blur_bboxes=None):
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
            if return_bbox:
                outputs = op(img_aug, level=aug_severity,
                             img_size=img_size, bboxes_xy=gt_bboxes, return_bbox=return_bbox, **self.kwargs)
                if isinstance(outputs, dict):
                    img_aug = outputs['img']
                    gt_bboxes = outputs['gt_bbox']
                elif isinstance(outputs, tuple):
                    img_aug = outputs[0]
                    gt_bboxes = outputs[1]
                else:
                    img_aug = outputs
            else:
                img_aug = op(img_aug, level=aug_severity,
                             img_size=img_size, bboxes_xy=gt_bboxes, return_bbox=return_bbox, blur_bboxes=blur_bboxes, **self.kwargs)

        if return_bbox:
            return img_aug, gt_bboxes
        else:
            return img_aug

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str



