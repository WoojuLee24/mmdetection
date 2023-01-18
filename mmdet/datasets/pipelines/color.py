# Copyright (c) OpenMMLab. All rights reserved.
import copy
import inspect
import math
import warnings

import cv2
import mmcv
import numpy as np
from numpy import random
import torch

from PIL import Image

from mmdet.core import PolygonMasks, find_inside_bboxes
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from ..builder import PIPELINES
from .compose import Compose
from .augmix import (autocontrast, equalize, posterize, solarize,
                     color, contrast, brightness, sharpness)


@PIPELINES.register_module()
class Color:
    def __init__(self, mean, std, aug_list='augs', to_rgb=True, no_jsd=False, aug_severity=1):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

        self.mixture_width = 3
        self.mixture_depth = -1

        self.aug_prob_coeff = 1.
        self.aug_severity = aug_severity

        self.no_jsd = no_jsd

        color_augs = [
            autocontrast, equalize, posterize, solarize,
        ]
        color_all = [ # WARN: don't use it if dataset-c
            autocontrast, equalize, posterize, solarize,
            color, contrast, brightness, sharpness
        ]
        if (aug_list == 'augs'):
            self.aug_list = color_augs
        elif aug_list == 'all':
            self.aug_list = color_all


    def __call__(self, results):

        if self.no_jsd:
            img = results['img'].copy()
            results['img'] = self.aug(img)
            return results
        else:
            img = results['img'].copy()
            results['img2'] = self.aug(img)
            results['img3'] = self.aug(img)
            results['img_fields'] = ['img', 'img2', 'img3']

            return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})'
        return repr_str

    def aug(self, img):
        IMAGE_HEIGHT, IMAGE_WIDTH, _ = img.shape
        img_size = (IMAGE_WIDTH, IMAGE_HEIGHT)

        # image_aug = img.copy()
        image_aug = Image.fromarray(img, "RGB")
        depth = self.mixture_depth if self.mixture_depth > 0 else np.random.randint(1, 4)
        for _ in range(depth):
            op = np.random.choice(self.aug_list)
            image_aug = op(image_aug, level=self.aug_severity, img_size=img_size)
        # Preprocessing commutes since all coefficients are convex
        image_aug = np.asarray(image_aug, dtype=np.float32)

        return image_aug
