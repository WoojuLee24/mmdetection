# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import numpy as np
from PIL import Image

from ..builder import PIPELINES

from .auto_augment import Translate, Shear
from .augmix_detection import bboxes_only_translate_y
from .transforms import CutOut


_MAX_LEVEL = 10


@PIPELINES.register_module()
class TranslateX(Translate):
    def __init__(self, level, prob=0.5,
                 img_fill_val=128, seg_ignore_label=255,
                 max_translate_offset=250.,
                 random_negative_prob=0.5,
                 min_size=0):
        super(TranslateX, self).__init__(level, prob,
                                         img_fill_val, seg_ignore_label,
                                         'horizontal',
                                         max_translate_offset,
                                         random_negative_prob,
                                         min_size)


@PIPELINES.register_module()
class BBoxOnlyTranslateY:
    def __init__(self, level, prob=0.5):
        assert isinstance(level, (int, float)), \
            'The level must be type int or float.'
        assert 0 <= level <= _MAX_LEVEL, \
            'The level used for calculating Translate\'s offset should be ' \
            'in range [0,_MAX_LEVEL]'
        assert 0 <= prob <= 1.0, \
            'The probability of translation should be in range [0, 1].'
        self.level = level
        self.prob = prob

    def __call__(self, results):
        if np.random.rand() > self.prob:
            return results

        img = results['img']
        img_height, img_width, _ = img.shape
        img_size = (img_width, img_height)

        if isinstance(img, np.ndarray):
            img_aug = Image.fromarray(img, 'RGB')
        elif isinstance(img, Image.Image):
            img_aug = img
        else:
            raise TypeError

        bboxes_xy = results['gt_bboxes']

        results['img'] = np.asarray(bboxes_only_translate_y(img_aug, bboxes_xy, self.level, img_size), img.dtype)
        return results


@PIPELINES.register_module()
class Cutout2:
    def __init__(self, level, prob=0.5,
                 cutout_const=100, fill_in=(0,0,0)):
        assert isinstance(level, (int, float)), \
            'The level must be type int or float.'
        assert 0 <= level <= _MAX_LEVEL, \
            'The level used for calculating Translate\'s offset should be ' \
            'in range [0,_MAX_LEVEL]'
        pad_size = int((level / _MAX_LEVEL) * cutout_const)
        cutout_ratio = (pad_size, pad_size)
        self.cutout = CutOut(n_holes=1, cutout_shape=None,
                             cutout_ratio=cutout_ratio,
                             fill_in=fill_in)
        assert 0 <= prob <= 1.0, \
            'The probability of translation should be in range [0, 1].'
        self.prob = prob

    def __call__(self, results):
        if np.random.rand() > self.prob:
            return results

        if not isinstance(results['img'], np.ndarray):
            results['img'] = np.asarray(results['img'], dtype=np.uint8)

        return self.cutout(results)


@PIPELINES.register_module()
class ShearY(Shear):
    def __init__(self,
                 level, prob=0.5,
                 img_fill_val=128,
                 seg_ignore_label=255,
                 max_shear_magnitude=0.3,
                 random_negative_prob=0.5,
                 interpolation='bilinear'):
        super(ShearY, self).__init__(level,
                                     img_fill_val=img_fill_val,
                                     seg_ignore_label=seg_ignore_label,
                                     prob=prob,
                                     direction='vertical',
                                     max_shear_magnitude=max_shear_magnitude,
                                     random_negative_prob=random_negative_prob,
                                     interpolation=interpolation)

