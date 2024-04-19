# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import torchvision.transforms as TT

from scepter.studio.preprocess.processors.base_processor import \
    BaseImageProcessor

__all__ = ['CenterCrop', 'PaddingCrop']


class CenterCrop(BaseImageProcessor):
    def __init__(self, cfg, language='en'):
        super().__init__(cfg, language=language)

    def __call__(self, image, **kwargs):
        w, h = image.size
        height_ratio = kwargs.get('height_ratio', 1)
        width_ratio = kwargs.get('width_ratio', 1)

        output_h_align_height, output_h_align_width = h, int(h / height_ratio *
                                                             width_ratio)
        if output_h_align_height * output_h_align_width <= w * h:
            output_height, output_width = output_h_align_height, output_h_align_width
        else:
            output_height, output_width = int(w / width_ratio *
                                              height_ratio), w
        image = TT.Resize(max(output_height, output_width))(image)
        image = TT.CenterCrop((output_height, output_width))(image)
        return image


class PaddingCrop(BaseImageProcessor):
    def __init__(self, cfg, language='en'):
        super().__init__(cfg, language=language)

    def get_caption(self, image, **kwargs):
        return image
