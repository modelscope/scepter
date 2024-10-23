# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import numbers

import numpy as np
import torch
import torchvision.transforms as TT
from PIL import Image
from scepter.modules.annotator.registry import ANNOTATORS
from scepter.modules.utils.distribute import we
from scepter.studio.preprocess.processors.base_processor import \
    BaseImageProcessor

__all__ = [
    'CenterCrop', 'ChangeSample', 'MaskEditSample', 'SwapSample',
    'CannyExtractor', 'ColorExtractor', 'InfoDrawContourExtractor',
    'DegradationExtractor', 'MidasExtractor', 'DoodleExtractor',
    'GrayExtractor', 'InpaintingExtractor', 'OpenposeExtractor',
    'OutpaintingExtractor', 'InfoDrawContourExtractor', 'ESAMExtractor',
    'InvertExtractor', 'DefaultMaskSample', 'MaskSwapEditSample',
    'OutpaintingResize', 'SwapMaskSwapEditSample', 'SourceMaskSample',
    'InpaintingSourceExtractor', 'LamaExtractor'
]


class CenterCrop(BaseImageProcessor):
    def __init__(self, cfg, language='en'):
        super().__init__(cfg, language=language)

    def process(self, image, height_ratio, width_ratio):
        if isinstance(image, dict):
            image = image['background']
        w, h = image.size
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

    def __call__(self, **kwargs):
        use_preview = kwargs.get('use_preview', True)
        target_image = kwargs.get('target_image', None)
        src_mask = kwargs.get('src_mask', None)
        src_image = kwargs.get('src_image', None)
        target_image = kwargs.get('preview_target_image',
                                  None) if use_preview else target_image
        src_mask = kwargs.get('preview_src_mask',
                              None) if use_preview else src_mask
        src_image = kwargs.get('preview_src_image',
                               None) if use_preview else src_image
        height_ratio = kwargs.get('height_ratio', 1)
        width_ratio = kwargs.get('width_ratio', 1)
        ret_data = {}
        if target_image is not None:
            target_image = self.process(target_image, height_ratio,
                                        width_ratio)
            ret_data['target_image'] = target_image
        if src_image is not None:
            src_image = self.process(src_image, height_ratio, width_ratio)
            ret_data['src_image'] = src_image
        if src_mask is not None:
            src_mask = self.process(src_mask, height_ratio, width_ratio)
            ret_data['src_mask'] = src_mask
        return ret_data


class ChangeSample(BaseImageProcessor):
    def __init__(self, cfg, language='en'):
        super().__init__(cfg, language=language)

    def __call__(self, **kwargs):
        # replace file with preview data
        preview_target_image = kwargs.get('preview_target_image', None)
        preview_src_mask = kwargs.get('preview_src_mask', None)
        preview_src_image = kwargs.get('preview_src_image', None)
        preview_caption = kwargs.get('preview_caption', None)
        ret_data = {}
        if preview_target_image is not None:
            ret_data['target_image'] = preview_target_image['composite']
        if preview_src_image is not None:
            ret_data['src_image'] = preview_src_image['background']
        if preview_src_mask is not None:
            ret_data['src_mask'] = preview_src_mask['layers'][0]
        if preview_caption is not None:
            ret_data['caption'] = preview_caption
        return ret_data


class SourceMaskSample(BaseImageProcessor):
    def __init__(self, cfg, language='en'):
        super().__init__(cfg, language=language)

    def __call__(self, **kwargs):
        # replace file with preview data
        # target_image = kwargs.get('target_image', None)
        # src_mask = kwargs.get('src_mask', None)
        # src_image = kwargs.get('src_image', None)
        use_preview = kwargs.get('use_preview', True)
        caption = kwargs.get('caption', None)
        preview_target_image = kwargs.get('preview_target_image', None)
        preview_src_mask = kwargs.get('preview_src_mask', None)
        preview_src_image = kwargs.get('preview_src_image', None)
        preview_caption = kwargs.get('preview_caption', None) if use_preview else caption
        ret_data = {}
        if preview_src_image is not None:
            if isinstance(preview_src_image, dict):
                prev_src_image = preview_src_image['image']
                prev_src_mask = preview_src_image['mask']
            else:
                prev_src_image = preview_src_image
                prev_src_mask = preview_src_mask
        else:
            prev_src_image = None
            prev_src_mask = None

        if isinstance(preview_target_image, dict):
            prev_target_image = preview_target_image['image']
        else:
            prev_target_image = preview_target_image
        # process src image and mask according to mask
        ret_data['src_image'] = None
        if prev_src_image is not None:
            w, h = prev_src_image.size
            if prev_src_mask is None:
                ret_data['src_mask'] = Image.new('L', (w, h), 0)
            else:
                prev_src_mask = prev_src_mask.convert('L')
                ow, oh = prev_src_mask.size
                if ow == w and oh == h:
                    ret_data['src_mask'] = prev_src_mask
                else:
                    ret_data['src_mask'] = Image.new('L', (ow, oh), 0)
            # process target image: target_image * mask + src_image * (1 - mask)
            if prev_target_image is not None:
                ret_data['target_image'] = prev_target_image
            else:
                ret_data['target_image'] = Image.new('RGB', (w, h), (0, 0, 0))
        if caption is not None:
            ret_data['caption'] = preview_caption
        return ret_data

class MaskEditSample(BaseImageProcessor):
    def __init__(self, cfg, language='en'):
        super().__init__(cfg, language=language)

    def __call__(self, **kwargs):
        # replace file with preview data
        # target_image = kwargs.get('target_image', None)
        # src_mask = kwargs.get('src_mask', None)
        # src_image = kwargs.get('src_image', None)
        use_preview = kwargs.get('use_preview', True)
        caption = kwargs.get('caption', None)
        preview_target_image = kwargs.get('preview_target_image', None)
        preview_src_mask = kwargs.get('preview_src_mask', None)
        preview_src_image = kwargs.get('preview_src_image', None)
        preview_caption = kwargs.get('preview_caption',
                                     None) if use_preview else caption
        ret_data = {}
        if preview_src_image is not None:
            if isinstance(preview_src_image, dict):
                prev_src_image = preview_src_image['background']
                prev_src_mask = preview_src_image['layers'][0].split(
                                                    )[-1].convert('L')
            else:
                prev_src_image = preview_src_image
                prev_src_mask = preview_src_mask
        else:
            prev_src_image = None
            prev_src_mask = None

        if isinstance(preview_target_image, dict):
            prev_target_image = preview_target_image['background']
        else:
            prev_target_image = preview_target_image
        # process src image and mask according to mask
        ret_data['src_image'] = None
        if prev_src_image is not None:
            w, h = prev_src_image.size
            if prev_src_mask is None:
                ret_data['src_mask'] = Image.new('L', (w, h), 0)
            else:
                prev_src_mask = prev_src_mask
                ow, oh = prev_src_mask.size
                if ow == w and oh == h:
                    ret_data['src_mask'] = prev_src_mask
                else:
                    ret_data['src_mask'] = Image.new('L', (ow, oh), 0)
            # process target image: target_image * mask + src_image * (1 - mask)
            if prev_target_image is not None:
                prev_target_image = prev_target_image.resize((w, h))
                ret_data['target_image'] = Image.composite(
                    prev_target_image, prev_src_image, prev_src_mask)
            else:
                ret_data['target_image'] = Image.new('RGB', (w, h), (0, 0, 0))
        if caption is not None:
            ret_data['caption'] = preview_caption
        return ret_data


class SwapMaskSwapEditSample(BaseImageProcessor):
    def __init__(self, cfg, language='en'):
        super().__init__(cfg, language=language)

    def __call__(self, **kwargs):
        # replace file with preview data
        target_image = kwargs.get('target_image', None)
        src_mask = kwargs.get('src_mask', None)
        src_image = kwargs.get('src_image', None)
        caption = kwargs.get('caption', None)
        preview_target_image = kwargs.get('preview_target_image', None)
        preview_src_mask = kwargs.get('preview_src_mask', None)
        preview_src_image = kwargs.get('preview_src_image', None)
        preview_caption = kwargs.get('preview_caption', None)

        ret_data = {}
        if preview_src_image is not None:
            if isinstance(preview_src_image, dict):
                prev_src_image = preview_src_image['background']
                prev_src_mask = preview_src_image['layers'][0].split(
                                                    )[-1].convert('L')
            else:
                prev_src_image = preview_src_image
                prev_src_mask = preview_src_mask
        else:
            prev_src_image = None
            prev_src_mask = None

        if isinstance(preview_target_image, dict):
            prev_target_image = preview_target_image['background']
        else:
            prev_target_image = preview_target_image
        # process src image and mask according to mask
        ret_data['target_image'] = prev_target_image
        if prev_src_image is not None:
            w, h = prev_src_image.size
            if prev_src_mask is None:
                ret_data['src_mask'] = Image.new('L', (w, h), 0)
            else:
                prev_src_mask = prev_src_mask
                ow, oh = prev_src_mask.size
                if ow == w and oh == h:
                    ret_data['src_mask'] = prev_src_mask
                else:
                    ret_data['src_mask'] = Image.new('L', (ow, oh), 0)
            prev_src_mask_alter = 255 - np.array(prev_src_mask)
            prev_src_mask_alter = Image.fromarray(
                prev_src_mask_alter.astype(np.uint8))
            # process target image: target_image * mask + src_image * (1 - mask)
            if prev_target_image is not None:
                prev_target_image = prev_target_image.resize((w, h))
                ret_data['src_image'] = Image.composite(
                    prev_target_image, prev_src_image, prev_src_mask_alter)
            else:
                ret_data['src_image'] = Image.new('RGB', (w, h), (0, 0, 0))
        if caption is not None:
            ret_data['caption'] = preview_caption
        return ret_data


class MaskSwapEditSample(BaseImageProcessor):
    def __init__(self, cfg, language='en'):
        super().__init__(cfg, language=language)

    def __call__(self, **kwargs):
        # replace file with preview data
        target_image = kwargs.get('target_image', None)
        src_mask = kwargs.get('src_mask', None)
        src_image = kwargs.get('src_image', None)
        caption = kwargs.get('caption', None)
        preview_target_image = kwargs.get('preview_target_image', None)
        preview_src_mask = kwargs.get('preview_src_mask', None)
        preview_src_image = kwargs.get('preview_src_image', None)
        preview_caption = kwargs.get('preview_caption', None)

        ret_data = {}
        if preview_src_image is not None:
            if isinstance(preview_src_image, dict):
                prev_src_image = preview_src_image['background']
                prev_src_mask = preview_src_image['layers'][0].split(
                                                    )[-1].convert('L')
            else:
                prev_src_image = preview_src_image
                prev_src_mask = preview_src_mask
        else:
            prev_src_image = None
            prev_src_mask = None

        if isinstance(preview_target_image, dict):
            prev_target_image = preview_target_image['background']
        else:
            prev_target_image = preview_target_image
        # process src image and mask according to mask
        ret_data['target_image'] = prev_src_image
        if prev_src_image is not None:
            w, h = prev_src_image.size
            if prev_src_mask is None:
                ret_data['src_mask'] = Image.new('L', (w, h), 0)
            else:
                prev_src_mask = prev_src_mask
                ow, oh = prev_src_mask.size
                if ow == w and oh == h:
                    ret_data['src_mask'] = prev_src_mask
                else:
                    ret_data['src_mask'] = Image.new('L', (ow, oh), 0)
            # process target image: target_image * mask + src_image * (1 - mask)
            if prev_target_image is not None:
                prev_target_image = prev_target_image.resize((w, h))
                ret_data['src_image'] = Image.composite(
                    prev_target_image, prev_src_image, prev_src_mask)
            else:
                ret_data['src_image'] = Image.new('RGB', (w, h), (0, 0, 0))
        if caption is not None:
            ret_data['caption'] = preview_caption
        return ret_data


class SwapSample(BaseImageProcessor):
    def __init__(self, cfg, language='en'):
        super().__init__(cfg, language=language)

    def __call__(self, **kwargs):
        # replace file with preview data
        target_image = kwargs.get('target_image', None)
        src_mask = kwargs.get('src_mask', None)
        src_image = kwargs.get('src_image', None)
        ret_data = {}
        if target_image is not None and src_image is not None:
            ret_data['target_image'] = src_image
            ret_data['src_mask'] = src_mask
            ret_data['src_image'] = target_image
        return ret_data


class DefaultMaskSample(BaseImageProcessor):
    def __init__(self, cfg, language='en'):
        super().__init__(cfg, language=language)

    def __call__(self, **kwargs):
        use_preview = kwargs.get('use_preview', True)
        target_image = kwargs.get('target_image', None)
        src_mask = kwargs.get('src_mask', None)
        src_image = kwargs.get('src_image', None)
        preview_target_image = kwargs.get(
            'preview_target_image', None) if use_preview else target_image
        preview_src_mask = kwargs.get('preview_src_mask',
                                      None) if use_preview else src_mask
        preview_src_image = kwargs.get('preview_src_image',
                                       None) if use_preview else src_image
        if preview_src_image is not None:
            if isinstance(preview_src_image, dict):
                prev_src_image = preview_src_image['background']
                prev_src_mask = preview_src_image['layers'][0].split(
                                                    )[-1].convert('L')
            else:
                prev_src_image = preview_src_image
                prev_src_mask = preview_src_mask
        else:
            prev_src_image = None
            prev_src_mask = None

        ret_data = {}
        default_src_mask = np.zeros_like(np.array(prev_src_mask))
        ret_data['src_image'] = prev_src_image
        ret_data['src_mask'] = Image.fromarray(
            default_src_mask.astype(np.uint8))
        ret_data['target_image'] = preview_target_image
        return ret_data


class PaddingCrop(BaseImageProcessor):
    def __init__(self, cfg, language='en'):
        super().__init__(cfg, language=language)

    def get_caption(self, image, **kwargs):
        return image


class BaseExtractor(BaseImageProcessor):
    def __init__(self, cfg, language='en'):
        super().__init__(cfg, language=language)
        self.model = cfg.get("MODEL", None)
        self.model_info = {'device': 'offline', 'model': None}

    def unload_model(self):
        if self.use_device.lower() == 'gpu':
            flag, msg = self.unload_model_gpu()
        else:
            flag, msg = self.unload_model_cpu()
        return flag, msg

    def unload_model_cpu(self):
        super().unload_model()
        self.model_info['device'] = 'offline'
        del self.model_info['model']
        self.model_info['model'] = None
        return True, ''

    def unload_model_gpu(self):
        allow_load, msg = self.unload_model_cpu()
        if self.delete_instance:
            self.model_info['device'] = 'offline'
            if self.model_info['model'] is not None:
                self.model_info['model'] = self.model_info['model'].to('cpu')
                del self.model_info['model']
            self.model_info['model'] = None
        elif (isinstance(self.model_info['device'], numbers.Number)
              or str(self.model_info['device']).startswith('cuda')):
            self.model_info['device'] = 'cpu'
            self.model_info['model'] = self.model_info['model'].to('cpu')
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        return True, ''

    def load_model(self):
        allow_load, msg = super().load_model()
        if not allow_load:
            return allow_load, msg
        if self.use_device.lower() == 'gpu':
            flag, msg = self.load_model_gpu()
        else:
            flag, msg = self.load_model_cpu()
        return flag, msg

    def load_model_cpu(self):
        if self.model_info['device'] == 'offline':
            self.model_info['device'] = 'cpu'
            self.model_info['model'] = ANNOTATORS.build(self.model)
        elif self.model_info['device'] == 'cpu':
            pass
        return True, ''

    def load_model_gpu(self):
        if self.model_info['device'] == 'offline':
            self.model_info['device'] = 'cpu'
            self.model_info['model'] = ANNOTATORS.build(self.model).to(
                we.device_id)
        elif self.model_info['device'] == 'cpu':
            self.model_info['device'] = we.device_id
            self.model_info['model'] = self.model_info['model'].to(
                we.device_id)
        return True, ''

    def model_inference(self, model, image, mask=None, **kwargs):
        image = np.array(image)
        return model(image)

    def __call__(self, **kwargs):
        target_image = kwargs.get('target_image', None)
        src_mask = kwargs.get('src_mask', None)
        src_image = kwargs.get('src_image', None)
        use_preview = kwargs.get('use_preview', True)
        caption = kwargs.get('caption', None)
        preview_target_image = kwargs.get(
            'preview_target_image', None) if use_preview else target_image
        preview_src_mask = kwargs.get('preview_src_mask',
                                      None) if use_preview else src_mask
        preview_src_image = kwargs.get('preview_src_image',
                                       None) if use_preview else src_image
        preview_caption = kwargs.get('preview_caption',
                                     None) if use_preview else caption
        ret_data = {}
        if preview_src_image is not None:
            if isinstance(preview_src_image, dict):
                prev_src_image = preview_src_image['background'].convert('RGB')
                prev_src_mask = preview_src_image['layers'][0].split(
                                                    )[-1].convert('L')
            else:
                prev_src_image = preview_src_image
                prev_src_mask = preview_src_mask
        else:
            prev_src_image = None
            prev_src_mask = None
        # process src image and mask according to mask
        ret_data['src_image'] = None
        if prev_src_image is not None:
            w, h = prev_src_image.size
            if prev_src_mask is None:
                ret_data['src_mask'] = Image.new('L', (w, h), 0)
                prev_src_mask = ret_data['src_mask']
            else:
                prev_src_mask = prev_src_mask
                ow, oh = prev_src_mask.size
                if ow == w and oh == h:
                    ret_data['src_mask'] = prev_src_mask
                else:
                    ret_data['src_mask'] = Image.new('L', (ow, oh), 0)
            preview_target_image_np = self.model_inference(
                self.model_info['model'], prev_src_image, prev_src_mask)
            if len(preview_target_image_np.shape) > 3:
                preview_target_image = Image.fromarray(
                    preview_target_image_np[0])
            else:
                preview_target_image = Image.fromarray(preview_target_image_np)
            # process target image: target_image * mask + src_image * (1 - mask)
            if preview_target_image is not None and np.sum(
                    np.array(prev_src_mask)) > 0:
                preview_target_image = preview_target_image.resize((w, h))
                ret_data['target_image'] = Image.composite(
                    preview_target_image, prev_src_image, prev_src_mask)
            else:
                ret_data['target_image'] = preview_target_image
        elif preview_target_image is not None:
            preview_target_image = Image.fromarray(
                self.model_inference(self.model_info['model'],
                                     preview_target_image))
            ret_data['target_image'] = preview_target_image
        if caption is not None:
            ret_data['caption'] = preview_caption
        return ret_data


class CannyExtractor(BaseExtractor):
    pass


class ColorExtractor(BaseExtractor):
    pass


class InfoDrawContourExtractor(BaseExtractor):
    pass


class DegradationExtractor(BaseExtractor):
    pass


class MidasExtractor(BaseExtractor):
    pass


class DoodleExtractor(BaseExtractor):
    pass


class GrayExtractor(BaseExtractor):
    pass

class LamaExtractor(BaseExtractor):
    def model_inference(self, model, image, mask=None, **kwargs):
        image = np.array(image)
        mask = np.array(mask)
        return model(image, mask)

    def __call__(self, **kwargs):
        target_image = kwargs.get('target_image', None)
        src_mask = kwargs.get('src_mask', None)
        src_image = kwargs.get('src_image', None)
        use_preview = kwargs.get('use_preview', True)
        caption = kwargs.get('caption', None)
        preview_target_image = kwargs.get('preview_target_image', None)
        preview_src_mask = kwargs.get('preview_src_mask', None)
        preview_src_image = kwargs.get('preview_src_image', None)
        preview_caption = kwargs.get('preview_caption', None)
        ret_data = {}
        if preview_src_image is not None:
            if isinstance(preview_src_image, dict):
                prev_src_image = preview_src_image['image']
                prev_src_mask = preview_src_image['mask']
            else:
                prev_src_image = preview_src_image
                prev_src_mask = preview_src_mask
        else:
            prev_src_image = None
            prev_src_mask = None
        # process src image and mask according to mask
        ret_data['src_image'] = prev_src_image
        if prev_src_image is not None:
            w, h = prev_src_image.size
            if prev_src_mask is None:
                ret_data['src_mask'] = Image.new('L', (w, h), 0)
                prev_src_mask = ret_data['src_mask']
            else:
                prev_src_mask = prev_src_mask.convert('L')
                ow, oh = prev_src_mask.size
                if ow == w and oh == h:
                    ret_data['src_mask'] = prev_src_mask
                else:
                    ret_data['src_mask'] = Image.new('L', (ow, oh), 0)
            preview_target_image_np = self.model_inference(
                self.model_info['model'], prev_src_image, prev_src_mask)
            if len(preview_target_image_np.shape) > 3:
                preview_target_image = Image.fromarray(
                    preview_target_image_np[0])
            else:
                preview_target_image = Image.fromarray(preview_target_image_np)
            # process target image: target_image * mask + src_image * (1 - mask)
            ret_data['target_image'] = preview_target_image
        elif preview_target_image is not None:
            preview_target_image = Image.fromarray(
                self.model_inference(self.model_info['model'],
                                     preview_target_image))
            ret_data['target_image'] = preview_target_image
        if caption is not None:
            ret_data['caption'] = preview_caption
        return ret_data

class InpaintingExtractor(BaseExtractor):
    def model_inference(self, model, image, mask=None, **kwargs):
        image = np.array(image)
        mask = np.array(mask) if mask is not None else None
        return model(image, mask)

    def __call__(self, **kwargs):
        target_image = kwargs.get('target_image', None)
        src_mask = kwargs.get('src_mask', None)
        src_image = kwargs.get('src_image', None)
        use_preview = kwargs.get('use_preview', True)
        caption = kwargs.get('caption', None)
        preview_target_image = kwargs.get(
            'preview_target_image', None) if use_preview else target_image
        preview_src_mask = kwargs.get('preview_src_mask', None)
        preview_src_image = kwargs.get('preview_src_image', None)
        preview_caption = kwargs.get('preview_caption',
                                     None) if use_preview else caption
        ret_data = {}
        if preview_src_image is not None:
            if isinstance(preview_src_image, dict):
                prev_src_image = preview_src_image['background']
                prev_src_mask = preview_src_image['layers'][0].split(
                                                    )[-1].convert('L')
            else:
                prev_src_image = preview_src_image
                prev_src_mask = preview_src_mask
        else:
            prev_src_image = None
            prev_src_mask = None

        if preview_target_image is not None:
            if isinstance(preview_target_image, dict):
                prev_target_image = preview_target_image['background']
            else:
                prev_target_image = preview_target_image
        else:
            prev_target_image = preview_target_image

        # process src image and mask according to mask
        ret_data['src_image'] = None
        if prev_src_image is not None:
            w, h = prev_src_image.size
            if prev_src_mask is None:
                ret_data['src_mask'] = Image.new('L', (w, h), 0)
                prev_src_mask = ret_data['src_mask']
            else:
                prev_src_mask = prev_src_mask
                ow, oh = prev_src_mask.size
                if ow == w and oh == h:
                    ret_data['src_mask'] = prev_src_mask
                else:
                    ret_data['src_mask'] = Image.new('L', (ow, oh), 0)
            preview_src_image_np = self.model_inference(
                self.model_info['model'], prev_src_image, prev_src_mask)
            if len(preview_src_image_np.shape) > 3:
                preview_src_image = Image.fromarray(preview_src_image_np[0])
            else:
                preview_src_image = Image.fromarray(preview_src_image_np)
            # process target image: target_image * mask + src_image * (1 - mask)
            if preview_src_image is not None and np.sum(
                    np.array(prev_src_mask)) > 0:
                preview_src_image = preview_src_image.resize((w, h))
                ret_data['src_image'] = Image.composite(
                    preview_src_image, prev_src_image, prev_src_mask)
            else:
                ret_data['src_image'] = preview_src_image
        ret_data['target_image'] = prev_target_image
        if caption is not None:
            ret_data['caption'] = preview_caption
        return ret_data


class InpaintingSourceExtractor(BaseExtractor):
    def __call__(self, **kwargs):
        target_image = kwargs.get('target_image', None)
        src_mask = kwargs.get('src_mask', None)
        src_image = kwargs.get('src_image', None)
        use_preview = kwargs.get('use_preview', True)
        caption = kwargs.get('caption', None)
        preview_target_image = kwargs.get('preview_target_image', None) if use_preview else target_image
        preview_src_mask = kwargs.get('preview_src_mask', None) if use_preview else src_mask
        preview_src_image = kwargs.get('preview_src_image', None) if use_preview else src_image
        preview_caption = kwargs.get('preview_caption', None) if use_preview else caption
        ret_data = {}
        if preview_src_image is not None:
            if isinstance(preview_src_image, dict):
                prev_src_image = preview_src_image['image']
                prev_src_mask = preview_src_mask['image']
            else:
                prev_src_image = preview_src_image
                prev_src_mask = preview_src_mask
        else:
            prev_src_image = None
            prev_src_mask = None

        if preview_target_image is not None:
            if isinstance(preview_target_image, dict):
                prev_target_image = preview_target_image['image']
            else:
                prev_target_image = preview_target_image
        else:
            prev_target_image = preview_target_image

        # process src image and mask according to mask
        ret_data['src_image'] = None
        if prev_src_image is not None:
            w, h = prev_src_image.size
            if prev_src_mask is None:
                ret_data['src_mask'] = Image.new('L', (w, h), 0)
                src_mask = ret_data['src_mask']
            else:
                src_mask = prev_src_mask.convert('L')
                ow, oh = prev_src_mask.size
                if ow == w and oh == h:
                    ret_data['src_mask'] = prev_src_mask
                else:
                    ret_data['src_mask'] = Image.new('L', (ow, oh), 0)
            ret_data['src_image'] = Image.composite(Image.fromarray(255 - np.array(prev_src_mask)), prev_src_image, src_mask)
        ret_data['target_image'] = prev_target_image
        if caption is not None:
            ret_data['caption'] = preview_caption
        return ret_data


class OpenposeExtractor(BaseExtractor):
    pass


class OutpaintingExtractor(BaseExtractor):
    def model_inference(self,
                        model,
                        image,
                        mask=None,
                        use_mask=False,
                        **kwargs):
        image = np.array(image)
        return model(image, mask=mask if use_mask else None, return_mask=True)

    def __call__(self, **kwargs):
        target_image = kwargs.get('target_image', None)
        src_mask = kwargs.get('src_mask', None)
        src_image = kwargs.get('src_image', None)
        use_preview = kwargs.get('use_preview', True)
        caption = kwargs.get('caption', None)
        use_mask = kwargs.get('use_mask', False)
        # preview_target_image = kwargs.get('preview_target_image', None) if use_preview else target_image
        preview_src_mask = kwargs.get('preview_src_mask', None)
        preview_src_image = kwargs.get('preview_src_image', None)
        preview_caption = kwargs.get('preview_caption',
                                     None) if use_preview else caption
        ret_data = {}
        if preview_src_image is not None:
            if isinstance(preview_src_image, dict):
                prev_src_image = preview_src_image[
                    'background'] if use_preview else src_image
                prev_src_mask = preview_src_image['layers'][0].split(
                                                    )[-1].convert('L')
            else:
                prev_src_image = preview_src_image if use_preview else src_image
                prev_src_mask = preview_src_mask
        else:
            prev_src_image = None
            prev_src_mask = None
        # process src image and mask according to mask
        if prev_src_image is not None:
            w, h = prev_src_image.size
            if prev_src_mask is None:
                ret_data['src_mask'] = Image.new('L', (w, h), 0)
                prev_src_mask = ret_data['src_mask']
            else:
                prev_src_mask = prev_src_mask
                ow, oh = prev_src_mask.size
                if ow == w and oh == h:
                    ret_data['src_mask'] = prev_src_mask
                else:
                    ret_data['src_mask'] = Image.new('L', (ow, oh), 0)
            outpaiting_results = self.model_inference(self.model_info['model'],
                                                      prev_src_image,
                                                      prev_src_mask,
                                                      use_mask=use_mask)
            src_image, target_image, src_mask = (
                outpaiting_results['src_image'], outpaiting_results['image'],
                outpaiting_results['mask'])
            if len(target_image.shape) > 3:
                preview_target_image = Image.fromarray(target_image[0])
            else:
                preview_target_image = Image.fromarray(target_image)

            if len(src_image.shape) > 3:
                preview_src_image = Image.fromarray(src_image[0])
            else:
                preview_src_image = Image.fromarray(src_image)

            if len(src_mask.shape) > 3:
                preview_src_mask = Image.fromarray(src_mask[0])
            else:
                preview_src_mask = Image.fromarray(src_mask)

            ret_data['target_image'] = preview_target_image
            ret_data['src_image'] = preview_src_image
            ret_data['src_mask'] = preview_src_mask
        if caption is not None:
            ret_data['caption'] = preview_caption
        return ret_data


class OutpaintingResize(BaseExtractor):
    def model_inference(self, model, image, target_image, mask=None, **kwargs):
        image = np.array(image)
        return model(image, target_image, mask=mask)

    def __call__(self, **kwargs):
        target_image = kwargs.get('target_image', None)
        src_mask = kwargs.get('src_mask', None)
        src_image = kwargs.get('src_image', None)
        use_preview = kwargs.get('use_preview', True)
        caption = kwargs.get('caption', None)
        use_mask = kwargs.get('use_mask', False)
        preview_target_image = kwargs.get(
            'preview_target_image', None) if use_preview else target_image
        preview_src_mask = kwargs.get('preview_src_mask',
                                      None) if use_preview else src_mask
        preview_src_image = kwargs.get('preview_src_image',
                                       None) if use_preview else src_image
        preview_caption = kwargs.get('preview_caption',
                                     None) if use_preview else caption
        ret_data = {}
        if preview_src_image is not None:
            if isinstance(preview_src_image, dict):
                prev_src_image = preview_src_image['background']
            else:
                prev_src_image = preview_src_image
            if isinstance(preview_src_mask, dict):
                prev_src_mask = preview_src_mask['layers'][0].split(
                                                    )[-1].convert('L')
            else:
                prev_src_mask = preview_src_mask
            if isinstance(preview_target_image, dict):
                prev_target_image = preview_target_image['background']
            else:
                prev_target_image = preview_target_image
        else:
            prev_src_image = None
            prev_src_mask = None
            prev_target_image = None
        # process src image and mask according to mask
        if prev_src_image is not None:
            w, h = prev_src_image.size
            if prev_src_mask is None:
                ret_data['src_mask'] = Image.new('L', (w, h), 0)
                prev_src_mask = ret_data['src_mask']
            else:
                prev_src_mask = prev_src_mask.convert('L')
                ow, oh = prev_src_mask.size
                if ow == w and oh == h:
                    ret_data['src_mask'] = prev_src_mask
                else:
                    ret_data['src_mask'] = Image.new('L', (ow, oh), 0)
            outpaiting_results = self.model_inference(self.model_info['model'],
                                                      prev_src_image,
                                                      prev_target_image,
                                                      prev_src_mask)
            src_image = outpaiting_results['src_image']
            if len(src_image.shape) > 3:
                preview_src_image = Image.fromarray(src_image[0])
            else:
                preview_src_image = Image.fromarray(src_image)
            ret_data['src_image'] = preview_src_image
            ret_data['target_image'] = preview_target_image
            ret_data['src_mask'] = preview_src_mask
        if caption is not None:
            ret_data['caption'] = preview_caption
        return ret_data


class InfoDrawAnimeAnnotator(BaseExtractor):
    pass


class ESAMExtractor(BaseExtractor):
    pass


class InvertExtractor(BaseExtractor):
    pass
