# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import io
import os
import random
import sys
import warnings

import numpy as np
import torch
from tqdm import tqdm

from scepter.modules.data.dataset import DATASETS, BaseDataset
from scepter.modules.utils.distribute import we
from scepter.modules.utils.file_system import FS

try:
    import decord
    decord.bridge.set_bridge('torch')
except ImportError:
    warnings.warn(
        'The `decord` package is required for loading the video dataset. Install with `pip install decord`'
    )


@DATASETS.register_class()
class VideoGenDataset(BaseDataset):
    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger=logger)
        self.prompt_prefix = cfg.get('PROMPT_PREFIX', '')
        self.path_prefix = cfg.get('PATH_PREFIX', '')
        self.p_zero = cfg.get('P_ZERO', 0.0)
        self.max_num_frames = cfg.get('NUM_FRAMES', 49)
        self.fps = cfg.get('FPS', 8)
        self.height = cfg.get('HEIGHT', 480)
        self.width = cfg.get('WIDTH', 720)
        self.skip_frames_start = cfg.get('SKIP_FRAMES_START', 0)
        self.skip_frames_end = cfg.get('SKIP_FRAMES_END', 0)
        self.data_type = cfg.get('DATA_TYPE', 't2v')

    def worker_init_fn(self, worker_id, num_workers=1):
        super().worker_init_fn(worker_id, num_workers=num_workers)
        randseed = np.random.randint(0, 2**32 - num_workers - 1)
        workerseed = randseed + worker_id
        random.seed(workerseed)
        np.random.seed(workerseed)

    def _preprocess_video_data(self, video_path):

        with FS.get_object(video_path) as video_data:
            video_reader = decord.VideoReader(io.BytesIO(video_data),
                                              width=self.width,
                                              height=self.height)
        video_num_frames = len(video_reader)

        start_frame = min(self.skip_frames_start, video_num_frames)
        end_frame = max(0, video_num_frames - self.skip_frames_end)
        if end_frame <= start_frame:
            frames = video_reader.get_batch([start_frame])
        elif end_frame - start_frame <= self.max_num_frames:
            frames = video_reader.get_batch(list(range(start_frame,
                                                       end_frame)))
        else:
            indices = list(
                range(start_frame, end_frame,
                      (end_frame - start_frame) // self.max_num_frames))
            frames = video_reader.get_batch(indices)

        # Ensure that we don't go over the limit
        frames = frames[:self.max_num_frames]
        selected_num_frames = frames.shape[0]

        # Choose first (4k + 1) frames as this is how many is required by the VAE
        remainder = (3 + (selected_num_frames % 4)) % 4
        if remainder != 0:
            frames = frames[:-remainder]
        selected_num_frames = frames.shape[0]

        assert (selected_num_frames - 1) % 4 == 0

        # Training transforms
        frames = frames.float().div_(127.5).sub_(1.)
        frames = frames.permute(3, 0, 1, 2).contiguous()  # [C, F, H, W]
        return frames

    def _parse_index(self, index):
        meta = dict()
        for key, value in zip(index[-1], index[:-1]):
            if key in ['oss_key', 'path', 'video_path', 'target_video_path']:
                meta['video_path'] = value
            elif key in ['source_video_path', 'src_video_path']:
                meta['src_video_path'] = value
            elif key in ['prompt', 'caption', 'text']:
                meta['prompt'] = value
            elif key in ['width', 'height']:
                meta[key] = int(value)
            else:
                meta[key] = value
        return meta

    def _get(self, index):
        meta = self._parse_index(index)

        video_path = os.path.join(self.path_prefix, meta.get('video_path', ''))
        video = self._preprocess_video_data(video_path)

        prompt = self.prompt_prefix + meta.get('prompt', '')
        if self.mode == 'train' and np.random.uniform() < self.p_zero:
            prompt = ''

        item = {
            'video': video,
            'prompt': prompt,
            'meta': meta,
        }
        if 'i2v' in self.data_type:
            item['image'] = item['video'][:, :1, :, :]
        if 'v2v' in self.data_type:
            src_video_path = os.path.join(self.path_prefix,
                                          meta.get('src_video_path', ''))
            src_video = self._preprocess_video_data(src_video_path)
            item['src_video'] = src_video
        return item

    def __len__(self):
        return sys.maxsize

    @staticmethod
    def collate_fn(batch):
        collect = {}
        for sample in batch:
            for k, v in sample.items():
                if k not in collect:
                    collect[k] = []
                collect[k].append(v)
        return collect


@DATASETS.register_class()
class VideoGenDatasetOTF(VideoGenDataset):
    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger)
        self.data_file = cfg.DATA_FILE
        self.delimiter = cfg.get('DELIMITER', '#;#')
        self.fields = cfg.get('FIELDS', ['video_path', 'prompt'])
        self.use_num = cfg.get('USE_NUM', -1)

        from scepter.modules.model.registry import MODELS
        model_cfg = cfg.get('MODEL', None)
        if model_cfg is not None:
            self.model = MODELS.build(
                cfg.MODEL,
                logger=logger).eval().requires_grad_(False).to(we.device_id)
        self.items = self.parse_data(self.data_file, self.delimiter,
                                     self.fields)
        if self.use_num and self.use_num > 0:
            self.items = self.items[:self.use_num]
        self.data = self.encode(self.items)
        self.real_number = len(self.data)
        if model_cfg is not None:
            self.model.to('cpu')
            del self.model
            torch.cuda.empty_cache()

    def parse_data(self, data_file, delimiter, fields):
        items = list()
        with FS.get_object(data_file) as local_data:
            rows = [
                i.split(delimiter,
                        len(fields) - 1)
                for i in local_data.decode('utf-8').strip().split('\n')
            ]
            for i, row in enumerate(rows):
                item = {}
                for key, value in zip(self.fields, row):
                    if key in ['oss_key', 'path', 'video_path']:
                        item['video_path'] = value
                    elif key in ['prompt', 'caption', 'text']:
                        item['prompt'] = value
                    elif key in ['width', 'height']:
                        item[key] = int(value)
                    else:
                        item[key] = value
                items.append(item)
        return items

    def encode(self, items):
        self.logger.info('Start to encode video data [{}]!'.format(len(items)))
        for item in tqdm(items):
            video_path = os.path.join(self.path_prefix,
                                      item.get('video_path', ''))
            video = self._preprocess_video_data(video_path)
            latent = self.model.encode_first_stage(
                video.unsqueeze(0).to(we.device_id)).squeeze(0)
            item['video_latent'] = latent.detach().cpu()
            item['video'] = video
            if self.data_type == 'i2v':
                item['image'] = item['video'][:, :1, :, :]
        return items

    def _get(self, index):
        return self.data[index % self.real_number]
