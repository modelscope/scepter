# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""
FrameSampler.

Sample:
    1. give start & end time, num_frames, e.g. 16 frames from [1.0s ,3.0s],
    usually used in real applications.
    fixed args:
        `sample_type`='uniform';
        `vid_len` (int): valid total frame numbers in video;
        `vid_fps` (float): video fps;
        `num_frames` (int): number of frames to be extracted;
    extra args:

    2. give a fixed clip duration, num_frames, e.g. 16 frames from a 2s clip.
    In train mode (`clip_id`=-1), this clip will be randomly sampled from video.
    In test mode (`clip_id`>=0), this clip is the center part of the video
    which acts the same as `DecodeVideoToTensor` op.
    fixed args: `sample_mode`='interval', `vid_len`, `vid_fps`, `num_frames`
    extra args: `clip_duration`, `clip_id`, `num_clips`=1

    3. give a fixed clip duration, constant total clips, current clip index, num_frames,
    e.g. three 2-s clips will be sampled from the video, and 16 frames from the first clip.
    Usually used in multi-view test.
    In train mode (`clip_id`=-1), constant total clips will be ignored, so this will act the same b.
    In test mode (`clip_id`>=0), video is splitted into constant clips (uniformly and allow overlap,
    a 3s video splits into three 2s clips, [0, 2), [0.5, 2.5), [1.0, 3.0) ),
    then sample frames from one clip.
    fixed args: `sample_mode`='interval', `vid_len`, `vid_fps`, `num_frames`
    extra args: `clip_duration`, `clip_id`, `num_clips`

    4. give num_frames, do segment-sampling, e.g. 16 frames from whole video, then splits the video into 16 segments,
    and sample one frame from each segment.
    In train mode (`clip_id`=-1), sample a frame randomly from a segment.
    In test mode (`clip_id`>=0), the center frame in each part will be chosen.
    fixed args: `sample_mode`='segment', `vid_len`, `vid_fps`, `num_frames`
    call args: `clip_id`, `num_clips`=1

    5. give constant total clips, current clip index, num_frames,
    e.g. splits the video into 16 segments, split one segment into 3 parts,
    if clip_index=0, sample one frame from the first part, and loop 16 times.
    In train mode, sample a frame randomly from a segment.
    In test mode, int(`clip_id`/`num_clips` * segment_frames) will be chosen.
    fixed args: `sample_mode`='segment', `vid_len`, `vid_fps`, `num_frames`
    call args: `clip_id`, `num_clips`

Output:
    A list of frame indices (torch.Tensor)

"""
import math
import random

import torch

from scepter.modules.utils.config import Config
from scepter.modules.utils.registry import Registry

FRAME_SAMPLERS = Registry('FRAME_SAMPLERS')


def do_frame_sample(sampling_type: str, vid_len: int, vid_fps: float,
                    num_frames: int, **kwargs) -> torch.Tensor:
    params = dict(vid_len=vid_len,
                  vid_fps=vid_fps,
                  num_frames=num_frames,
                  **kwargs)
    return FRAME_SAMPLERS.build(
        Config(cfg_dict={'NAME': sampling_type}, load=False))(**params)


@FRAME_SAMPLERS.register_class('uniform')
class UniformSampler(object):
    def __init__(self, cfg, logger=None):
        self.cfg = cfg
        self.logger = logger

    def __call__(self,
                 vid_len: int,
                 vid_fps: float,
                 num_frames: int,
                 start_sec: float = 0,
                 end_sec: float = -1) -> torch.Tensor:
        start_sec = max(start_sec, 0)
        if end_sec < 0:
            new_end_sec = vid_len / vid_fps
        else:
            new_end_sec = min(end_sec, vid_len / vid_fps)
        assert new_end_sec > start_sec, (
            f'end_sec should be greater then start_sec, '
            f'got end_sec={new_end_sec}, start_sec={start_sec}')
        end_sec = new_end_sec

        start_idx = math.floor(start_sec / vid_fps)
        end_idx_exc = min(vid_len, math.ceil(end_sec / vid_fps))

        index = torch.linspace(start_idx, end_idx_exc, num_frames)
        index = torch.clamp(index, 0, vid_len - 1).long()

        return index


@FRAME_SAMPLERS.register_class('interval')
class IntervalSampler(object):
    def __init__(self, cfg, logger=None):
        self.cfg = cfg
        self.logger = logger

    def __call__(self,
                 vid_len: int,
                 vid_fps: float,
                 num_frames: int,
                 clip_duration: float,
                 clip_id: int = 0,
                 num_clips: int = 1) -> torch.Tensor:
        if num_frames == 1:
            return torch.randint(0, vid_len, (1, ))

        clip_len = int(clip_duration / vid_fps)
        max_idx = max(vid_len, clip_len, 0)

        if clip_id == -1:
            start_idx = random.uniform(0, max_idx)
        else:
            if num_clips == 1:
                start_idx = max_idx / 2
            else:
                start_idx = max_idx * clip_id / num_clips

        end_idx = start_idx + clip_len - 1
        index = torch.linspace(start_idx, end_idx, num_frames)
        index = torch.clamp(index, 0, vid_len - 1).long()
        return index


@FRAME_SAMPLERS.register_class('segment')
class SegmentSampler(object):
    def __init__(self, cfg, logger=None):
        self.cfg = cfg
        self.logger = logger

    def __call__(self,
                 vid_len: int,
                 vid_fps: float,
                 num_frames: int,
                 clip_id: int = 0,
                 num_clips: int = 1) -> torch.Tensor:
        index = torch.zeros(num_frames)
        index_range = torch.linspace(0, vid_len, num_frames + 1)
        for idx in range(num_frames):
            if clip_id == -1:
                index[idx] = random.uniform(index_range[idx],
                                            index_range[idx + 1])
            else:
                if num_clips == 1:
                    index[idx] = (index_range[idx] + index_range[idx + 1]) / 2
                else:
                    index[idx] = index_range[idx] + (
                        index_range[idx + 1] -
                        index_range[idx]) * (clip_id + 1) / num_clips

        index = torch.round(torch.clamp(index, 0, vid_len - 1)).long()

        return index
