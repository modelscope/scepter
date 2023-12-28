# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import os
from fractions import Fraction
from typing import Callable, Optional, Union

import cv2
import numpy as np
import torch
import torch.utils.dlpack as dlpack

from scepter.modules.utils.file_system import FS
from scepter.modules.utils.video_reader.frame_sampler import do_frame_sample


class _Wrapper(object):
    @property
    def len(self) -> int:
        raise NotImplementedError

    @property
    def fps(self) -> float:
        raise NotImplementedError

    @property
    def duration(self) -> float:
        raise NotImplementedError

    def sample_frames(self, decode_list: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class VideoReaderWrapper(_Wrapper):
    def __init__(self, video_path):
        import decord
        self._video_path = video_path
        self._decoder_type = 'decord'
        self._vr = decord.VideoReader(self._video_path)

    @property
    def len(self):
        return len(self._vr)

    @property
    def fps(self):
        return self._vr.get_avg_fps()

    @property
    def duration(self):
        return float(self.len) / self.fps

    def __del__(self):
        if self._vr is not None:
            del self._vr
            self._vr = None

    def sample_frames(self, decode_list: torch.Tensor) -> torch.Tensor:
        frames = dlpack.from_dlpack(
            self._vr.get_batch(decode_list).to_dlpack()).clone()
        return frames


class FramesReaderWrapper(_Wrapper):
    def __init__(self, frame_dir: str, extract_fps: float, suffix='.jpg'):
        self._frame_dir = frame_dir
        self._extract_fps = extract_fps
        self._suffix = suffix
        self._frame_list = sorted([
            os.path.join(self._frame_dir, t)
            for t in os.listdir(self._frame_dir) if t.endswith(self._suffix)
        ])
        self._frames = [None] * len(self._frame_list)

    @property
    def len(self) -> int:
        return len(self._frame_list)

    @property
    def fps(self):
        return self._extract_fps

    @property
    def duration(self):
        return float(self.len) / self.fps

    def _load_frame(self, idx):
        path = self._frame_list[idx]
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self._frames[idx] = img

    def sample_frames(self, decode_list: torch.Tensor) -> torch.Tensor:
        ret = []
        for idx in decode_list.numpy():
            if self._frames[idx] is None:
                self._load_frame(idx)
            ret.append(self._frames[idx].copy())
        ret = np.asarray(ret)
        return torch.from_numpy(ret)


class EasyVideoReader(object):
    """ A video reader which is easy to use in real applications.

    Args:
         video_path (str): Path of video file.
         num_frames (int): Extract frames for one sample.
         clip_duration (Union[float, Fraction, str]): Clip duration to be extracted uniformly.
         overlap (Union[float, Fraction, str]): The offset (in secs) of
         the next clip overlaps the last clip, default is 0 no overlap.
         transforms (Optional[Callable]): Do transform operations, default is None.

    """
    def __init__(self,
                 video_path: str,
                 num_frames: int,
                 clip_duration: Union[float, Fraction, str],
                 overlap: Union[float, Fraction, str] = Fraction(0),
                 transforms: Optional[Callable] = None):
        self._video_path: str = video_path
        self._num_frames: int = num_frames
        self._clip_duration: Fraction = Fraction(clip_duration)
        self._overlap: Fraction = Fraction(overlap)
        assert self._overlap < self._clip_duration, 'Overlap must be smaller than clip_duration!'
        self._transforms = transforms

        self._last_end: Fraction = Fraction(0)

        client = FS.get_fs_client(self._video_path)
        local_path = client.get_object_to_local_file(self._video_path)
        self._vr = VideoReaderWrapper(local_path)

    def __iter__(self):
        return self

    def __next__(self):
        start_sec = max(Fraction(0), self._last_end - self._overlap)
        end_sec = start_sec + self._clip_duration

        if end_sec > self._vr.duration:
            del self._vr
            raise StopIteration

        decode_list = do_frame_sample('uniform',
                                      self._vr.len,
                                      self._vr.fps,
                                      self._num_frames,
                                      start_sec=float(start_sec),
                                      end_sec=float(end_sec))
        output_tensor = self._vr.sample_frames(decode_list)

        self._last_end = end_sec

        output = {
            'video': output_tensor,
            'meta': {
                'video_path': self._video_path,
                'start_sec': float(start_sec),
                'end_sec': float(end_sec)
            }
        }

        if self._transforms is not None:
            return self._transforms(output)
        return output
