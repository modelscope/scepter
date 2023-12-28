# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import numbers
import os.path as osp
import queue
import random
import threading

import numpy as np
import torch

from scepter.modules.transform import LoadImageFromFile
from scepter.modules.transform.registry import TRANSFORMS
from scepter.modules.utils.config import dict_to_yaml
from scepter.modules.utils.file_system import DATA_FS as FS
from scepter.modules.utils.video_reader.frame_sampler import do_frame_sample
from scepter.modules.utils.video_reader.video_reader import VideoReaderWrapper


def _interval_based_sampling(vid_length,
                             vid_fps,
                             target_fps,
                             clip_idx,
                             num_clips,
                             num_frames,
                             interval,
                             minus_interval=False):
    """ Generates the frame index list using interval based sampling.

    Args:
        vid_length (int): The length of the whole video (valid selection range).
        vid_fps (float): The original video fps.
        target_fps (int): The target decode fps.
        clip_idx (int): -1 for random temporal sampling, and positive values for sampling specific clip from the video.
        num_clips (int): The total clips to be sampled from each video.
            Combined with clip_idx, the sampled video is the "clip_idx-th" video from "num_clips" videos.
        num_frames (int): Number of frames in each sampled clips.
        interval (int): The interval to sample each frame.
        minus_interval (bool):

    Returns:
        index (torch.Tensor): The sampled frame indexes.
    """
    if num_frames == 1:
        index = [random.randint(0, vid_length - 1)]
    else:
        # transform FPS
        clip_length = num_frames * interval * vid_fps / target_fps

        max_idx = max(vid_length - clip_length, 0)
        if clip_idx == -1:  # random sampling
            start_idx = random.uniform(0, max_idx)
        else:
            if num_clips == 1:
                start_idx = max_idx / 2
            else:
                start_idx = max_idx * clip_idx / num_clips
        if minus_interval:
            end_idx = start_idx + clip_length - interval
        else:
            end_idx = start_idx + clip_length - 1

        index = torch.linspace(start_idx, end_idx, num_frames)
        index = torch.clamp(index, 0, vid_length - 1).long()

    return index


def _segment_based_sampling(vid_length, clip_idx, num_clips, num_frames,
                            random_sample):
    """ Generates the frame index list using segment based sampling.

    Args:
        vid_length (int): The length of the whole video (valid selection range).
        clip_idx (int): -1 for random temporal sampling, and positive values for sampling specific clip from the video.
        num_clips (int): The total clips to be sampled from each video.
            Combined with clip_idx, the sampled video is the "clip_idx-th" video from "num_clips" videos.
        num_frames (int): Number of frames in each sampled clips.
        random_sample (bool): Whether or not to randomly sample from each segment. True for train and False for test.

    Returns:
        index (torch.Tensor): The sampled frame indexes.
    """
    index = torch.zeros(num_frames)
    index_range = torch.linspace(0, vid_length, num_frames + 1)
    for idx in range(num_frames):
        if random_sample:
            index[idx] = random.uniform(index_range[idx], index_range[idx + 1])
        else:
            if num_clips == 1:
                index[idx] = (index_range[idx] + index_range[idx + 1]) / 2
            else:
                index[idx] = index_range[idx] + (index_range[
                    idx + 1] - index_range[idx]) * (clip_idx + 1) / num_clips
    index = torch.round(torch.clamp(index, 0, vid_length - 1)).long()

    return index


R = threading.Lock()


@TRANSFORMS.register_class()
class LoadVideoFromFile(object):
    """ Open video file, extract frames, convert to tensor.

    Args:
        num_frames (int): T dimension value.
        sample_type (str): See
            `from essmc2.metric.video_reader import FRAME_SAMPLERS; print(FRAME_SAMPLERS)` to get candidates,
            default is 'interval'.
        clip_duration (Optional[float]): Needed for 'interval' sampling type.
        decoder (str): Video decoder name, default is decord.

    """
    def __init__(self, cfg, logger=None):
        self.num_frames = cfg.NUM_FRAMES
        self.sample_type = cfg.get('SAMPLE_TYPE', 'interval')
        self.clip_duration = cfg.get('CLIP_DURATION', None)

        assert self.sample_type in ('uniform', 'interval', 'segment'), \
            f'Expected sample type in (uniform, interval, segment), got {self.sample_type}'
        if self.sample_type == 'interval':
            assert isinstance(self.clip_duration, numbers.Number), \
                'Interval style sampling needs clip_duration not None'
        self.decoder = cfg.get('DECODER', 'decord')

    def __call__(self, item):
        """
        Args:
            item (dict):
                item['meta']['prefix'] (Optional[str]): Prefix of video_path.
                item['meta']['video_path'] (str): Required.
                item['meta']['clip_id'] (Optional[int]): Multi-view test needs it, default is 0.
                item['meta']['num_clips'] (Optional[int]): Multi-view test needs it, default is 1.
                item['meta']['start_sec'] (Optional[float]): Uniform sampling needs it.
                item['meta']['end_sec'] (Optional[float]): Uniform sampling needs it.

        Returns:
            item(dict):
                item['video'] (torch.Tensor): a THWC tensor.
        """
        meta = item['meta']
        video_path = meta['video_path'] if 'prefix' not in meta else osp.join(
            meta['prefix'], meta['video_path'])

        with FS.get_from(video_path) as local_path:
            vr = VideoReaderWrapper(local_path, decoder=self.decoder)

            params = dict()
            clip_id = meta.get('clip_id') or 0
            num_clips = meta.get('num_clips') or 1
            if self.sample_type == 'interval':
                # default is test mode for interval and segment
                params.update(clip_duration=self.clip_duration,
                              clip_id=clip_id,
                              num_clips=num_clips)
            elif self.sample_type == 'segment':
                # default is test mode for interval and segment
                params.update(clip_id=clip_id, num_clips=num_clips)
            else:
                # uniform, needs start_sec, clip_duration or end_sec
                start_sec = meta['start_sec'] - meta['start_sec']
                if 'end_sec' in meta:
                    end_sec = meta['end_sec'] - meta['start_sec']
                elif self.clip_duration is not None:
                    end_sec = start_sec + self.clip_duration
                else:
                    raise ValueError(
                        'Uniform sampling needs start_sec & end_sec / start_sec & clip_duration'
                    )
                params.update(start_sec=start_sec, end_sec=end_sec)

            decode_list = do_frame_sample(self.sample_type, vr.len, vr.fps,
                                          self.num_frames, **params)
            item['video'] = vr.sample_frames(decode_list)

        return item

    @staticmethod
    def get_config_template():
        '''
        { "ENV" :
            { "description" : "",
              "A" : {
                    "value": 1.0,
                    "description": ""
               }
            }
        }
        :return:
        '''
        para_dict = [{
            'SAMPLE_TYPE': {
                'value': 'interval',
                'description': 'sample type'
            },
            'CLIP_DURATION': {
                'value': None,
                'description': 'clip duration'
            }
        }]
        return dict_to_yaml('TRANSFORM',
                            __class__.__name__,
                            para_dict,
                            set_name=True)


@TRANSFORMS.register_class()
class LoadVideoFromFrameList(object):
    """ extract frames, convert to tensor.

    Args:
        num_frames (int): T dimension value.
        sample_type (str): See
            `from essmc2.metric.video_reader import FRAME_SAMPLERS; print(FRAME_SAMPLERS)` to get candidates,
            default is 'interval'.
        clip_duration (Optional[float]): Needed for 'interval' sampling type.
        decoder (str): Video decoder name, default is decord.
    """
    para_dict = [{
        'NUM_FRAMES': {
            'value': 30,
            'description': 'clip length!'
        },
        'SAMPLE_TYPE': {
            'value': 'interval',
            'description': 'sample type'
        },
        'CLIP_DURATION': {
            'value': None,
            'description': 'clip duration'
        },
        'RGB_ORDER': {
            'value': 'RGB',
            'description': 'rgb order'
        },
        'BACKEND': {
            'value': 'pillow',
            'description': 'input backend'
        }
    }]

    def __init__(self, cfg, logger=None):
        self.load_ins = LoadImageFromFile(cfg, logger=logger)
        self.num_frames = cfg.NUM_FRAMES
        self.sample_type = cfg.get('SAMPLE_TYPE', 'interval')
        self.clip_duration = cfg.get('CLIP_DURATION', None)

        assert self.sample_type in ('uniform', 'interval', 'segment'), \
            f'Expected sample type in (uniform, interval, segment), got {self.sample_type}'
        if self.sample_type == 'interval':
            assert isinstance(self.clip_duration, numbers.Number), \
                'Interval style sampling needs clip_duration not None'

    def __call__(self, item):
        """
        Args:
            item (dict):
                item['meta']['video_path'] (str): Required.
                item['meta']['clip_id'] (Optional[int]): Multi-view test needs it, default is 0.
                item['meta']['num_clips'] (Optional[int]): Multi-view test needs it, default is 1.
                item['meta']['start_sec'] (Optional[float]): Uniform sampling needs it.
                item['meta']['end_sec'] (Optional[float]): Uniform sampling needs it.
                item['meta']['frames'](list): all frames file path list

        Returns:
            item(dict):
                item['video'] (torch.Tensor): a THWC tensor.
        """
        meta = item['meta']
        params = dict()
        clip_id = meta.get('clip_id') or 0
        num_clips = meta.get('num_clips') or 1
        if self.sample_type == 'interval':
            # default is test mode for interval and segment
            params.update(clip_duration=self.clip_duration,
                          clip_id=clip_id,
                          num_clips=num_clips)
        elif self.sample_type == 'segment':
            # default is test mode for interval and segment
            params.update(clip_id=clip_id, num_clips=num_clips)
        else:
            # uniform, needs start_sec, clip_duration or end_sec
            start_sec = meta['start_sec'] - meta['start_sec']
            if 'end_sec' in meta:
                end_sec = meta['end_sec'] - meta['start_sec']
            elif self.clip_duration is not None:
                end_sec = start_sec + self.clip_duration
            else:
                raise ValueError(
                    'Uniform sampling needs start_sec & end_sec / start_sec & clip_duration'
                )
            params.update(start_sec=start_sec, end_sec=end_sec)
        frames = meta['frames']
        fps = meta['fps']
        decode_list = do_frame_sample(self.sample_type, len(frames), fps,
                                      self.num_frames, **params)
        sample_frames = [{'meta': {'img_path': frame}} for frame in frames]
        img_path_queue = queue.Queue()
        [
            img_path_queue.put_nowait([idx, item])
            for idx, item in enumerate(sample_frames)
        ]
        img_queue = queue.Queue()

        def download_file():
            while not img_path_queue.empty():
                R.acquire()
                try:
                    idx, item = img_path_queue.get_nowait()
                except Exception:
                    R.release()
                    continue
                R.release()
                img_queue.put_nowait([idx, self.load_ins(item)])

        threading_list = []
        for _ in range(8):
            t = threading.Thread(target=download_file)
            t.daemon = True
            t.start()
            threading_list.append(t)
        [th.join() for th in threading_list]
        # print(f"one video download time {time.time() - st}")
        sample_frames = []
        while not img_queue.empty():
            sample_frames.append(img_queue.get_nowait())
        sample_frames.sort(key=lambda x: x[0])
        # sample_frames = [self.load_ins(item) for item in sample_frames]
        item['video'] = np.array(
            [sample_frames[frame_id][1]['img'] for frame_id in decode_list])
        # item['video'] = item['video'].transpose([0, 3, 1, 2])
        item['meta'].pop('frames')
        return item

    @staticmethod
    def get_config_template():
        '''
        { "ENV" :
            { "description" : "",
              "A" : {
                    "value": 1.0,
                    "description": ""
               }
            }
        }
        :return:
        '''
        return dict_to_yaml('TRANSFORM',
                            __class__.__name__,
                            LoadVideoFromFrameList.para_dict,
                            set_name=True)


@TRANSFORMS.register_class()
class DecodeVideoToTensor(object):
    def __init__(self, cfg, logger=None):
        """ DecodeVideoToTensor
        Args:
            num_frames (int): Decode frames number.
            target_fps (int): Decode frames fps, default is 30.
            sample_mode (str): Interval or segment sampling, default is interval.
            sample_interval (int): Sample interval between output frames for interval sample mode, default is 4.
            sample_minus_interval (bool): If minus interval for interval sample mode, default is False.
            repeat (int): Number of clips to be decoded from each video, if repeat > 1, outputs will be named like
                'video-0', 'video-1'. Normally, 1 for classification task, 2 for contrastive learning.
        """
        import decord
        from decord import VideoReader
        self.VideoReader = VideoReader
        decord.bridge.set_bridge('torch')
        self.num_frames = cfg.NUM_FRAMES
        self.target_fps = cfg.get('TARGET_FPS', 30)
        self.sample_mode = cfg.get('SAMPLE_MODE', 'interval')
        self.sample_interval = cfg.get('SAMPLE_INTERVAL', 4)
        self.sample_minus_interval = cfg.get('SAMPLE_MINUS_INTERVAL', False)
        self.repeat = cfg.get('REPEAT', 1)

    def __call__(self, item):
        """ Call to invoke decode
        Args:
            item (dict): A dict contains which file to decode and how to decode.
                Normally, it has structure like
                    {
                        "meta": {
                            "prefix" (str, None): if not None, prefix will be added to video_path.
                            "video_path" (str): Absolute (prefix is None) or relative path.
                            "clip_idx" (int): -1 means random sampling, >=0 means do temporal crop.
                            "num_clips" (int): if clip_idx >= 0, clip_idx must < num_clips
                        }
                    }

        Returns:
            A dict contains original input item and "video" tensor.
        """
        meta = item['meta']
        video_path = meta['video_path'] \
            if 'prefix' not in meta else osp.join(meta['prefix'], meta['video_path'])

        with FS.get_from(video_path) as local_path:
            vr = self.VideoReader(local_path)
            # default is test mode
            clip_id = meta.get('clip_id') or 0
            num_clips = meta.get('num_clips') or 1

            vid_len = len(vr)
            vid_fps = vr.get_avg_fps()

            frame_list = []
            for _ in range(self.repeat):
                if self.sample_mode == 'interval':
                    decode_list = _interval_based_sampling(
                        vid_len, vid_fps, self.target_fps, clip_id, num_clips,
                        self.num_frames, self.sample_interval,
                        self.sample_minus_interval)
                else:
                    decode_list = _segment_based_sampling(
                        vid_len, clip_id, num_clips, self.num_frames,
                        clip_id == -1)

                # Decord gives inconsistent result for avi files. Getting full frames will fix it, although slower.
                # See https://github.com/dmlc/decord/issues/195
                if video_path.lower().endswith('avi'):
                    full_decode_list = list(
                        range(0,
                              torch.max(decode_list).item() + 1))
                    full_frames = vr.get_batch(full_decode_list)
                    frames = full_frames[decode_list].clone()
                else:
                    frames = vr.get_batch(decode_list).clone()

                frame_list.append(frames)

            if self.repeat == 1:
                item['video'] = frame_list[0]
            else:
                for idx, frame_tensor in zip(range(self.repeat), frame_list):
                    item[f'video-{idx}'] = frame_tensor

            del vr
        return item

    @staticmethod
    def get_config_template():
        '''
        { "ENV" :
            { "description" : "",
              "A" : {
                    "value": 1.0,
                    "description": ""
               }
            }
        }
        :return:
        '''
        para_dict = [{
            'NUM_FRAMES': {
                'value': 30,
                'description': 'num frame'
            },
            'TARGET_FPS': {
                'value': 30,
                'description': 'target fps'
            },
            'SAMPLE_MODE': {
                'value': 'interval',
                'description': 'sample mode'
            },
            'SAMPLE_INTERVAL': {
                'value': 4,
                'description': 'sample interval'
            },
            'SAMPLE_MINUS_INTERVAL': {
                'value': False,
                'description': 'sample minus interval'
            },
            'REPEAT': {
                'value': 1,
                'description': 'repeat'
            }
        }]
        return dict_to_yaml('TRANSFORM',
                            __class__.__name__,
                            para_dict,
                            set_name=True)
