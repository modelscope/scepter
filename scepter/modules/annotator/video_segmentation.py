# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from abc import ABCMeta

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
try:
    from sklearn.cluster import KMeans
except:
    import warnings
    warnings.warn("ignore sklearn import, please pip install scikit-learn.")

from scepter.modules.annotator.base_annotator import BaseAnnotator
from scepter.modules.annotator.registry import ANNOTATORS
from scepter.modules.utils.config import dict_to_yaml
from scepter.modules.utils.file_system import FS
import pycocotools.mask as mask_utils


def single_mask_to_rle(mask):
    rle = mask_utils.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle

def single_rle_to_mask(rle):
    mask = np.array(mask_utils.decode(rle)).astype(np.uint8)
    return mask

def single_mask_to_xyxy(mask):
    bbox = np.zeros((4), dtype=int)
    rows, cols = np.where(np.array(mask))
    if len(rows) > 0 and len(cols) > 0:
        x_min, x_max = np.min(cols), np.max(cols)
        y_min, y_max = np.min(rows), np.max(rows)
        bbox[:] = [x_min, y_min, x_max, y_max]
    return bbox.tolist()

@ANNOTATORS.register_class()
class SAM2DrawVideoAnnotator(BaseAnnotator, metaclass=ABCMeta):
    para_dict = {}

    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger=logger)
        self.task_type = cfg.get('TASK_TYPE', 'input_box')
        from sam2.build_sam import build_sam2_video_predictor
        config_path = FS.get_from(cfg.CONFIG_PATH, local_path=cfg.CONFIG_LOCAL_PATH, wait_finish=True)
        pretrained_model = FS.get_from(cfg.PRETRAINED_MODEL, wait_finish=True)
        self.video_predictor = build_sam2_video_predictor(config_path, pretrained_model, fill_hole_area=0)

    def forward(self,
                video,
                input_box=None,
                mask=None,
                task_type=None):
        task_type = task_type if task_type is not None else self.task_type

        if mask is not None:
            if isinstance(mask, Image.Image):
                mask = np.array(mask)
            elif isinstance(mask, torch.Tensor):
                mask = mask.detach().cpu().numpy()
            elif isinstance(mask, np.ndarray):
                mask = mask.copy()
            else:
                raise f'Unsurpport datatype{type(mask)}, only surpport np.ndarray, torch.Tensor, Pillow Image.'

        if task_type == 'mask_point':
            if len(mask.shape) == 3:
                scribble = mask.transpose(2, 1, 0)[0]
            else:
                scribble = mask.transpose(1, 0)   # (H, W) -> (W, H)
            labeled_array, num_features = ndimage.label(scribble >= 255)
            centers = ndimage.center_of_mass(scribble, labeled_array,
                                             range(1, num_features + 1))
            point_coords = np.array(centers)
            point_labels = np.array([1] * len(centers))
            sample = {
                'points': point_coords,
                'labels': point_labels
            }
        elif task_type == 'mask_box':
            if len(mask.shape) == 3:
                scribble = mask.transpose(2, 1, 0)[0]
            else:
                scribble = mask.transpose(1, 0)  # (H, W) -> (W, H)
            labeled_array, num_features = ndimage.label(scribble >= 255)
            centers = ndimage.center_of_mass(scribble, labeled_array,
                                             range(1, num_features + 1))
            centers = np.array(centers)
            # (x1, y1, x2, y2)
            x_min = centers[:, 0].min()
            x_max = centers[:, 0].max()
            y_min = centers[:, 1].min()
            y_max = centers[:, 1].max()
            bbox = np.array([x_min, y_min, x_max, y_max])
            sample = {'box': bbox}
        elif task_type == 'input_box':
            if isinstance(input_box, list):
                input_box = np.array(input_box)
            sample = {'box': input_box}
        elif task_type == 'mask':
            sample = {'mask': mask}
        else:
            raise NotImplementedError

        ann_frame_idx = 0
        object_id = 0
        with (torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16)):

            inference_state = self.video_predictor.init_state(video_path=video)

            if task_type in ['mask_point', 'mask_box', 'input_box']:
                _, out_obj_ids, out_mask_logits = self.video_predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=object_id,
                    **sample
                )
            elif task_type in ['mask']:
                _, out_obj_ids, out_mask_logits = self.video_predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=object_id,
                    **sample
                )
            else:
                raise NotImplementedError

            video_segments = {}  # video_segments contains the per-frame segmentation results
            for out_frame_idx, out_obj_ids, out_mask_logits in self.video_predictor.propagate_in_video(inference_state):
                frame_segments = {}
                for i, out_obj_id in enumerate(out_obj_ids):
                    mask = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze(0)
                    frame_segments[out_obj_id] = {
                        "mask": single_mask_to_rle(mask),
                        "mask_area": int(mask.sum()),
                        "mask_box": single_mask_to_xyxy(mask),
                    }
                video_segments[out_frame_idx] = frame_segments

        ret_data = {
            "annotations": video_segments
        }
        return ret_data

    @staticmethod
    def get_config_template():
        return dict_to_yaml('ANNOTATORS',
                            __class__.__name__,
                            SAM2DrawVideoAnnotator.para_dict,
                            set_name=True)
