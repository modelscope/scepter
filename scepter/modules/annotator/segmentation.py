# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import math
import random
from abc import ABCMeta

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from scipy import ndimage
from pycocotools import mask as mask_utils
from scepter.modules.annotator.base_annotator import BaseAnnotator
from scepter.modules.annotator.registry import ANNOTATORS
from scepter.modules.utils.config import dict_to_yaml
from scepter.modules.utils.distribute import we
from scepter.modules.utils.file_system import FS
from sklearn.cluster import KMeans
from torchvision.ops.boxes import batched_nms


def find_dominant_color(image, k=1):
    pixels = image.reshape((-1, 3))
    mask = (pixels != [0, 0, 0]).all(axis=1)
    pixels = pixels[mask]
    try:
        kmeans = KMeans(n_clusters=k, n_init='auto')
        kmeans.fit(pixels)
        dominant_color = kmeans.cluster_centers_.astype(int)[0]
    except:
        dominant_color = np.array([255, 255, 255])
    return dominant_color


def cv2_resize_crop(image, resize_size, crop_size):
    resize_height, resize_width = resize_size
    crop_height, crop_width = crop_size

    resized_image = cv2.resize(image, (resize_width, resize_height))

    center_x, center_y = resize_width // 2, resize_height // 2
    crop_start_x = max(center_x - crop_width // 2, 0)
    crop_start_y = max(center_y - crop_height // 2, 0)
    crop_end_x = crop_start_x + crop_width
    crop_end_y = crop_start_y + crop_height

    crop_end_x = min(crop_end_x, resize_width)
    crop_end_y = min(crop_end_y, resize_height)

    center_cropped_image = resized_image[crop_start_y:crop_end_y,
                                         crop_start_x:crop_end_x]

    return center_cropped_image


@ANNOTATORS.register_class()
class ESAMAnnotator(BaseAnnotator, metaclass=ABCMeta):
    para_dict = {}

    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger=logger)
        try:
            from efficient_sam.efficient_sam import build_efficient_sam
            from segment_anything.utils.amg import (
                batched_mask_to_box,
                calculate_stability_score,
                mask_to_rle_pytorch,
                remove_small_regions,
                rle_to_mask,
            )
        except:
            raise NotImplementedError(
                f'Please install efficient_sam and segment_anything modules.')

        pretrained_model = cfg.get('PRETRAINED_MODEL', None)
        if pretrained_model:
            with FS.get_from(pretrained_model, wait_finish=True) as local_path:
                self.efficient_sam_module = build_efficient_sam(
                    encoder_patch_embed_dim=384,
                    encoder_num_heads=6,
                    checkpoint=local_path).eval().to(we.device_id)
        self.GRID_SIZE = cfg.get('GRID_SIZE', 16)
        self.save_mode = cfg.get('SAVE_MODE', 'P')
        self.use_dominant_color = cfg.get('USE_DOMINANT_COLOR', False)
        self.return_mask = cfg.get('RETURN_MASK', False)

    @torch.no_grad()
    def get_predictions_given_embeddings_and_queries(self, img, points,
                                                     point_labels, model):
        from segment_anything.utils.amg import calculate_stability_score
        predicted_masks, predicted_iou = [], []
        bs = 128
        num = int(float(self.GRID_SIZE * self.GRID_SIZE) /
                  bs) if self.GRID_SIZE * self.GRID_SIZE % bs == 0 else int(
                      float(self.GRID_SIZE * self.GRID_SIZE) / bs) + 1
        for i in range(num):
            predicted_mask_item, predicted_iou_item = model(
                img[None, ...], points[:, i * bs:(i + 1) * bs, ...],
                point_labels[:, i * bs:(i + 1) * bs, :])
            predicted_masks.append(predicted_mask_item)
            predicted_iou.append(predicted_iou_item)
            torch.cuda.empty_cache()
        # # predicted:  torch.Size([1, 1024, 3]) torch.Size([1, 1024, 3, 512, 512])
        # print('predicted: ', predicted_iou.size(), predicted_masks.size())
        predicted_masks = torch.cat(predicted_masks, dim=1)
        predicted_iou = torch.cat(predicted_iou, dim=1)
        sorted_ids = torch.argsort(predicted_iou, dim=-1, descending=True)
        predicted_iou_scores = torch.take_along_dim(predicted_iou,
                                                    sorted_ids,
                                                    dim=2)
        predicted_masks = torch.take_along_dim(predicted_masks,
                                               sorted_ids[..., None, None],
                                               dim=2)
        predicted_masks = predicted_masks[0]
        iou = predicted_iou_scores[0, :, 0]
        index_iou = iou > 0.7
        iou_ = iou[index_iou]
        masks = predicted_masks[index_iou]
        score = calculate_stability_score(masks, 0.0, 1.0)
        score = score[:, 0]
        index = score > 0.9
        masks = masks[index]
        iou_ = iou_[index]
        masks = torch.ge(masks, 0.0)
        return masks, iou_

    def singel_mask_to_rle(self, mask):
        rle = mask_utils.encode(
            np.array(mask[:, :, None], order='F', dtype='uint8'))[0]
        rle['counts'] = rle['counts'].decode('utf-8')
        return rle

    def process_small_region(self, rles):
        from segment_anything.utils.amg import rle_to_mask, remove_small_regions, \
            batched_mask_to_box, mask_to_rle_pytorch
        new_masks = []
        scores = []
        min_area = 100
        nms_thresh = 0.7
        for rle in rles:
            mask = rle_to_mask(rle[0])

            mask, changed = remove_small_regions(mask, min_area, mode='holes')
            unchanged = not changed
            mask, changed = remove_small_regions(mask,
                                                 min_area,
                                                 mode='islands')
            unchanged = unchanged and not changed

            new_masks.append(torch.as_tensor(mask).unsqueeze(0))
            # Give score=0 to changed masks and score=1 to unchanged masks
            # so NMS will prefer ones that didn't need postprocessing
            scores.append(float(unchanged))

        # Recalculate boxes and remove any new duplicates
        masks = torch.cat(new_masks, dim=0).to(we.device_id)
        boxes = batched_mask_to_box(masks)
        keep_by_nms = batched_nms(
            boxes.float(),
            torch.as_tensor(scores).to(we.device_id),
            torch.zeros_like(boxes[:, 0]),  # categories
            iou_threshold=nms_thresh,
        )

        # Only recalculate RLEs for masks that have changed
        for i_mask in keep_by_nms:
            if scores[i_mask] == 0.0:
                mask_torch = masks[i_mask].unsqueeze(0)
                rles[i_mask] = mask_to_rle_pytorch(mask_torch)
        masks = [rle_to_mask(rles[i][0]) for i in keep_by_nms]
        return masks

    def run_everything_ours(self, img_tensor, model):
        from segment_anything.utils.amg import mask_to_rle_pytorch
        img_tensor = img_tensor.squeeze(0)
        _, original_image_h, original_image_w = img_tensor.shape
        xy = []
        for i in range(self.GRID_SIZE):
            curr_x = 0.5 + i / self.GRID_SIZE * original_image_w
            for j in range(self.GRID_SIZE):
                curr_y = 0.5 + j / self.GRID_SIZE * original_image_h
                xy.append([curr_x, curr_y])

        xy = torch.from_numpy(np.array(xy))
        points = xy
        num_pts = xy.shape[0]
        point_labels = torch.ones(num_pts, 1)
        with torch.no_grad():
            predicted_masks, predicted_iou = self.get_predictions_given_embeddings_and_queries(
                img_tensor,
                points.reshape(1, num_pts, 1, 2).to(we.device_id),
                point_labels.reshape(1, num_pts, 1).to(we.device_id),
                model,
            )
        # print('predicted_masks: ', predicted_masks[0][0:1].dtype, predicted_masks[0][0:1].device)
        rle = [mask_to_rle_pytorch(m[0:1]) for m in predicted_masks]
        # transform to numpy
        size, counts = [], []
        for rle_item in rle:
            size.append(rle_item[0]['size'])
            counts += rle_item[0]['counts']
            counts += '#'
        predicted_masks = self.process_small_region(rle)
        return predicted_masks

    def forward(self, image, return_mask=None):
        return_mask = return_mask if return_mask is not None else self.return_mask
        if isinstance(image, Image.Image):
            image = np.array(image)
        elif isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        elif isinstance(image, np.ndarray):
            image = image.copy()
        else:
            raise f'Unsurpport datatype{type(image)}, only surpport np.ndarray, torch.Tensor, Pillow Image.'
        h, w = image.shape[:2]
        max_rate = max(float(w) / 1024.0, float(h) / 1024.0)
        w_ori = int(float(w) / max_rate)
        h_ori = int(float(h) / max_rate)
        # image = T.ToTensor()(T.Resize((h_ori, w_ori))(Image.fromarray(image)))
        # image_pad = T.Pad((0, 0, 1024 - w_ori, 1024 - h_ori))(image)
        image_pad = T.Pad((0, 0, 1024 - w_ori, 1024 - h_ori))(T.Resize(
            (h_ori, w_ori))(Image.fromarray(image)))
        input_image = T.ToTensor()(image_pad)
        input_image = input_image.unsqueeze(0).to(we.device_id)

        mask_efficient_sam_vits = self.run_everything_ours(
            input_image, self.efficient_sam_module)
        annos = []
        mask_efficient_sam_vits = sorted(list(mask_efficient_sam_vits),
                                         key=lambda m: int(m.sum()),
                                         reverse=True)
        mask_efficient_sam_vits = mask_efficient_sam_vits[:256]
        for mask in mask_efficient_sam_vits:
            mask_item = mask_utils.encode(
                np.array(mask[:, :, None], order='F', dtype='uint8'))[0]
            mask_item['counts'] = mask_item['counts'].decode('utf-8')
            mask_area = int(mask.sum())
            annos.append({'mask': mask_item, 'mask_area': mask_area})

        annos = sorted(annos, key=lambda x: x['mask_area'], reverse=True)
        seg_img = None
        dominant_palette = []
        image_pad_np = np.array(image_pad)
        for idx, anno in enumerate(annos):
            color = idx
            if idx > 255:
                break
            mask = np.array(mask_utils.decode(anno['mask'])).astype(np.uint8)
            h, w = mask.shape[:2]
            if seg_img is None:
                seg_img = np.ones((h, w, 3)) * 255
            if self.use_dominant_color:
                masked_image = cv2.bitwise_and(image_pad_np,
                                               image_pad_np,
                                               mask=mask)
                dominant_color = find_dominant_color(masked_image).tolist()
                dominant_palette.append(dominant_color)
            seg_img[mask.astype(bool)] = [color, color, color]
        seg_img = Image.fromarray(seg_img.astype(np.uint8)).convert('L')

        resize_rate = max(float(h_ori) / 1024.0, float(w_ori) / 1024.0)
        h_new = int(float(h_ori) / resize_rate)
        w_new = int(float(w_ori) / resize_rate)
        seg_img = seg_img.crop((0, 0, w_new, h_new))
        if self.save_mode == 'P':
            palette = []
            for i in range(256):
                if not self.use_dominant_color:
                    palette_item = [random.randint(0, 255) for _ in range(3)]
                else:
                    palette_item = dominant_palette[i] if i < len(
                        dominant_palette) else [255, 255, 255]
                palette += palette_item
            seg_img = seg_img.convert('P')
            seg_img.putpalette(palette)
            seg_rgb_img = seg_img.convert('RGB')
            if return_mask:
                return {
                    'image': np.array(seg_rgb_img),
                    'mask': np.array(seg_img)
                }
            else:
                return np.array(seg_rgb_img)
        else:
            return np.array(seg_img)

    @staticmethod
    def get_config_template():
        return dict_to_yaml('ANNOTATORS',
                            __class__.__name__,
                            ESAMAnnotator.para_dict,
                            set_name=True)




@ANNOTATORS.register_class()
class SAMAnnotatorDraw(BaseAnnotator, metaclass=ABCMeta):
    para_dict = {}

    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger=logger)
        from segment_anything import sam_model_registry, SamPredictor
        from segment_anything.utils.transforms import ResizeLongestSide

        self.transform = ResizeLongestSide(1024)
        self.task_type = cfg.get('TASK_TYPE', 'input_box')
        self.sam_model = cfg.get('SAM_MODEL', 'vit_b')
        pretrained_model = cfg.get('PRETRAINED_MODEL', 'sam_vit_b_01ec64.pth')

        if pretrained_model:
            with FS.get_from(pretrained_model, wait_finish=True) as local_path:
                seg_model = sam_model_registry[self.sam_model](checkpoint=local_path).eval().to(we.device_id)
                self.sam_predictor = SamPredictor(seg_model)

    def forward(self, image, input_box=None, mask=None, task_type=None, multimask_output=False):
        task_type = task_type if task_type is not None else self.task_type

        if isinstance(image, Image.Image):
            image = np.array(image)
        elif isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        elif isinstance(image, np.ndarray):
            image = image.copy()
        else:
            raise f'Unsurpport datatype{type(image)}, only surpport np.ndarray, torch.Tensor, Pillow Image.'

        if mask is not None:
            if isinstance(mask, Image.Image):
                mask = np.array(mask)
            elif isinstance(mask, torch.Tensor):
                mask = mask.detach().cpu().numpy()
            elif isinstance(mask, np.ndarray):
                mask = mask.copy()
            else:
                raise f'Unsurpport datatype{type(mask)}, only surpport np.ndarray, torch.Tensor, Pillow Image.'

        original_size = image.shape[:2]
        if task_type == 'mask_point':
            scribble = mask.transpose(2, 1, 0)[0]
            labeled_array, num_features = ndimage.label(scribble >= 255)
            centers = ndimage.center_of_mass(scribble, labeled_array, range(1, num_features + 1))
            point_coords = np.array(centers)
            point_labels = np.array([1] * len(centers))
            sample = {'point_coords': point_coords, 'point_labels': point_labels}

        elif task_type == 'mask_box':
            scribble = mask.transpose(2, 1, 0)[0]
            labeled_array, num_features = ndimage.label(scribble >= 255)
            centers = ndimage.center_of_mass(scribble, labeled_array, range(1, num_features + 1))
            centers = np.array(centers)
            ### (x1, y1, x2, y2)
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

        self.sam_predictor.set_image(image)
        masks, scores, logits = self.sam_predictor.predict(**sample, multimask_output=True)
        index = np.argmax(scores)

        ret_data = {
            "mask": (masks[index]* 255).astype(np.uint8),
            "score": scores[index]
        }
        return ret_data

    @staticmethod
    def get_config_template():
        return dict_to_yaml('ANNOTATORS',
                            __class__.__name__,
                            SAMAnnotatorDraw.para_dict,
                            set_name=True)