from abc import ABCMeta

import torch
import cv2
import numpy as np
from PIL import Image
from scepter.modules.utils.distribute import we
from scepter.modules.utils.file_system import FS
from scepter.modules.utils.config import dict_to_yaml
from scepter.modules.annotator.base_annotator import BaseAnnotator
from scepter.modules.annotator.registry import ANNOTATORS

def dilate_mask(mask, dilate_factor=15):
    mask = mask.astype(np.uint8)
    mask = cv2.dilate(
        mask,
        np.ones((dilate_factor, dilate_factor), np.uint8),
        iterations=1
    )
    return mask

@ANNOTATORS.register_class()
class LamaAnnotator(BaseAnnotator, metaclass=ABCMeta):
    para_dict = {}
    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger=logger)
        from modelscope.pipelines.builder import PIPELINES
        from modelscope.pipelines.cv import ImageInpaintingPipeline
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks
        from modelscope.metainfo import Pipelines
        from modelscope.models.cv.image_inpainting.refinement import refine_predict
        from torch.utils.data._utils.collate import default_collate

        @PIPELINES.register_module(Tasks.image_inpainting, module_name=Pipelines.image_inpainting + "-v2")
        class ImageInpaintingPipelineV2(ImageInpaintingPipeline):
            def perform_inference(self, data):
                px_budget = 9000000
                batch = default_collate([data])
                if self.refine:
                    assert 'unpad_to_size' in batch, 'Unpadded size is required for the refinement'
                    assert 'cuda' in str(self.device), 'GPU is required for refinement'
                    gpu_ids = str(self.device).split(':')[-1]
                    cur_res = refine_predict(
                        batch,
                        self.infer_model,
                        gpu_ids=gpu_ids,
                        modulo=self.pad_out_to_modulo,
                        n_iters=15,
                        lr=0.002,
                        min_side=512,
                        max_scales=3,
                        px_budget=px_budget)
                    cur_res = cur_res[0].permute(1, 2, 0).detach().cpu().numpy()
                else:
                    with torch.no_grad():
                        batch = self.move_to_device(batch, self.device)
                        batch['mask'] = (batch['mask'] > 0) * 1
                        batch = self.infer_model(batch)
                        cur_res = batch['inpainted'][0].permute(
                            1, 2, 0).detach().cpu().numpy()
                        unpad_to_size = batch.get('unpad_to_size', None)
                        if unpad_to_size is not None:
                            orig_height, orig_width = unpad_to_size
                            cur_res = cur_res[:orig_height, :orig_width]

                cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
                cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
                return cur_res

        lama_model_dir = FS.get_dir_to_local_dir(cfg.PRETRAINED_MODEL)
        self.lama_model = pipeline(Tasks.image_inpainting, model=lama_model_dir,
                              pipeline_name=Pipelines.image_inpainting + "-v2", refine=True,
                              device="cuda:{}".format(we.device_id))
    def forward(self, image, mask):
        mask = dilate_mask(mask, dilate_factor=19)
        input_mask = Image.fromarray(mask)
        mask_expanded = np.tile(np.expand_dims(mask, axis=-1), (1, 1, 3))
        input_image_np = np.array(image)
        input_image_np[mask_expanded == 255] = 0
        input_image = Image.fromarray(input_image_np)
        input = {
            'img': input_image,
            'mask': input_mask,
        }
        result = self.lama_model(input)
        output_img = result['output_img']
        return output_img[..., ::-1]

    @staticmethod
    def get_config_template():
        return dict_to_yaml('ANNOTATORS',
                            __class__.__name__,
                            LamaAnnotator.para_dict,
                            set_name=True)



