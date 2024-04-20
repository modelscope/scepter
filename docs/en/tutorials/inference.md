# Inference

In this tutorial, we'll cover the use of the scepter framework for convenient inference, including inference using the command line or specific method classes, and we'll give examples of inference methods for additional tasks.

## Command Line
Inference of SDXL generation models using the command line.
```shell
python scepter/tools/run_inference.py --cfg scepter/methods/examples/generation/stable_diffusion_xl_1024.yaml --prompt 'a cute dog' --save_folder 'inference'  # generation on SD XL
```

## Class Instantiation
Inference of SD2.1 generation models using the class instantiation.
```python
from torchvision.utils import save_image
from scepter.modules.utils.config import Config
from scepter.modules.utils.file_system import FS
from scepter.modules.utils.logger import get_logger
from scepter.modules.inference.diffusion_inference import DiffusionInference
# init file system - modelscope
FS.init_fs_client(Config(load=False, cfg_dict={'NAME': 'ModelscopeFs', 'TEMP_DIR': 'cache/data'}))
# init model config
logger = get_logger(name='scepter')
cfg = Config(cfg_file='scepter/methods/studio/inference/stable_diffusion/sd21_pro.yaml')
diff_infer = DiffusionInference(logger)
diff_infer.init_from_cfg(cfg)
# start inference
output = diff_infer({'prompt': 'a cute dog'})
save_image(output['images'], 'sd21_test_prompt_a_cute_dog.png')
```

## Additional Tasks

### Fine-tuned Model Inference

```shell
python scepter/tools/run_inference.py --cfg scepter/methods/scedit/t2i/sd15_512_sce_t2i_swift.yaml --pretrained_model 'cache/save_data/sd15_512_sce_t2i_swift/checkpoints/ldm_step-100.pth' --prompt 'A close up of a small rabbit wearing a hat and scarf' --save_folder 'trained_test_prompt_rabbit'
```

### Controllable Image Synthesis Inference

- SCEdit
```shell
python scepter/tools/run_inference.py --cfg scepter/methods/scedit/ctr/sd21_768_sce_ctr_canny.yaml --num_samples 1 --prompt 'a single flower is shown in front of a tree' --save_folder 'test_flower_canny' --image_size 768 --task control --image 'asset/images/flower.jpg' --control_mode canny --pretrained_model ms://damo/scepter_scedit@controllable_model/SD2.1/canny_control/0_SwiftSCETuning/pytorch_model.bin   # canny
python scepter/tools/run_inference.py --cfg scepter/methods/scedit/ctr/sd21_768_sce_ctr_pose.yaml --num_samples 1 --prompt 'super mario' --save_folder 'test_mario_pose' --image_size 768 --task control --image 'asset/images/pose_source.png' --control_mode source --pretrained_model ms://damo/scepter_scedit@controllable_model/SD2.1/pose_control/0_SwiftSCETuning/pytorch_model.bin   # pose
```

