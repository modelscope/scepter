# Training

We provide a framework for training and validation.

The scripts below are just for illustration purposes. To achieve better results, you can modify the corresponding parameters as needed.

## Start Training
There are different ways to start a training:

- calling scepter/tools/run_train.py:
```bash
# calling at SCEPTER root:
PYTHONPATH=./ python scepter/tools/run_train.py --cfg [path-to-your-yaml]

# calling scepter library:
pip install scepter
python -m scepter.tools.run_train --cfg [path-to-your-yaml]
```
- calling your own script:
```bash
# calling at SCEPTER root:
PYTHONPATH=./ python [path-to-your-script] --cfg [path-to-your-yaml]

# calling scepter library:
pip install scepter
python [path-to-your-script] --cfg [path-to-your-yaml]
```
your scepter should be like:
```python
from scepter.tools.run_train import run

if __name__ == '__main__':
    run()
```

## Popular Tasks
### Text-to-Image Generation

- SCEdit
```bash
python scepter/tools/run_train.py --cfg scepter/methods/scedit/t2i/sd15_512_sce_t2i.yaml  # SD v1.5
python scepter/tools/run_train.py --cfg scepter/methods/scedit/t2i/sd21_768_sce_t2i.yaml  # SD v2.1
python scepter/tools/run_train.py --cfg scepter/methods/scedit/t2i/sdxl_1024_sce_t2i.yaml  # SD XL
```

- Existing Tuning Strategies
```bash
python scepter/tools/run_train.py --cfg scepter/methods/examples/generation/stable_diffusion_1.5_512.yaml  # fully-tuning on SD v1.5
python scepter/tools/run_train.py --cfg scepter/methods/examples/generation/stable_diffusion_2.1_768_lora.yaml  # lora-tuning on SD v2.1
```

- Data Text Format
```bash
# Download the 3D_example_txt.zip as previously mentioned
python scepter/tools/run_train.py --cfg scepter/methods/scedit/t2i/sdxl_1024_sce_t2i_datatxt.yaml
```

### Controllable Image Synthesis

- SCEdit

The YAML configuration can be modified to combine different base models and conditions. The following is provided as an example.
```bash
python scepter/tools/run_train.py --cfg scepter/methods/scedit/ctr/sd15_512_sce_ctr_hed.yaml  # SD v1.5 + hed
python scepter/tools/run_train.py --cfg scepter/methods/scedit/ctr/sd21_768_sce_ctr_canny.yaml  # SD v2.1 + canny
python scepter/tools/run_train.py --cfg scepter/methods/scedit/ctr/sd21_768_sce_ctr_pose.yaml  # SD v2.1 + pose
python scepter/tools/run_train.py --cfg scepter/methods/scedit/ctr/sdxl_1024_sce_ctr_depth.yaml  # SD XL + depth
python scepter/tools/run_train.py --cfg scepter/methods/scedit/ctr/sdxl_1024_sce_ctr_color.yaml  # SD XL + color
```

- Data Text Format
```bash
# Download the 3D_example_txt.zip as previously mentioned
python scepter/tools/run_train.py --cfg scepter/methods/scedit/ctr/sdxl_1024_sce_ctr_color_datatxt.yaml
```


## Customize Modules
You can register your own Modules like DATASET, SAMPLERS, TRANSFORMS, MODELS, SOVLERS, HOOKS, OPTIMIZERS into SCEPTER.
Refer to `example/`, build the modules of your task in `example/{task}`.
```bash
cd example/classifier
python run.py --cfg classifier.yaml
```
