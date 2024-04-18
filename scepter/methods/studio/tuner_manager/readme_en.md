---
frameworks:
- Pytorch
license: Apache License 2.0
tasks:
- efficient-diffusion-tuning
---

<p align="center">

  <h2 align="center">{MODEL_NAME}</h2>
  <p align="center">
    <br>
        <a href="https://github.com/modelscope/scepter/"><img src="https://img.shields.io/badge/powered by-scepter-6FEBB9.svg"></a>
    <br>
  </p>

## Model Introduction
{MODEL_DESCRIPTION}

## Model Parameters
<table>
<thead>
  <tr>
    <th rowspan="2">Base Model</th>
    <th rowspan="2">Tuner Type</th>
    <th colspan="4">Training Parameters</th>
  </tr>
  <tr>
    <th>Batch Size</th>
    <th>Epochs</th>
    <th>Learning Rate</th>
    <th>Resolution</th>
  </tr>
</thead>
<tbody align="center">
  <tr>
    <td rowspan="8">{BASE_MODEL}</td>
    <td>{TUNER_TYPE}</td>
    <td>{TRAIN_BATCH_SIZE}</td>
    <td>{TRAIN_EPOCH}</td>
    <td>{LEARNING_RATE}</td>
    <td>[{HEIGHT}, {WIDTH}]</td>
  </tr>
</tbody>
</table>


<table>
<thead>
  <tr>
    <th>Data Type</th>
    <th>Data Space</th>
    <th>Data Name</th>
    <th>Data Subset</th>
  </tr>
</thead>
<tbody align="center">
  <tr>
    <td>{DATA_TYPE}</td>
    <td>{MS_DATA_SPACE}</td>
    <td>{MS_DATA_NAME}</td>
    <td>{MS_DATA_SUBNAME}</td>
  </tr>
</tbody>
</table>


## Model Performance
Given the input "{EVAL_PROMPT}," the following image may be generated:

![image]({IMAGE_PATH})

## Model Usage
### Command Line Execution
* Run using Scepter's SDK, taking care to use different configuration files in accordance with the different base models, as per the corresponding relationships shown below
<table>
<thead>
  <tr>
    <th rowspan="2">Base Model</th>
    <th rowspan="1">LORA</th>
    <th colspan="1">SCE</th>
    <th colspan="1">TEXT_LORA</th>
    <th colspan="1">TEXT_SCE</th>
  </tr>
</thead>
<tbody align="center">
  <tr>
    <td rowspan="8">SD1.5</td>
    <td><a href="https://github.com/modelscope/scepter/blob/main/scepter/methods/examples/generation/stable_diffusion_1.5_512_lora.yaml">lora_cfg</a></td>
    <td><a href="https://github.com/modelscope/scepter/blob/main/scepter/methods/scedit/t2i/sd15_512_sce_t2i_swift.yaml">sce_cfg</a></td>
    <td><a href="https://github.com/modelscope/scepter/blob/main/scepter/methods/examples/generation/stable_diffusion_1.5_512_text_lora.yaml">text_lora_cfg</a></td>
    <td><a href="https://github.com/modelscope/scepter/blob/main/scepter/methods/scedit/t2i/stable_diffusion_1.5_512_text_sce.yaml">text_sce_cfg</a></td>
  </tr>
</tbody>
<tbody align="center">
  <tr>
    <td rowspan="8">SD2.1</td>
    <td><a href="https://github.com/modelscope/scepter/blob/main/scepter/methods/examples/generation/stable_diffusion_2.1_768_lora.yaml">lora_cfg</a></td>
    <td><a href="https://github.com/modelscope/scepter/blob/main/scepter/methods/scedit/t2i/sd21_768_sce_t2i_swift.yaml">sce_cfg</a></td>
    <td><a href="https://github.com/modelscope/scepter/blob/main/scepter/methods/examples/generation/stable_diffusion_2.1_768_text_lora.yaml">text_lora_cfg</a></td>
    <td><a href="https://github.com/modelscope/scepter/blob/main/scepter/methods/scedit/t2i/sd21_768_text_sce_t2i_swift.yaml">text_sce_cfg</a></td>
  </tr>
</tbody>
<tbody align="center">
  <tr>
    <td rowspan="8">SDXL</td>
    <td><a href="https://github.com/modelscope/scepter/blob/main/scepter/methods/examples/generation/stable_diffusion_xl_1024_lora.yaml">lora_cfg</a></td>
    <td><a href="https://github.com/modelscope/scepter/blob/main/scepter/methods/scedit/t2i/sdxl_1024_sce_t2i_swift.yaml">sce_cfg</a></td>
    <td><a href="https://github.com/modelscope/scepter/blob/main/scepter/methods/examples/generation/stable_diffusion_xl_1024_text_lora.yaml">text_lora_cfg</a></td>
    <td><a href="https://github.com/modelscope/scepter/blob/main/scepter/methods/scedit/t2i/sdxl_1024_text_sce_t2i_swift.yaml">text_sce_cfg</a></td>
  </tr>
</tbody>
</table>

* Running from Source Code

```shell
git clone https://github.com/modelscope/scepter.git
cd scepter
pip install -r requirements/recommended.txt
PYTHONPATH=. python scepter/tools/run_inference.py
  --pretrained_model {this model folder}
  --cfg {lora_cfg} or {sce_cfg} or {text_lora_cfg} or {text_sce_cfg}
  --prompt '{EVAL_PROMPT}'
  --save_folder 'inference'
```

* Running after Installing Scepter (Recommended)
```shell
pip install scepter
python -m scepter/tools/run_inference.py
  --pretrained_model {this model folder}
  --cfg {lora_cfg} or {sce_cfg} or {text_lora_cfg} or {text_sce_cfg}
  --prompt '{EVAL_PROMPT}'
  --save_folder 'inference'
```
### Running with Scepter Studio

```shell
pip install scepter
# Launch Scepter Studio
python -m scepter.tools.webui
```

* Refer to the following guides for model usage.

(video url)

## Model Reference
If you wish to use this model for your own purposes, please cite it as follows.
```bibtex
@misc{{MODEL_NAME},
    title = {{MODEL_NAME}, {MODEL_URL}},
    author = {{USER_NAME}},
    year = {2024}
}
```
This model was trained using [Scepter Studio](https://github.com/modelscope/scepter); [Scepter](https://github.com/modelscope/scepter)
is an algorithm framework and toolbox developed by the Alibaba Tongyi Wanxiang Team. It provides a suite of tools and models for image generation, editing, fine-tuning, data processing, and more. If you find our work beneficial for your research,
please cite as follows.
```bibtex
@misc{scepter,
    title = {SCEPTER, https://github.com/modelscope/scepter},
    author = {SCEPTER},
    year = {2023}
}
```
