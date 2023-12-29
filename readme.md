<h1 align="center">🪄SCEPTER</h1>

<p align="center">
<img src="https://img.shields.io/badge/python-%E2%89%A53.8-5be.svg">
<img src="https://img.shields.io/badge/pytorch-%E2%89%A51.12%20%7C%20%E2%89%A52.0-orange.svg">
<a href="https://github.com/modelscope/scepter/"><img src="https://img.shields.io/badge/scepter-Build from source-6FEBB9.svg"></a>
</p>

## 📖 Table of Contents
- [Introduction](#-introduction)
- [News](#-news)
- [Installation](#-installation)
- [Getting Started](#-getting-started)
- [Learn More](#-learn-more)
- [License](#license)

## 📝 Introduction

SCEPTER is an open-source code repository dedicated to generative training, fine-tuning, and inference, encompassing a suite of downstream tasks such as image generation, transfer, editing. It integrates popular community-driven implementations as well as proprietary methods by Tongyi Lab of Alibaba Group, offering a comprehensive toolkit for researchers and practitioners in the field of AIGC. This versatile library is designed to facilitate innovation and accelerate development in the rapidly evolving domain of generative models.

Main Feature:

- Training:
  - distribute: DDP / FSDP / FairScale
- Inference
  - text-to-image generation
  - controllable image synthesis (TODO)
- Deploy-Gradio (TODO)
  - fine-tuning
  - inference

Currently supported approches (and counting):

1. SD Series: [Stable Diffusion v1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5) / [Stable Diffusion v2.1](https://huggingface.co/runwayml/stable-diffusion-v1-5) / [Stable Diffusion XL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
2. SCEdit: [SCEdit: Efficient and Controllable Image Diffusion Generation via Skip Connection Editing](https://arxiv.org/abs/2312.11392)  [![Arxiv link](https://img.shields.io/static/v1?label=arXiv&message=SCEdit&color=red&logo=arxiv)](https://arxiv.org/abs/2312.11392) [![Page link](https://img.shields.io/badge/Page-SCEdit-Gree)](https://scedit.github.io/)
3. Res-Tuning(TODO): [Res-Tuning: A Flexible and Efficient Tuning Paradigm via Unbinding Tuner from Backbone](https://arxiv.org/abs/2310.19859) [![Arxiv link](https://img.shields.io/static/v1?label=arXiv&message=ResTuning&color=red&logo=arxiv)](https://arxiv.org/abs/2310.19859) [![Page link](https://img.shields.io/badge/Page-ResTuning-Gree)](https://res-tuning.github.io/)

## 🎉 News
- [2023.12]: We propose [SCEdit](https://arxiv.org/abs/2312.11392), an efficient and controllable generation framework.
- [2023.12]: We release [🪄SCEPTER](https://github.com/modelscope/scepter/) library.

## 🛠️ Installation

- Create new environment

```shell
conda env create -f environment.yaml
conda activate scepter
```

- Install SCEPTER by the `pip` command:

```shell
pip install -e .
```

## 🚀 Getting Started

### Dataset

#### Text-to-Image generation

We use a [custom-stylized dataset](https://modelscope.cn/datasets/damo/style_custom_dataset/summary), which included classes 3D, anime, flat illustration, oil painting, sketch, and watercolor, each with 30 image-text pairs.

```python
# pip install modelscope
from modelscope.msdatasets import MsDataset
ms_train_dataset = MsDataset.load('style_custom_dataset', namespace='damo', subset_name='3D', split='train_short')
print(next(iter(ms_train_dataset)))
```

### Training

#### Text-to-Image generation

- SCEdit

```python
# SD v1.5
python scepter/tools/run_train.py --cfg scepter/methods/SCEdit/t2i_sd15_512_sce.yaml 
# SD v2.1
python scepter/tools/run_train.py --cfg scepter/methods/SCEdit/t2i_sd21_768_sce.yaml
# SD XL
python scepter/tools/run_train.py --cfg scepter/methods/SCEdit/t2i_sdxl_1024_sce.yaml
```

- Existing strategies
```python
# fully-tuning on SD v1.5
python scepter/tools/run_train.py --cfg scepter/methods/examples/generation/stable_diffusion_1.5_512.yaml
# lora-tuning on SD v2.1
python scepter/tools/run_train.py --cfg scepter/methods/examples/generation/stable_diffusion_2.1_768_lora.yaml 
```
#### Controllable Image Synthesis
  
TODO

### Inference

```python
# generation on SD v1.5
python scepter/tools/run_inference.py --cfg scepter/methods/examples/generation/stable_diffusion_1.5_512.yaml --prompt 'a cute dog' --save_folder 'inference'
# generation on SD v2.1
python scepter/tools/run_inference.py --cfg scepter/methods/examples/generation/stable_diffusion_2.1_768.yaml --prompt 'a cute dog' --save_folder 'inference'
# generation on SD XL
python scepter/tools/run_inference.py --cfg scepter/methods/examples/generation/stable_diffusion_xl_1024.yaml --prompt 'a cute dog' --save_folder 'inference'
```


## 🔍 Learn More

- [ModelScope library](https://github.com/modelscope/modelscope/)

  ModelScope Library is the model library of ModelScope project, which contains a large number of popular models.

- [Alibaba TongYi Vision Intelligence Lab](https://github.com/damo-vilab)

  Discover more about open-source projects on image generation, video generation, and editing tasks.

## License

This project is licensed under the [Apache License (Version 2.0)](https://github.com/modelscope/modelscope/blob/master/LICENSE).
