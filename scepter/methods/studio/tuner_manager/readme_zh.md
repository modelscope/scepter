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

## 模型介绍
{MODEL_DESCRIPTION}

## 模型参数
<table>
<thead>
  <tr>
    <th rowspan="2">基础模型</th>
    <th rowspan="2">微调类型</th>
    <th colspan="4">训练参数</th>
  </tr>
  <tr>
    <th>批次大小</th>
    <th>轮数</th>
    <th>学习率</th>
    <th>分辨率</th>
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
    <th>数据类型</th>
    <th>数据空间</th>
    <th>数据名称</th>
    <th>数据子集</th>
  </tr>
</thead>
<tbody align="center">
  <tr>
    <td> {DATA_TYPE}</td>
    <td>{MS_DATA_SPACE}</td>
    <td>{MS_DATA_NAME}</td>
    <td>{MS_DATA_SUBNAME}</td>
  </tr>
</tbody>
</table>


## 模型效果

输入 "{EVAL_PROMPT}"，可能会得到如下图像：

![image]({IMAGE_PATH})


## 模型使用
### 命令行运行

* 使用scepter的sdk进行运行，注意需要按照模型参数中基模型的不同使用不同的配置文件，其对应关系如下
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

* 从源码运行

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

* 安装scepter后运行（推荐）
```shell
pip install scepter
python -m scepter/tools/run_inference.py
  --pretrained_model {this model folder}
  --cfg {lora_cfg} or {sce_cfg} or {text_lora_cfg} or {text_sce_cfg}
  --prompt '{EVAL_PROMPT}'
  --save_folder 'inference'
```
### 使用Scepter Studio运行
```shell
pip install scepter
启动scepter studio
python -m scepter.tools.webui
```
* 参考以下指南使用模型


## 模型引用
如果你想使用该模型应用于自己的场景，请按照如下方式引用该模型。
```bibtex
@misc{{MODEL_NAME},
    title = {{MODEL_NAME}, {MODEL_URL}},
    author = {{USER_NAME}},
    year = {2024}
}
```
该模型是基于[Scepter Studio](https://github.com/modelscope/scepter)训练得到；[scepter](https://github.com/modelscope/scepter)
是由阿里巴巴通义万相团队开发的算法框架和工具箱，提供图像生成、编辑、微调、数据处理等一系列工具和模型。如果您觉得我们的工作有益于您的工作，
请按照如下方式引用。
```bibtex
@misc{scepter,
    title = {SCEPTER, https://github.com/modelscope/scepter},
    author = {SCEPTER},
    year = {2023}
}
```
