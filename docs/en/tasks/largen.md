<h1 align="center"> Locate, Assign, Refine: Taming Customized Image Inpainting with Text-Subject Guidance </h1>

<p align="center">
    <strong>Yulin Pan</strong>
    ·
    <strong>Chaojie Mao</strong>
    ·
    <strong>Zeyinzi Jiang</strong>
    ·
    <strong>Zhen Han</strong>
    ·
    <strong>Jingfeng Zhang</strong>
    <br>
    <a href="https://arxiv.org/abs/2403.19534"><img src="https://img.shields.io/static/v1?label=arXiv&message=LARGen&color=red&logo=arxiv"></a>
    <a href="https://ali-vilab.github.io/largen-page/"><img src="https://img.shields.io/badge/Page-LARGen-Gree"></a>
</p>

LARGen is a unified image inpainting framework that supports text-guided, subject-guided and text-subject-guided inpainting simutaneously.
Four LARGen-based fantastic applications are now supported by SCEPTER Studio:
1. Zoom Out
2. Virtual Try On
3. Text-Guided Inpainting
4. Text-Subject-Guided Inpainting

## Basic Usage

Here's a demo showcasing the use of LARGen-based functions.
<p align="left">
<img src="https://raw.githubusercontent.com/ali-vilab/largen-page/main/public/images/largen.gif" width="1300">
</p>

## Gallery

### LAR-Gen: Zoom Out
<table>
  <tr>
    <td><strong>Origin Image</strong><br>Prompt: a temple on fire</td>
    <td><strong>Zoom-Out</strong><br>CenterAround:0.75</td>
    <td><strong>Zoom-Out</strong><br>CenterAround:0.75</td>
    <td><strong>Zoom-Out</strong><br>CenterAround:0.75</td>
    <td><strong>Zoom-Out</strong><br>CenterAround:0.75</td>
  </tr>
  <tr>
    <td><img src="../../../asset/images/zoom_out/ex1_scene_im.jpg" width="240"></td>
    <td><img src="../../../asset/images/zoom_out/ex1_zoom_out1.jpg" width="240"></td>
    <td><img src="../../../asset/images/zoom_out/ex1_zoom_out2.jpg" width="240"></td>
    <td><img src="../../../asset/images/zoom_out/ex1_zoom_out3.jpg" width="240"></td>
    <td><img src="../../../asset/images/zoom_out/ex1_zoom_out4.jpg" width="240"></td>
  </tr>
</table>

### LAR-Gen: Virtual Try-on
<table>
  <tr>
    <td><strong>Model Image</strong></td>
    <td><strong>Model Mask</strong></td>
    <td><strong>Clothing Image</strong></td>
    <td><strong>Clothing Mask</strong></td>
    <td><strong>Try-on Output</strong></td>
  </tr>
  <tr>
    <td><img src="../../../asset/images/virtual_try_on/model.jpg" width="240"></td>
    <td><img src="../../../asset/images/virtual_try_on/ex2_scene_mask.jpg" width="240"></td>
    <td><img src="../../../asset/images/virtual_try_on/tshirt.jpg" width="240"></td>
    <td><img src="../../../asset/images/virtual_try_on/ex2_subject_mask.jpg" width="240"></td>
    <td><img src="../../../asset/images/virtual_try_on/try_on_out.jpg" width="240"></td>
  </tr>
</table>

### LAR-Gen: Inpainting (Text guided)
<table>
  <tr>
    <td><strong>Origin Image</strong><br>Prompt: a blue and white porcelain</td>
    <td><strong>Inpainting Mask1</strong></td>
    <td><strong>Inpainting Output1</strong></td>
    <td><strong>Inpainting Mask2</strong><br>Prompt: a clock</td>
    <td><strong>Inpainting Output2</strong></td>
  </tr>
  <tr>
    <td><img src="../../../asset/images/inpainting_text/ex3_scene_im.jpg" width="240"></td>
    <td><img src="../../../asset/images/inpainting_text/ex3_scene_mask.jpg" width="240"></td>
    <td><img src="../../../asset/images/inpainting_text/inpainting_text.jpg" width="240"></td>
    <td><img src="../../../asset/images/inpainting_text/ex3_scene_mask2.jpg" width="240"></td>
    <td><img src="../../../asset/images/inpainting_text/inpainting_text2.jpg" width="240"></td>
  </tr>
</table>

### LAR-Gen: Inpainting (Text and Subject guided)
<table>
  <tr>
    <td><strong>Origin Image</strong><br>Prompt: a dog wearing sunglasses</td>
    <td><strong>Origin Mask</strong></td>
    <td><strong>Reference Image</strong></td>
    <td><strong>Reference Mask</strong></td>
    <td><strong>Inpainting Output</strong></td>
  </tr>
  <tr>
    <td><img src="../../../asset/images/inpainting_text_ref/ex4_scene_im.jpg" width="240"></td>
    <td><img src="../../../asset/images/inpainting_text_ref/ex4_scene_mask.jpg" width="240"></td>
    <td><img src="../../../asset/images/inpainting_text_ref/ex4_subject_im.jpg" width="240"></td>
    <td><img src="../../../asset/images/inpainting_text_ref/ex4_subject_mask.jpg" width="240"></td>
    <td><img src="../../../asset/images/inpainting_text_ref/inpainting_text_ref.jpg" width="240"></td>
  </tr>
</table>

## Features

| **Model** | **Locate** | **Assign** | **Refine** |
|:---------:|:----------:|:----------:|:----------:|
|   SD v1.5 |     ⏳     |    ⏳       |     ⏳     |
|   SD XL   |     🪄     |    🪄       |     ⏳     |

- 🪄 denotes that the feature has been supported.
- ⏳ denotes that the feature has not been integrated currently.


## Pretrained Models

| **Model**  | **URL** |
|:----------:|:-------:|
| largen-sdxl-s22k | [ModelScope](https://www.modelscope.cn/models/iic/LARGEN/summary)  |


## BibTeX
If our work is useful for your research, please consider citing:
```bibtex
@article{pan2024locate,
  title={Locate, Assign, Refine: Taming Customized Image Inpainting with Text-Subject Guidance},
  author={Pan, Yulin and Mao, Chaojie and Jiang, Zeyinzi and Han, Zhen and Zhang, Jingfeng},
  journal={arXiv preprint arXiv:2403.19534},
  year={2024}
}
```
