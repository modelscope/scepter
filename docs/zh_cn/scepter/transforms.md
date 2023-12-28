# 数据转换（Transforms）

数据预处理模块

## 总览

支持多种数据预处理方式：

1. scepter.transforms.image
2. scepter.transforms.io
3. scepter.transforms.io_video
4. scepter.transforms.tensor
5. scepter.transforms.augmention
6. scepter.transforms.video
7. scepter.transforms.transform_xl
8. scepter.transforms.identity
9. scepter.transforms.compose

<hr/>

## 基础用法

```python
# 以 scepter.modules.transform.image.RandomResizedCrop 为例
from scepter.modules.transform.image import RandomResizedCrop
from scepter.modules.utils.config import Config
import PIL

cfg = Config(load=False,
             cfg_dict={"SIZE": 224, "RATIO": [3. / 4., 4. / 3.], "SCALE": [0.08, 1.0], "INTERPOLATION": "bilinear"})
transform = RandomResizedCrop(cfg)
input_img = {"img": PIL.Image}
output_img = transform(input_img)
```

<hr/>

## **scepter.modules.transform.image**

一些用于图像的预处理方法.

<br>

### <font color="#0FB0E4">scepter.modules.transform.image.ImageTransform</font>

初始化ImageTransform类, 从cfg中获取并定义一些必要的参数.

**Parameters**

- **INPUT_KEY** —— (str) input key or key list
- **OUTPUT_KEY** —— (str) output key or key list
- **BACKEND** —— (str) backend, choose from pillow, cv2, torchvision

<br>

### <font color="#0FB0E4">scepter.modules.transform.image.RandomResizedCrop</font>

对图像进行随机crop到指定大小.

**Parameters**

- **SIZE** —— (int) crop size
- **RATIO** —— (list) ratio
- **SCALE** —— (list) scale
- **INTERPOLATION** —— (str) interpolation

<br>

### <font color="#0FB0E4">scepter.modules.transform.image.RandomHorizontalFlip</font>

以给定的概率随机水平翻转给定的图像.

**Parameters**

- **P** —— (float) probability

<br>

### <font color="#0FB0E4">scepter.modules.transform.image.Normalize</font>

利用平均值和标准差对图像进行归一化.

**Parameters**

- **MEAN** —— (list) mean
- **STD** —— (list) std

<br>

### <font color="#0FB0E4">scepter.modules.transform.image.ImageToTensor</font>

把PIL.Image / numpy.ndarray / unit8 转成float32 tensor.

**Parameters**

<br>

### <font color="#0FB0E4">scepter.modules.transform.image.Resize</font>

把给定的图像按照给定的尺寸进行resize.

**Parameters**

- **INTERPOLATION** —— (str) interpolation
- **SIZE** —— (int) resized size

<br>

### <font color="#0FB0E4">scepter.modules.transform.image.CenterCrop</font>

对给定的图像从中心进行crop.

**Parameters**

- **SIZE** —— (int) crop size

<br>

### <font color="#0FB0E4">scepter.modules.transform.image.FlexibleResize</font>

对给定的图像按照给定的尺寸进行resize.

**Parameters**

- **INTERPOLATION** —— (str) interpolation

<br>

### <font color="#0FB0E4">scepter.modules.transform.image.FlexibleCenterCrop</font>

对给定的图像按照给定的尺寸进行center crop.

**Parameters**

<br>

## **scepter.modules.transform.io**

一些用于图像的本地磁盘读取方法.

<br>

### <font color="#0FB0E4">scepter.modules.transform.io.LoadPILImageFromFile</font>

将本地图片文件读取成PIL.Image的形式.

**Parameters**

- **RGB_ORDER** —— (str) "RGB" or "BGR"

<br>

### <font color="#0FB0E4">scepter.modules.transform.io.LoadCvImageFromFile</font>

将本地图片文件读取成cv2的形式.

**Parameters**

- **RGB_ORDER** —— (str) "RGB" or "BGR"

<br>

### <font color="#0FB0E4">scepter.modules.transform.io.LoadImageFromFile</font>

将本地图片文件读取成指定格式.

**Parameters**

- **RGB_ORDER** —— (str) "RGB" or "BGR"
- **BACKEND** —— (str) "pillow" or "cv2" or "torchvision"

<br>

### <font color="#0FB0E4">scepter.modules.transform.io.LoadImageFromFileList</font>

将输入的一组图片读取成指定格式.

**Parameters**

- **RGB_ORDER** —— (str) "RGB" or "BGR"
- **BACKEND** —— (str) "pillow" or "cv2" or "torchvision"
- **FILE_KEYS** —— (list) The file keys for input

<br>

## **scepter.modules.transform.io_video**

一些用于视频的本地磁盘读取方法.

<br>

### <font color="#0FB0E4">scepter.modules.transform.io_video.DecodeVideoToTensor</font>

将本地视频文件解码成tensor.

**Parameters**

- **NUM_FRAMES** —— (int) decode frames number
- **TARGET_FPS** —— (int) decode frames fps, default is 30.
- **SAMPLE_MODE** —— (str) interval or segment sampling, default is interval
- **SAMPLE_INTERVAL** —— (int) sample interval between output frames for interval sample mode
- **SAMPLE_MINUS_INTERVAL** —— (float) wheather minus interval for interval sample mode
- **REPEAT** —— (str) number of clips to be decoded from each video

<br>

### <font color="#0FB0E4">scepter.modules.transform.io_video.LoadVideoFromFile</font>

将本地视频文件解码读取成帧序列的形式.

**Parameters**

- **NUM_FRAMES** —— (int) decode frames number
- **SAMPLE_TYPE** —— (str) sample type
- **CLIP_DURATION** —— (float) needed for 'interval' sampling type
- **DECODER** —— (str) video decoder name

<br>

## **scepter.modules.transform.tensor**

一些处理tensor的方法.

<br>

### <font color="#0FB0E4">scepter.modules.transform.tensor.ToTensor</font>

将输入的其他形式的data转成tensor.

**Parameters**

- **KEYS** —— (list) keys of input data

<br>

### <font color="#0FB0E4">scepter.modules.transform.tensor.Select</font>

选择输入data中的一些key并输出.

**Parameters**

- **META_KEYS** —— (list) chosen keys of input data

<br>

### <font color="#0FB0E4">scepter.modules.transform.tensor.Rename</font>

将输入data的keys重新命名.

**Parameters**

- **IN_KEYS** —— (list) input data keys
- **OUT_KEYS** —— (list) output data keys

<br>

## **scepter.modules.transform.augmention**

一些图片颜色增强的方法.

<br>

### <font color="#0FB0E4">scepter.modules.transform.augmention.ColorJitterGeneral</font>

随机改变图像的亮度、对比度和饱和度.

**Parameters**

- **BRIGHTNESS** —— (float) (float or tuple of float (min, max)): How much to jitter brightness
- **CONTRAST** —— (float) (float or tuple of float (min, max)): How much to jitter contrast
- **SATURATION** —— (float) (float or tuple of float (min, max)): How much to jitter saturation
- **HUE** —— (float) (float or tuple of float (min, max)): How much to jitter hue
- **GRAYSCALE** —— (float) probablitities for rgb-to-gray 0~1
- **CONSISTENT** —— (bool) for video input whether the augment scale is consistent or not
- **SHUFFLE** —— (bool) shuffle the transform's order when there are multiple transforms
- **GRAY_FIRST** —— (bool) whether use grayscale or not
- **IS_SPLIT** —— (bool) whether randomly chance the channel as the gray results

<br>

## **scepter.modules.transform.video**

一些处理video的方法.

<br>

### <font color="#0FB0E4">scepter.modules.transform.video.VideoTransform</font>

初始化VideoTransform类, 从cfg中获取并定义一些必要的参数.

**Parameters**

- **INPUT_KEY** —— (str) input key or key list
- **OUTPUT_KEY** —— (str) output key or key list
- **BACKEND** —— (str) backend, choose from pillow, cv2, torchvision

<br>

### <font color="#0FB0E4">scepter.modules.transform.video.RandomResizedCropVideo</font>

对视频进行随机crop到指定大小.

**Parameters**

- **META_KEYS** —— (list) chosen keys of input data

<br>

### <font color="#0FB0E4">scepter.modules.transform.video.CenterCropVideo</font>

将输入data的keys重新命名.

**Parameters**

- **SIZE** —— (int) crop size
- **RATIO** —— (list) ratio
- **SCALE** —— (list) scale
- **INTERPOLATION** —— (str) interpolation

<br>

### <font color="#0FB0E4">scepter.modules.transform.video.RandomHorizontalFlipVideo</font>

以给定的概率随机水平翻转给定的视频.

**Parameters**

- **P** —— (float) probability

<br>

### <font color="#0FB0E4">scepter.modules.transform.video.NormalizeVideo</font>

利用平均值和标准差对视频进行归一化.

**Parameters**

- **MEAN** —— (list) mean
- **STD** —— (list) std

<br>

### <font color="#0FB0E4">scepter.modules.transform.video.VideoToTensor</font>

把PIL.Image / numpy.ndarray / unit8 转成float32 tensor.

**Parameters**

<br>

### <font color="#0FB0E4">scepter.modules.transform.video.AutoResizedCropVideo</font>

对给定的视频从中心进行crop.

**Parameters**

- **SCALE** —— (list) scale

<br>

### <font color="#0FB0E4">scepter.modules.transform.video.ResizeVideo</font>

把给定的视频按照给定的尺寸进行resize.

**Parameters**

- **SCALE** —— (list) scale
- **INTERPOLATION** —— (str) interpolation

<br>

## **scepter.modules.transform.transform_xl**

sdxl中进行图像处理得到所需坐标的一些方法.

<br>

### <font color="#0FB0E4">scepter.modules.transform.transform_xl.FlexibleCropXL</font>

对图像进行裁剪，并获取其原始尺寸、目标尺寸和裁剪坐标（top/left）.

**Parameters**

- **INPUT_KEY** —— (str) input key or key list
- **OUTPUT_KEY** —— (str) output key or key list
- **BACKEND** —— (str) backend, choose from pillow, cv2, torchvision
- **SIZE** —— (int or list) crop size if 'image_size' not in meta

<br>

## **scepter.modules.transform.identity**

sdxl中进行图像处理得到所需坐标的一些方法.

<br>

### <font color="#0FB0E4">scepter.modules.transform.identity.Identity</font>

返回图像本身.

**Parameters**

<br>

## **scepter.modules.transform.compose**

组合各类transform方法.

<br>

### <font color="#0FB0E4">scepter.modules.transform.compose.Compose</font>

将scepter.transforms中的各个transform对象组合为pipeline.

**Parameters**

- **TRANSFORMS** —— (list) transform config list

<br>
