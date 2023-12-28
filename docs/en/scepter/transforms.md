# Transforms

Data pre-processing module

## Overview

Supports various data pre-processing methods：

1. ***scepter.transforms.image***
2. ***scepter.transforms.io***
3. ***scepter.transforms.io_video***
4. ***scepter.transforms.tensor***
5. ***scepter.transforms.augmention***
6. ***scepter.transforms.video***
7. ***scepter.transforms.transform_xl***
8. ***scepter.transforms.identity***
9. ***scepter.transforms.compose***

<hr/>

## Basic Usage

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

Some pre-processing methods used for images.

<br>

### <font color="#0FB0E4">scepter.modules.transform.image.ImageTransform</font>

Initialize the ***ImageTransform*** class, obtain and define some necessary parameters from ***cfg***.

**Parameters**

- **INPUT_KEY** —— (str) input key or key list
- **OUTPUT_KEY** —— (str) output key or key list
- **BACKEND** —— (str) backend, choose from pillow, cv2, torchvision

<br>

### <font color="#0FB0E4">scepter.modules.transform.image.RandomResizedCrop</font>

Randomly crop the image to a specified size.

**Parameters**

- **SIZE** —— (int) crop size
- **RATIO** —— (list) ratio
- **SCALE** —— (list) scale
- **INTERPOLATION** —— (str) interpolation

<br>

### <font color="#0FB0E4">scepter.modules.transform.image.RandomHorizontalFlip</font>

Randomly horizontally flip the given image with a given probability(***P***).

**Parameters**

- **P** —— (float) probability

<br>

### <font color="#0FB0E4">scepter.modules.transform.image.Normalize</font>

Normalize the image using ***mean*** and standard deviation(***std***).

**Parameters**

- **MEAN** —— (list) mean
- **STD** —— (list) std

<br>

### <font color="#0FB0E4">scepter.modules.transform.image.ImageToTensor</font>

transform ***PIL.Image / numpy.ndarray / unit8*** to ***float32 tensor***.

**Parameters**

<br>

### <font color="#0FB0E4">scepter.modules.transform.image.Resize</font>

Resize the given image to the given ***Size***.

**Parameters**

- **INTERPOLATION** —— (str) interpolation
- **SIZE** —— (int) resized size

<br>

### <font color="#0FB0E4">scepter.modules.transform.image.CenterCrop</font>

Crop the given image from the center.

**Parameters**

- **SIZE** —— (int) crop size

<br>

### <font color="#0FB0E4">scepter.modules.transform.image.FlexibleResize</font>

Resize the given image to the given ***Size***.

**Parameters**

- **INTERPOLATION** —— (str) interpolation

<br>

### <font color="#0FB0E4">scepter.modules.transform.image.FlexibleCenterCrop</font>

Center crop the given image to the given ***Size***.

**Parameters**

<br>

## **scepter.modules.transform.io**

Some methods for reading images from local disk.

<br>

### <font color="#0FB0E4">scepter.modules.transform.io.LoadPILImageFromFile</font>

Read a local image file into ***PIL.Image*** format.

**Parameters**

- **RGB_ORDER** —— (str) "RGB" or "BGR"

<br>

### <font color="#0FB0E4">scepter.modules.transform.io.LoadCvImageFromFile</font>

Read a local image file into ***cv2*** format.

**Parameters**

- **RGB_ORDER** —— (str) "RGB" or "BGR"

<br>

### <font color="#0FB0E4">scepter.modules.transform.io.LoadImageFromFile</font>

Read a local image file into a specific format.

**Parameters**

- **RGB_ORDER** —— (str) "RGB" or "BGR"
- **BACKEND** —— (str) "pillow" or "cv2" or "torchvision"

<br>

### <font color="#0FB0E4">scepter.modules.transform.io.LoadImageFromFileList</font>

Read a set of input images into a specific format.

**Parameters**

- **RGB_ORDER** —— (str) "RGB" or "BGR"
- **BACKEND** —— (str) "pillow" or "cv2" or "torchvision"
- **FILE_KEYS** —— (list) The file keys for input

<br>

## **scepter.modules.transform.io_video**

Some methods for reading videos from local disk.

<br>

### <font color="#0FB0E4">scepter.modules.transform.io_video.DecodeVideoToTensor</font>

Decode local video files into tensors.

**Parameters**

- **NUM_FRAMES** —— (int) decode frames number
- **TARGET_FPS** —— (int) decode frames fps, default is 30.
- **SAMPLE_MODE** —— (str) interval or segment sampling, default is interval
- **SAMPLE_INTERVAL** —— (int) sample interval between output frames for interval sample mode
- **SAMPLE_MINUS_INTERVAL** —— (float) wheather minus interval for interval sample mode
- **REPEAT** —— (str) number of clips to be decoded from each video

<br>

### <font color="#0FB0E4">scepter.modules.transform.io_video.LoadVideoFromFile</font>

Decode and read local video files into a sequence of frames.

**Parameters**

- **NUM_FRAMES** —— (int) decode frames number
- **SAMPLE_TYPE** —— (str) sample type
- **CLIP_DURATION** —— (float) needed for 'interval' sampling type
- **DECODER** —— (str) video decoder name

<br>

## **scepter.modules.transform.tensor**

Some methods for processing tensors.

<br>

### <font color="#0FB0E4">scepter.modules.transform.tensor.ToTensor</font>

Convert input data from other formats into tensors.

**Parameters**

- **KEYS** —— (list) keys of input data

<br>

### <font color="#0FB0E4">scepter.modules.transform.tensor.Select</font>

Select some keys from the input data and output them.

**Parameters**

- **META_KEYS** —— (list) chosen keys of input data

<br>

### <font color="#0FB0E4">scepter.modules.transform.tensor.Rename</font>

Rename the keys of the input data.

**Parameters**

- **IN_KEYS** —— (list) input data keys
- **OUT_KEYS** —— (list) output data keys

<br>

## **scepter.modules.transform.augmention**

Some methods for enhancing the colors in images.

<br>

### <font color="#0FB0E4">scepter.modules.transform.augmention.ColorJitterGeneral</font>

Randomly adjust the brightness, contrast, and saturation of an image.

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

Some methods for processing videos.

<br>

### <font color="#0FB0E4">scepter.modules.transform.video.VideoTransform</font>

To initialize a ***VideoTransform*** class and define the necessary parameters from a ***cfg***.

**Parameters**

- **INPUT_KEY** —— (str) input key or key list
- **OUTPUT_KEY** —— (str) output key or key list
- **BACKEND** —— (str) backend, choose from pillow, cv2, torchvision

<br>

### <font color="#0FB0E4">scepter.modules.transform.video.RandomResizedCropVideo</font>

Perform random cropping of the video to a specified size.

**Parameters**

- **META_KEYS** —— (list) chosen keys of input data

<br>

### <font color="#0FB0E4">scepter.modules.transform.video.CenterCropVideo</font>

Renaming the keys of the input data.

**Parameters**

- **SIZE** —— (int) crop size
- **RATIO** —— (list) ratio
- **SCALE** —— (list) scale
- **INTERPOLATION** —— (str) interpolation

<br>

### <font color="#0FB0E4">scepter.modules.transform.video.RandomHorizontalFlipVideo</font>

Randomly flip the given video horizontally with a given probability.

**Parameters**

- **P** —— (float) probability

<br>

### <font color="#0FB0E4">scepter.modules.transform.video.NormalizeVideo</font>

Normalize the video using the ***mean*** and standard deviation(***std***).

**Parameters**

- **MEAN** —— (list) mean
- **STD** —— (list) std

<br>

### <font color="#0FB0E4">scepter.modules.transform.video.VideoToTensor</font>

transform ***PIL.Image / numpy.ndarray / unit8*** to ***float32 tensor***.

**Parameters**

<br>

### <font color="#0FB0E4">scepter.modules.transform.video.AutoResizedCropVideo</font>

Crop the given video from the center.

**Parameters**

- **SCALE** —— (list) scale

<br>

### <font color="#0FB0E4">scepter.modules.transform.video.ResizeVideo</font>

Resize the given video to the specified dimensions.

**Parameters**

- **SCALE** —— (list) scale
- **INTERPOLATION** —— (str) interpolation

<br>

## **scepter.modules.transform.transform_xl**

Some methods for image processing in SDXL to obtain the desired coordinates.

<br>

### <font color="#0FB0E4">scepter.modules.transform.transform_xl.FlexibleCropXL</font>

Crop an image and obtain its original size, target size, and cropping coordinates (top/left).

**Parameters**

- **INPUT_KEY** —— (str) input key or key list
- **OUTPUT_KEY** —— (str) output key or key list
- **BACKEND** —— (str) backend, choose from pillow, cv2, torchvision
- **SIZE** —— (int or list) crop size if 'image_size' not in meta

<br>

## **scepter.modules.transform.identity**

Some methods to process images and obtain the required coordinates in SDXL.

<br>

### <font color="#0FB0E4">scepter.modules.transform.identity.Identity</font>

Return the image itself.

**Parameters**

<br>

## **scepter.modules.transform.compose**

Combine various transform methods.

<br>

### <font color="#0FB0E4">scepter.modules.transform.compose.Compose</font>

Combine the various transform objects from ***scepter.transforms*** into a pipeline.

**Parameters**

- **TRANSFORMS** —— (list) transform config list

<br>
