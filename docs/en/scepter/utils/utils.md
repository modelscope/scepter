# Dependency Components (Utils)

Relies on SDKs, which are used to organize modules and SDKs that are frequently reused throughout the framework and to aggregate them based on functional relevance.

## Overview

1. Parameter sdk (scepter.utils.config)
2. Path sdk (scepter.utils.directory)
3. PyTorch distributed sdk (scepter.utils.distribute)
4. Model export sdk (scepter.utils.export_model)
5. File system sdk (scepter.utils.file_system)
6. Logging sdk (scepter.utils.logger)
7. Video processing sdk (scepter.utils.video_reader), see the document (video_reader.md)
8. Module registration sdk (scepter.utils.registry)
9. Data sdk (scepter.utils.data)
10. Model sdk (scepter.utils.model)
11. Sampler sdk (scepter.utils.sampler)
12. Probing sdk (scepter.utils.probe)

<hr/>

## 1. Parameter sdk (scepter.modules.utils.config)

### Basic Usage

```python
from scepter.utils.config import Config
# Initialize Config object from a dict
fs_cfg = Config(load=False, cfg_dict={"NAME": "LocalFs"})
print(fs_cfg.NAME)
# Initialize Config object from a json file
import json
json.dump({"NAME": "LocalFs"}, open("examples.json", "w"))
fs_cfg = Config(load=True, cfg_file="examples.json")
print(fs_cfg.NAME)
# Initialize Config object from a yaml file
import yaml
yaml.dump({"NAME": "LocalFs"}, open("examples.yaml", "w"))
fs_cfg = Config(load=True, cfg_file="examples.yaml")
print(fs_cfg.NAME)
# Initialize Config object from an argparse object, in this mode cfg parameters are required, otherwise an error will be thrown.
import argparse
parser = argparse.ArgumentParser(
    description="Argparser for Cate process:\n"
)
parser.add_argument(
    "--stage",
    dest="stage",
    help="Running stage!",
    default="train"
)
fs_cfg = Config(load=True, parser_ins=parser)
print(fs_cfg.args)
```

<hr/>

### <font color="#0FB0E4">function **__init__**</font>

(cfg_dict: dict = {}, load = True, cfg_file = None, logger = None, parser_ins: argparse.ArgumentParser = None)

**Parameters**

- **cfg_dict** — A dict containing parameters, default is {}.

- **load** — When True, it indicates parameters need to be loaded from a file or argparse.

- **cfg_file** — Supports loading parameters from json or yaml files.

- **logger** — Logging instance, if None, a default logging instance to stdio will be initialized.

- **parser_ins** — An argparse instance, default includes cfg parameter for passing in a parameter file.

-- parser_ins will by default include the following system parameters:

  - cfg(--cfg) used to specify the parameter file location

  - local_rank(--local_rank) the default parameter read by torchrun, default is 0, can be ignored

  - launcher(-l) the method for starting the code, default is spawn, alternative is torchrun

  - data_online(-d) set global data not to be persisted to disk, should be set on pai clusters

  - share_storage(-s) set whether global data download is on a shared file system, like nas. When set, it implies the file system is shared across nodes, and only needs downloading at rank=0; when not set, it means data is downloaded on different nodes, and should only be downloaded when device_id=0.

### <font color="#0FB0E4">function **dict_to_yaml**</font>

(module_name: str, name: str, json_config: dict, set_name: bool = False)

**Parameters**

- **module_name** — The module name, used at the start of the template to explain which module's template it is.

- **name** — The default name for the Name field.

- **json_config** — Parameter description, needs to satisfy {} (indicating dependency on a sub-module), [] (dependency on multiple sub-modules), {"value":"", "description":""} (leaf parameter value).

- **set_name** — Whether to set the Name field.

**Returns**

- **str** — Template text

## 2. Path sdk (scepter.modules.utils.directory)
Some commonly used path functions
### Basic Usage
```python
from scepter.utils.directory import osp_path
# Automatically join paths based on the path prefix
prefix = "xxxx"
data_file = "example_videos/1.mp4"
# Outputs as xxxx/example_videos/1.mp4
print(osp_path(prefix, data_file))
# Also outputs as xxxx/example_videos/1.mp4
data_file = "xxxx/example_videos/1.mp4"
print(osp_path(prefix, data_file))
from scepter.utils.directory import get_relative_folder
# Get the folder path at a specified level according to the path
# By default, the last level xxxx/example_videos/
print(get_relative_folder(data_file))
# The second last level xxxx/
print(get_relative_folder(data_file, keep_index=-2))
from scepter.utils.directory import get_md5
# Get the md5 code of the text/path 34a447fb46d0b786a3999c9dad01d470
print(get_md5(data_file))
```
<hr/>

### <font color="#0FB0E4">function **osp_path**</font>

( prefix: str, data_file: str ) -> str

Automatically join paths based on the path prefix

**Parameters**

- **prefix** —— Path prefix.
- **data_file** —— File path.

**Returns**

- **str** —— Joined path after concatenation

### <font color="#0FB0E4">function **get_relative_folder**</font>

( abs_path: str, keep_index: int = -1 ) -> str

Get the folder path at a specified level according to the path

**Parameters**

- **abs_path** —— File path.
- **keep_index** —— Level to keep, -1 for the last level, -2 for the second last level.

**Returns**

- **str** —— Parsed path after resolution

### <font color="#0FB0E4">function **get_md5**</font>

( ori_str: str) -> str

Get the md5 code based on a string/path

**Parameters**

- **ori_str** —— File path or string.

**Returns**

- **str** —— md5 code

## 3. PyTorch Distributed(scepter.modules.utils.distribute)
PyTorch distributed initialization SDK. By using this SDK, users can avoid focusing on the implementation details of PyTorch's distributed initialization.
### Basic Usage

```python
from scepter.utils.distribute import we
from scepter.utils.config import Config

cfg = Config(cfg_dict={}, load=False)


def fn():
    pass


print(we)
# Launch task
we.init_env(cfg, fn, logger=None)
```
<hr/>

### <font color="#0FB0E4">class **Workenv**</font>

This is a class used to uniformly manage the running environment. It is usually not necessary to initialize this class. In scepter.modules.utils.distribute,
a global instance, 'we', will be initialized to manage some key flag variables.
- Specific explanations of some parameters of we are as follows:
  - initialized marks whether the PyTorch process group has been initialized, default is False.
  - is_distributed marks whether it is currently running in distributed mode, default is False.
  - sync_bn marks whether to use sync_bn, default is False.
  - rank marks the current process's rank, default is 0.
  - world_size marks the total number of processes, default is 1.
  - device_id marks the current device ID being used, default is 0.
  - device_count marks the total number of devices in the current environment, default is 1.
  - use_pl marks whether the pytorch_lighting engine is used in the current environment, default is False.
  - launcher marks the method of starting the environment, default is spawn.
  - data_online marks whether the io part of the data in the current environment is persisted to disk, default is False.
  - share_storage marks whether different nodes in the current environment use the same file system, such as nas, default is False.
### <font color="#0FB0E4">function **we.init_env**</font>

( config: scepter.modules.utils.config.Config, fn: function, logger: logging.Logger = None )

As the entry point for executing any task.

**Parameters**

- **config** —— The passed instance of parameters.
- **fn** —— The function that needs to be executed.
- **logger** —— A standard logging instance.

### <font color="#0FB0E4">function **we.get_env**</font>
() -> dict

Retrieve all class-internal parameters of we, stored in the form of a dict.


### <font color="#0FB0E4">function **we.set_env**</font>
(we_env: dict)

Reset all class-internal parameters of we, using a dict as input.

**Parameters**

- **we_env** —— A dict, each key represents an internal class variable.

### <font color="#0FB0E4">function **get_dist_info**</font>
() -> int, int

Obtain the environment's rank/world size, which is directly acquired through torch's methods, typically used when the environment is not initialized with we.init_env.

**Returns**

- **rank** —— The rank of the current process, default is 0.
- **world_size** —— The total number of processes in the current environment, when in single-process mode it is 1.

### <font color="#0FB0E4">function **gather_data**</font>
(data: [list, dict, tensor, object] ) -> data

Using scepter.distributed.all_gather to collect any instance, and merge it into a summarized instance on rank=0 process.

**Parameters**
  - **data** —— Supports dict/list, where elements can be any instance or tensor.

**Returns**
  - **data** —— A summarized data with the same structure as the input data.

### <font color="#0FB0E4">function **gather_list**</font>
(data: [list] ) -> data

Using scepter.distributed.all_gather to collect any list instance, and merge it into a summarized instance on rank=0 process.

**Parameters**
  - **data** —— Supports list, where elements can be any instance or tensor.

**Returns**
  - **data** —— A summarized data with the same structure as the input data.

### <font color="#0FB0E4">function **gather_picklable**</font>
(data: [object] ) -> data

Using scepter.distributed.all_gather to collect any picklable instance, and merge it into a summarized instance on rank=0 process.

**Parameters**
  - **data** —— A serializable instance.

**Returns**
  - **data** —— A summarized data with the same structure as the input data.

### <font color="#0FB0E4">function **broadcast**</font>
(tensor: **torch.Tensor**, src: **str**, group: **list** )

An optimized version of torch.distributed.broadcast, automatically checks if it is a distributed environment.

**Parameters**
  - **tensor** —— The tensor to be broadcast.
  - **src** —— The source device for broadcasting.
  - **group** —— The group for broadcasting.

**Returns**
  - **data** —— A summarized data with the same structure as the input data.
* Other functions such as barrier, all_reduce, reduce, send, recv, isend, irecv, scatter have also been adapted for this operation.


### <font color="#0FB0E4">function **gather_gpu_tensors**</font>
(tensor: torch.Tensor ) -> tensor: torch.Tensor

Using torch.distributed.all_gather to collect GPU tensors, and merge then transfer them to the CPU on rank=0.
Since cloning is involved, this may cause additional GPU memory waste.

**Parameters**
  - **tensor** —— The GPU tensor input.

**Returns**
  - **tensor** —— The output tensor on the CPU for process rank=0.

## 4. 模型导出sdk(scepter.utils.export_model)
 APIs for exporting models to TorchScript/ONNX formats.
### Basic Usage

```python
from scepter.utils.export_model import save_develop_model_multi_io

save_develop_model_multi_io(
    model,
    input_size,
    input_type,
    input_name,
    output_name,
    limit,
    save_onnx_path=None,
    save_pt_path=None
)
```
<hr/>

### <font color="#0FB0E4">function **save_develop_model_multi_io**</font>
(model: torch.nn.Module, input_size: list, input_type: list, input_name: list,
output_name: list, limit: list, save_onnx_path: str = None, save_pt_path: str = None) -> pt_module, onnx_module

Supports importing and exporting models with multiple inputs and outputs

**Parameters**
  - **model** —— The model instance to be exported.
  - **input_size** —— A list where each tuple contains the shape information of the data, such as [[1, 3, 224, 224]].
  - **input_type** —— A list where each tuple contains the type information of the data, corresponding to input_size, with possible values ("float32", "float16", "int8", "int16", "int32", "int64"). For example, ["float32"].
  - **input_name** —— A list used to name each input variable for ONNX, such as ["image"], corresponding to the above input_size and input_type.
  - **output_name** —— A list used to name each output variable for ONNX, such as ["output"]
  - **limit** ——  A list where each tuple defines the upper and lower bounds for that input, such as [[-1, 1]], representing that the input tensor for the image is between -1 and 1.
  - **save_onnx_path** —— If not None, the ONNX model will be exported and stored at this location.
  - **save_pt_path** —— If not None, the TorchScript model will be exported and stored at this location.



**Returns**
  - **tensor** —— The output tensor on the CPU for process rank=0.

## 5. 文件系统sdk(scepter.utils.file_system)
Refer to [file_clients](file_clients.md)

## 6. Logging SDK(scepter.utils.logger)
Used to instantiate a standard logging instance for printing information.

### Basic Usage

```python
from scepter.utils.logger import get_logger, init_logger

std_logger = get_logger(name="scepter")
init_logger(std_logger, log_file="", dist_launcher="pytorch")
```
<hr/>

### <font color="#0FB0E4">function **get_logger**</font>
(name: str) -> logger

Retrieve a logging instance.

**Parameters**
  - **name** —— The log prefix; it will be printed first every time the logger prints.

**Returns**
  - **logger** —— Returns a logging instance.

### <font color="#0FB0E4">function **init_logger**</font>
(in_logger: logger, log_file: str) -> logger

Re-initialize a logging instance, which can assign a file for output storage.

**Parameters**
  - **in_logger** —— The existing logging instance.
  - **log_file** —— The desired file location for storage.
  - **dist_launcher** —— No longer important, deprecated

### <font color="#0FB0E4">function **as_time**</font>
(s: int) -> str

Convert time in seconds s to the standard format of xxx days xxx hours xxx mins xxx secs

**Parameters**
  - **s** ——  Represents the number of seconds s.

**Returns**
  - **str** —— Formatted output.

### <font color="#0FB0E4">function **time_since**</font>
(since: int, percent: float) -> str

Calculate the time remaining until completion based on the current usage time and percentage.

**Parameters**
  - **since** —— Represents the current elapsed time.
  - **percent** —— Represents the percentage of completion.

**Returns**
  - **str** —— Formatted output.

## 7. Video Processing SDK (scepter.utils.video_reader)
APIs for handling video reading.

### Basic Usage

```python
from scepter.utils.video_reader.frame_sampler import do_frame_sample
from scepter.utils.video_reader.video_reader import (
    VideoReaderWrapper, EasyVideoReader, FramesReaderWrapper
)
```
<hr/>

### <font color="#0FB0E4">function **do_frame_sample**</font>
(sampling_type: str, vid_len: int, vid_fps: int, num_frames: int, kwargs) -> list

Get a frame sampler for videos.

**Parameters**
  - **sampling_type** —— The type of sampler, currently supports UniformSampler (uniform sampler), IntervalSampler (interval sampler), SegmentSampler (segment sampler).
  - **vid_len** —— Video length.
  - **vid_fps** —— Frame rate of the video.
  - **num_frames** —— Number of frames in the video.
  - **kwargs** ——  Required parameters for the corresponding sampler, refer to the source code of the corresponding sampler.

**Returns**
  - **list** —— Sampled frames result.

### <font color="#0FB0E4">class **VideoReaderWrapper**</font>
A standard class for reading videos, with the underlying decoder being decord.

#### <font color="#0FB0E4">function **VideoReaderWrapper.__init__**</font>
(video_path: str)

Initialize a video instance.

**Parameters**
  - **video_path** —— Video link.

#### <font color="#0FB0E4">function **VideoReaderWrapper.len**</font>
() -> int

Get the total number of video frames.

**Returns**
  - **int** —— Number of video frames.

#### <font color="#0FB0E4">function **VideoReaderWrapper.fps**</font>
() -> float

Get video frame rate.

**Returns**
  - **float** —— Video frame rate.

#### <font color="#0FB0E4">function **VideoReaderWrapper.duration**</font>
() -> float

Get video duration.

**Returns**
  - **float** —— Video duration.

#### <font color="#0FB0E4">function **VideoReaderWrapper.sample_frames**</font>
(decode_list: torch.Tensor) -> torch.Tensor

Tensor Get frame data based on frame numbers.

**Parameters**
  - **decode_list** —— List of sampled frame numbers.
**Returns**
  - **tensor** —— Data tensor.

### <font color="#0FB0E4">class **FramesReaderWrapper**</font>
Reads frame data in order from a given fully decoded frame folder.

#### <font color="#0FB0E4">function **FramesReaderWrapper.__init__**</font>
(frame_dir: str, extract_fps: float, suffix: str)

Initialize a video instance.

**Parameters**
  - **frame_dir** —— The frame folder.
  - **extract_fps** —— FPS for extracting frames.
  - **suffix** —— Suffix for the frame files, default is jpg.

#### <font color="#0FB0E4">function **FramesReaderWrapper.len**</font>
() -> int

Get the total number of video frames.

**Returns**
  - **int** —— Number of video frames.

#### <font color="#0FB0E4">function **FramesReaderWrapper.fps**</font>
() -> float

Get video frame rate.

**Returns**
  - **float** —— Video frame rate.

#### <font color="#0FB0E4">function **FramesReaderWrapper.duration**</font>
() -> float

Get video duration.

**Returns**
  - **float** —— Video duration.

#### <font color="#0FB0E4">function **FramesReaderWrapper.sample_frames**</font>
(decode_list: torch.Tensor) -> torch.Tensor

Get frame data based on frame numbers.

**Parameters**
  - **decode_list** —— List of sampled frame numbers.
**Returns**
  - **tensor** —— Data tensor.

### <font color="#0FB0E4">class **EasyVideoReader**</font>
Used for reading, sampling, and preprocessing long videos.

#### <font color="#0FB0E4">function **EasyVideoReader.__init__**</font>
(video_path: str, num_frames: int, clip_duration: Union[float, Fraction, str],
overlap: Union[float, Fraction, str] = Fraction(0), transforms: Optional[Callable] = None)

Initialize a video instance.

**Parameters**
  - **video_path** —— Video link.
  - **num_frames** —— Number of video frames.
  - **clip_duration** —— Length of each clip.
  - **overlap** —— Proportion of overlap between clips.
  - **transforms** ——  Preprocessing operators.

#### <font color="#0FB0E4">function **EasyVideoReader.__iter__**</font>
() -> int

Iterator

#### <font color="#0FB0E4">function **EasyVideoReader.__next__**</font>
() -> float

Iterator, with each iteration returning a tensor of a segment.

**Returns**
  - **tensor** —— The tensor of the video segment.

## 8. Module Registration SDK (scepter.utils.registry)
Used for managing various registered classes.

### Basic Usage

```python
from scepter.utils.registry import Registry
from scepter.utils.config import Config

MODELS = Registry('MODELS')


@MODELS.register_class()
class ResNet(object):
    pass


config = Config(load=False, cfg_dict={"NAME": "ResNet"})
resnet = MODELS.build(config)
```
<hr/>

### <font color="#0FB0E4">class **Registry**</font>
Registry

#### <font color="#0FB0E4">function **Registry.__init__**</font>
(name: str, build_func: function = None, common_para: Config = None, allow_types: tuple = ("class", "function"))

Initialize the registry module instance

**Parameters**
  - **name** —— Module name.
  - **build_func** —— The function called when building the module.
  - **common_para** —— Common parameters under this module.
  - **allow_types** —— The types of classes or functions allowed to be registered in this module, by default, registration of both is allowed.

#### <font color="#0FB0E4">function **Registry.build**</font>
(cfg: Config, logger: logger = None, kwargs) -> cls_obj

Build an instance of the target class

**Returns**
  - **cls_obj** —— An instance of a specific class.

#### <font color="#0FB0E4">function **Registry.register_class**</font>
(name: str)

Register a class

**Returns**
  - **name** —— Registration name.

#### <font color="#0FB0E4">function **Registry.register_function**</font>
(name: str)

Register a function

**Returns**
  - **name** —— Registration name.

## 9. Data SDK(scepter.utils.data)
Used for transferring data between devices

### Basic Usage

```python
import torch
from scepter.utils.data import transfer_data_to_numpy, transfer_data_to_cpu, transfer_data_to_cuda

data = {"a": torch.Tensor([0])}
transfer_data_to_numpy(data)
transfer_data_to_cpu(data)
transfer_data_to_cuda(data)
```
<hr/>


#### <font color="#0FB0E4">function **transfer_data_to_numpy**</font>
(data: list/dict of torch.Tensor) -> (data: list/dict of numpy.ndarray)

Transfer data to numpy

**Parameters**
  - **data** —— Stored as a list/dict of torch.Tensor.
**Returns**
  - **data** —— Stored as a list/dict of numpy.ndarray, consistent with the input format.

#### <font color="#0FB0E4">function **transfer_data_to_cpu**</font>
(data: list/dict of torch.Tensor(cuda)) -> (data: list/dict of torch.Tensor(cpu))

Transfer data from GPU to CPU

**Parameters**
  - **data** —— Stored as a list/dict of torch.Tensor[CUDA].
**Returns**
  - **data** —— Stored as a list/dict of torch.Tensor[CPU], consistent with the input format.

#### <font color="#0FB0E4">function **transfer_data_to_cuda**</font>
(data: list/dict of torch.Tensor(cpu)) -> (data: list/dict of torch.Tensor(cuda))

Transfer data from CPU to GPU

**Parameters**
  - **data** —— Stored as a list/dict of torch.Tensor[CPU].
**Returns**
  - **data** —— Stored as a list/dict of torch.Tensor[CUDA], consistent with the input format.

## 10. Model SDK(torch.utils.model)
Used for operations such as loading and evaluating models

### Basic Usage

```python
import torch
from scepter.utils.model import move_model_to_cpu, load_pretrained,
    count_params, init_weights
```
<hr/>


#### <font color="#0FB0E4">function **move_model_to_cpu**</font>
(params: list/dict of torch.Tensor[cuda]) -> (data: torch.Tensor[cpu])

Move parameter data from GPU to CPU.

**Parameters**
  - **params** —— Stored as OrderedDict of torch.Tensor[cuda].
**Returns**
  - **params** —— Stored as torch.Tensor[cpu], consistent with the input format.

#### <font color="#0FB0E4">function **load_pretrained**</font>
(model: torch.nn.Module, path: str, map_location="cpu", logger=None,
                    sub_level=None)

Load parameters into the model.

**Parameters**
  - **model** —— The torch.nn.Module model instance.
  - **path** —— Pretrained model parameters.
  - **map_location** —— cpu/cuda。
  - **logger** —— Standard logging instance.
  - **sub_level** —— For example, when using DDP, sub-level indexing might be needed.


#### <font color="#0FB0E4">function **count_params**</font>
(model: torch.nn.Module) -> (float)

Count the total parameters of the model.

**Parameters**
  - **model** —— The torch.nn.Module model instance.
**Returns**
  - **float** —— The quantity of model parameters (number of floating-point values).

#### <font color="#0FB0E4">function **init_weights**</font>
(model: torch.nn.Module)

Initialize the parameters of the model modules.

**Parameters**
  - **module** —— The torch.nn.Module model instance.

## 11. Sampler SDK(scepter.utils.sampler)
Samplers are quite universal, and in most cases, custom development is not required. Here are provided several common types of sampler.

### Basic Usage

```python
import torch
from scepter.utils.sampler import MultiFoldDistributedSampler,
    EvalDistributedSampler, MultiLevelBatchSampler, MixtureOfSamplers
```
<hr/>


#### <font color="#0FB0E4">class **MultiFoldDistributedSampler**</font>

Multi-fold sampler, supports repeating multiple rounds of data within one epoch.

#### <font color="#0FB0E4">function **MultiFoldDistributedSampler.__init__**</font>

( dataset: torch.data.dataset, num_folds=1, num_replicas=None, rank=None, shuffle=True)

**Parameters**

- **dataset** —— An instance of torch.data.dataset class.
- **num_folds** —— Int, indicates the number of times the data is repeated.
- **num_replicas** —— Indicates the number of data partitions, usually consistent with world-size.
- **rank** —— Indicates the current process number.
- **shuffle** —— Whether to shuffle the data.

#### <font color="#0FB0E4">function **MultiFoldDistributedSampler.__iter__**</font>
()

Iterator, each iteration returns an index of a sample.

#### <font color="#0FB0E4">function **MultiFoldDistributedSampler.set_epoch**</font>
(epoch: int)

Set the current epoch.

**Parameters**

- **epoch** —— The current epoch.


#### <font color="#0FB0E4">class **EvalDistributedSampler**</font>

A sampler for testing, when not using padding mode, it will be observed that the last rank has fewer data than other ranks.

#### <font color="#0FB0E4">function **EvalDistributedSampler.__init__**</font>

( dataset: torch.data.dataset, num_replicas: Optional[int] =None, rank: Optional[int] =None, padding： bool =False)

**Parameters**

- **dataset** —— An instance of torch.data.dataset class.
- **num_replicas** —— Indicates the number of data partitions, usually consistent with world-size.
- **rank** —— Rank indicates the current process number.
- **padding** —— Whether the data needs to be padded, if padded it can ensure the last rank has the same amount of data as the other ranks.

#### <font color="#0FB0E4">function **EvalDistributedSampler.__iter__**</font>
()

Iterator, each iteration returns an index of a sample.

#### <font color="#0FB0E4">function **EvalDistributedSampler.set_epoch**</font>
(epoch: int)

Set the current epoch.

**Parameters**

- **epoch** —— The current epoch.

#### <font color="#0FB0E4">class **MultiLevelBatchSampler**</font>

A sampler for multi-level indexing of large-scale data.

#### <font color="#0FB0E4">function **MultiLevelBatchSampler.__init__**</font>

(index_file: str, batch_size: int, rank: int =0, seed: int = 8888)

**Parameters**

- **index_file** —— Index file for multi-level data indexing.
- **batch_size** —— The size of a batch.
- **rank** —— Rank indicates the current process number.
- **seed** —— Random sampling seed, obtained globally from data.registry.

#### <font color="#0FB0E4">function **MultiLevelBatchSampler.__iter__**</font>
()

Iterator, each iteration returns an index of a sample.


#### <font color="#0FB0E4">class **MixtureOfSamplers**</font>

A sampler for multi-level indexing of large-scale data.

#### <font color="#0FB0E4">function **MixtureOfSamplers.__init__**</font>

(samplers: list(sampler), probabilities: list(float), rank: int =0, seed: int = 8888)

**Parameters**

- **samplers** —— A list of samplers used for mixing.
- **probabilities** —— The probability of each sampler.
- **rank** —— Rank indicates the current process number.
- **seed** —— Random sampling seed, obtained globally from data.registry.

#### <font color="#0FB0E4">function **MixtureOfSamplers.__iter__**</font>
()

Iterator, each iteration returns an index of a sample.

## 12. Prober SDK(scepter.utils.probe)
Used for probing variable statistics of various components.

### Basic Usage

```python
import numpy as np
from scepter.model.base_model import BaseModel
from scepter.utils.config import Config
from scepter.utils.file_system import FS
from scepter.utils.probe import ProbeData


class TestModel(BaseModel):
    def forward(self, data):
        self.register_probe(data)
        # Test ProbeData example + view_distribute
        self.register_probe(
            {"data_key_dist": ProbeData(data["data_key"], view_distribute=True),
             "data_folder": ProbeData(data["data_folder"], view_distribute=True)}
        )


class TestModel2(BaseModel):
    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger=logger)
        self.test_model = TestModel(cfg, logger=logger)

    def forward(self, data):
        # Test list of str done
        # Test dict of str done
        # Test dict of number done
        # Test list of number done
        # Test number done
        # Test str done
        # Test np.array done
        # Test list of np.ndarray must manually create ProbeData done
        # Test 2D image must manually create ProbeData
        # Test 3D image must manually create ProbeData
        # Test 3D multiple 2D images must manually create ProbeData
        # Test 3D list image must manually create ProbeData
        # Test 4D Array image done
        # Test 4D Array image save_html
        # Test 4D List image save_html
        self.register_probe(data)
        self.register_probe({
            "test_np_list": ProbeData([np.zeros([40, 40, 3]).astype(dtype=np.int8) for _ in range(5)]),
            "test_2d_img": ProbeData(np.zeros([40, 40]).astype(dtype=np.uint8), is_image=True),
            "test_3d_n2d_img": ProbeData(np.zeros([10, 40, 40]).astype(dtype=np.uint8), is_image=True),
            "test_3d_img": ProbeData(np.zeros([40, 40, 3]).astype(dtype=np.uint8), is_image=True),
            "test_list_3d_img": ProbeData([np.zeros([40, 40, 3]).astype(dtype=np.uint8) for _ in range(5)],
                                          is_image=True),
            "test_4d_img": ProbeData(np.zeros([10, 40, 40, 3]).astype(dtype=np.uint8), is_image=True),
            "test_4d_img_html": ProbeData(np.zeros([10, 40, 40, 3]).astype(dtype=np.uint8), is_image=True,
                                          build_html=True, build_label="4d_data"),
            "test_4d_img_list_html": ProbeData([np.zeros([10, 40, 40, 3]).astype(dtype=np.uint8) for _ in range(5)],
                                               is_image=True,
                                               build_html=True, build_label=[f"4d_data_{i}" for i in range(5)]),
        })
        # Test nested types
        self.test_model(data)


cfg = Config(cfg_file="./config/general_config.yaml")
if cfg.have("FILE_SYSTEMS"):
    for file_sys in cfg.FILE_SYSTEMS:
        fs_prefix = FS.init_fs_client(file_sys)
else:
    fs_prefix = FS.init_fs_client(cfg)

_model = TestModel2(cfg)

data = {
    "data_key": [1, 1],
    "data_folder": {"mj": 1., "mj_square": 2.},
    "timestamp": 1,
    "valid_str": "right",
    "test_np": np.zeros([40, 40, 3]).astype(dtype=np.int8)
}
_model(data)
probe = _model.probe_data()
for key in probe:
    print(key, probe[key].to_log(prefix=f"xxx/{key}"))
```
<hr/>

Use in conjunction with Hooks as follows (where PROB_INTERVAL is the probe storage interval, i.e., the number of calls to probe_data()):

```yaml
-
  NAME: ProbeDataHook
  PROB_INTERVAL: 100
```
#### <font color="#0FB0E4">class **ProbeData**</font>

Instance of probe data.

#### <font color="#0FB0E4">function **ProbeData.__init__**</font>

(data, is_image = False, build_html = False, build_label = None, view_distribute = False)

**Parameters**
- **data** —— The probe data passed in, currently supports str, Number, list, dict, tensor.
- **is_image** —— Whether to store as an image.
- **build_html** —— Whether to store as html.
- **build_label** —— The label for saving html.
- **view_distribute** —— To count the frequency of some values.
