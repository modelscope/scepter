# 依赖组件（Utils）

依赖SDK，该部分用于对框架全局经常复用的模块和sdk进行整理，并根据功能相关性进行聚合。

## 总览
1. 参数sdk(scepter.utils.config)
2. 路径sdk(scepter.utils.directory)
3. torch分布式sdk(scepter.utils.distribute)
4. 模型导出sdk(scepter.utils.export_model)
5. 文件系统sdk(scepter.utils.file_system)
6. 日志sdk(scepter.utils.logger)
7. 视频处理sdk(scepter.utils.video_reader)，文档参考(video_reader.md)
8. 模块注册sdk(scepter.utils.registry)
9. 数据sdk(scepter.utils.data)
10. 模型sdk(scepter.utils.model)
11. 采样器sdk(scepter.utils.sampler)
12. 探针器sdk(scepter.utils.probe)

<hr/>

## 1. 参数sdk(scepter.modules.utils.config)

### 基础用法

```python
from scepter.utils.config import Config

# 从一个dict对象 初始化 Config对象
fs_cfg = Config(load=False, cfg_dict={"NAME": "LocalFs"})
print(fs_cfg.NAME)
# 从一个json文件中初始化 Config对象
import json

json.dump({"NAME": "LocalFs"}, open("examples.json", "w"))
fs_cfg = Config(load=True, cfg_file="examples.json")
print(fs_cfg.NAME)
# 从一个yaml文件中初始化 Config对象
import yaml

yaml.dump({"NAME": "LocalFs"}, open("examples.yaml", "w"))
fs_cfg = Config(load=True, cfg_file="examples.yaml")
print(fs_cfg.NAME)
# 从 argparse 对象中初始化 Config对象，该模式下cfg的参数为必需参数，否则会报错。
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

( cfg_dict: dict = {}, load = True, cfg_file = None, logger = None, parser_ins: argparse.ArgumentParser = None )

**Parameters**

- **cfg_dict** —— 包含参数的dict，默认为{}。
- **load** —— 为True时说明需要从文件或argparse中载入参数。
- **cfg_file** —— 支持从json文件或者yaml文件中载入参数。
- **logger** —— 日志示例，如果为None，则会默认初始化一个stdio的日志实例。
- **parser_ins** —— argparse实例，默认有cfg参数，用于传入参数文件。
-- parser_ins 默认会加入系统参数，说明如下：
  - cfg(--cfg) 用于指定参数文件位置
  - local_rank(--local_rank) torchrun默认读取参数，默认为0，可不管
  - launcher(-l) 启动代码的方式，默认为spawn，可选值为 torchrun
  - data_online(-d) 设置全局下载数据不落盘，在pai集群上应设置该值
  - share_storage(-s) 设置全局下载数据是否共享文件系统，如nas。当设置时，说明文件系统不同节点互通，此时只需在rank=0时下载即可；当不设置时，说明
是在不同的节点进行数据下载，此时应该只在device_id=0时下载。

### <font color="#0FB0E4">function **dict_to_yaml**</font>
( module_name: str, name: str, json_config: dict, set_name: bool = False )

**Parameters**

- **module_name** —— 模块名称，用于在模版开始说明是哪个模块的模版。
- **name** —— Name字段的默认名称。
- **json_config** —— 参数说明，需要满足{}(表示依赖一个子模块), []（依赖多个子模块）, {"value":"", "description":""} （叶子参数值）。
- **set_name** —— 是否设置Name字段。

**Returns**

- **str** —— 模版文本



## 2. 路径sdk(scepter.modules.utils.directory)
一些常用的路径函数
### 基础用法

```python
from scepter.utils.directory import osp_path

# 根据路径前缀进行自动化路径拼接
prefix = "xxxx"
data_file = "example_videos/1.mp4"
# 输出为 xxxx/example_videos/1.mp4
print(osp_path(prefix, data_file))
# 输出也为 xxxx/example_videos/1.mp4
data_file = "xxxx/example_videos/1.mp4"
print(osp_path(prefix, data_file))

from scepter.utils.directory import get_relative_folder

# 根据路径获取指定层级的文件夹路径
# 默认最后一级 xxxx/example_videos/
print(get_relative_folder(data_file))
# 倒数第二级 xxxx/
print(get_relative_folder(data_file, keep_index=-2))

from scepter.utils.directory import get_md5

# 获取文本/路径的md5码 34a447fb46d0b786a3999c9dad01d470
print(get_md5(data_file))
```
<hr/>

### <font color="#0FB0E4">function **osp_path**</font>

( prefix: str, data_file: str ) -> str

根据路径前缀进行自动化路径拼接

**Parameters**

- **prefix** —— 路径前缀。
- **data_file** —— 文件路径。

**Returns**

- **str** —— 拼接以后的路径

### <font color="#0FB0E4">function **get_relative_folder**</font>

( abs_path: str, keep_index: int = -1 ) -> str

根据路径获取指定层级的文件夹路径

**Parameters**

- **abs_path** —— 文件路径。
- **keep_index** —— 保留层级，-1代表倒数第一级，-2 为倒数第二级。

**Returns**

- **str** —— 解析以后的路径

### <font color="#0FB0E4">function **get_md5**</font>

( ori_str: str) -> str

根据字符串/路径获取md5码

**Parameters**

- **ori_str** —— 文件路径或字符串。

**Returns**

- **str** —— md5码

## 3. torch分布式sdk(scepter.modules.utils.distribute)
torch分布式初始化sdk，使用该sdk，可以让用户不要关注torch的分布式初始化的实现。
### 基础用法

```python
from scepter.utils.distribute import we
from scepter.utils.config import Config

cfg = Config(cfg_dict={}, load=False)


def fn():
    pass


print(we)
# 启动任务
we.init_env(cfg, fn, logger=None)
```
<hr/>

### <font color="#0FB0E4">class **Workenv**</font>

这是一个用于统一管理运行环境的类，通常不需要使用该类做初始化，在scepter.modules.utils.distribute
中会初始化一个全局的实例we，用于管理一些关键性的标志变量。

- 关于we的一些参数，具体说明如下：
  - initialized 标记是否初始化torch的process group，默认为False。
  - is_distributed 标记当前是否为分布式运行，默认为False。
  - sync_bn 标记是否使用sync_bn，默认为False。
  - rank 标记当前process的rank，默认为0。
  - world_size 标记当前所有的进程数，默认为1。
  - device_id 标记当前使用的设备ID，默认为0。
  - device_count 标记当前环境下的所有设备数，默认为1。
  - use_pl 标记当前环境是否使用pytorch_lighting引擎，默认为False
  - launcher 标记当前环境的启动方式，默认为spawn。
  - data_online 标记当前环境下io部分的数据是否落盘，默认为False。
  - share_storage 标记当前环境下不同节点是否使用相同的文件系统，如nas，默认为False。

### <font color="#0FB0E4">function **we.init_env**</font>

( config: scepter.modules.utils.config.Config, fn: function, logger: logging.Logger = None )

作为启动任何任务的执行入口。

**Parameters**

- **config** —— 传入的参数实例。
- **fn** —— 需要执行的函数。
- **logger** —— 标准的日志实例。

### <font color="#0FB0E4">function **we.get_env**</font>
() -> dict

获取we的所有类内参数，以dict的形式存储。


### <font color="#0FB0E4">function **we.set_env**</font>
(we_env: dict)

重新设置we的所有类内参数，以dict的形式作为输入。

**Parameters**

- **we_env** —— dict，每个key代表一个类内变量。

### <font color="#0FB0E4">function **get_dist_info**</font>
() -> int, int

获取环境的rank/world size，这个是直接通过torch的方法来获取的，一般用于当初始化环境的方式
不是we.init_env的时候使用。

**Returns**

- **rank** —— 当前进程的rank值，默认为0
- **world_size** —— 当前环境的总进程数， 当单进程时为1。

### <font color="#0FB0E4">function **gather_data**</font>
(data: [list, dict, tensor, object] ) -> data

通过scepter.distributed.all_gather将任意实例收集起来，并在rank=0进程合并为一个汇总后的实例。

**Parameters**
  - **data** —— 支持dict/list，其中元素支持任意实例或者tensor。

**Returns**
  - **data** —— 一个和输入data相同结构的汇总过的数据。

### <font color="#0FB0E4">function **gather_list**</font>
(data: [list] ) -> data

通过scepter.distributed.all_gather将任意实例收集起来，并在rank=0进程合并为一个汇总后的实例。

**Parameters**
  - **data** —— 支持list，其中元素支持任意实例或者tensor。

**Returns**
  - **data** —— 一个和输入data相同结构的汇总过的数据。

### <font color="#0FB0E4">function **gather_picklable**</font>
(data: [object] ) -> data

通过scepter.distributed.all_gather将任意实例收集起来，并在rank=0进程合并为一个汇总后的实例。

**Parameters**
  - **data** —— 为一个可序列化的实例。

**Returns**
  - **data** —— 一个和输入data相同结构的汇总过的数据。

### <font color="#0FB0E4">function **broadcast**</font>
(tensor: **torch.Tensor**, src: **str**, group: **list** )

为torch.distributed.broadcast的优化版本，自动确认是否为分布式环境。

**Parameters**
  - **tensor** —— 需要广播的tensor。
  - **src** —— 需要广播的源设备。
  - **group** —— 需要广播的组。

**Returns**
  - **data** —— 一个和输入data相同结构的汇总过的数据。
* 其他函数如barrier、all_reduce、 reduce、send、recv、isend、irecv、scatter也做了此操作。


### <font color="#0FB0E4">function **gather_gpu_tensors**</font>
(tensor: torch.Tensor ) -> tensor: torch.Tensor

通过torch.distributed.all_gather将gpu tensor收集起来，并在rank=0合并并转到cpu上。
因为涉及到clone，因此有可能造成额外的显存浪费。

**Parameters**
  - **tensor** —— 输入的gpu上的tensor。

**Returns**
  - **tensor** —— 输出的在进程rank=0上的cpu的tensor。

## 4. 模型导出sdk(scepter.utils.export_model)
 用于模型导出为torchscript/Onnx格式的api。
### 基础用法

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

支持多输入多输出的模型导入和导出

**Parameters**
  - **model** —— 待导出的模型实例
  - **input_size** —— 为一个list，每个元组包含数据的shape信息，如[[1, 3, 224, 224]]。
  - **input_type** —— 为一个list，每个元组包含数据的type信息，与input_size一一对应，可选值为（"float32"，
"float16"，"int8"，"int16"，"int32"，"int64"）。如["float32"]。
  - **input_name** —— 为一个list，为onnx的每个输入变量命名，如["image"]，与上述input_size、
input_type 一一对应。
  - **output_name** —— 为一个list，为onnx的每个输出变量命名，如["output"]
  - **limit** —— 为一个list，每个元祖定义了该输入的上下届，如[[-1, 1]]，代表了image的输入张量在-1~1之间。
  - **save_onnx_path** —— 不为None时，会导出onnx模型，存储在该位置。
  - **save_pt_path** —— 不为None时，会导出torchscript模型，存储在该位置。



**Returns**
  - **tensor** —— 输出的在进程rank=0上的cpu的tensor。

## 5. 文件系统sdk(scepter.utils.file_system)
参考[file_clients](file_clients.md)

## 6. 日志sdk(scepter.utils.logger)
用于实例化一个标准的日志实例，用于打印信息。

### 基础用法

```python
from scepter.utils.logger import get_logger, init_logger

std_logger = get_logger(name="std_torch")
init_logger(std_logger, log_file="", dist_launcher="pytorch")
```
<hr/>

### <font color="#0FB0E4">function **get_logger**</font>
(name: str) -> logger

获取日志实例。

**Parameters**
  - **name** —— 日志前缀，每次打印会首先打印该前缀。

**Returns**
  - **logger** —— 返回一个logging实例。

### <font color="#0FB0E4">function **init_logger**</font>
(in_logger: logger, log_file: str) -> logger

二次初始化日志实例，可以为该实例分配一个文件落盘。

**Parameters**
  - **in_logger** —— 已有的日志实例。
  - **log_file** —— 希望存储的文件位置。
  - **dist_launcher** —— 已经不重要了，deprecated

### <font color="#0FB0E4">function **as_time**</font>
(s: int) -> str

时间s转换为标准的xxx days xxx hours xxx mins xxx secs

**Parameters**
  - **s** —— 代表秒数s。

**Returns**
  - **str** —— 格式化的输出。

### <font color="#0FB0E4">function **time_since**</font>
(since: int, percent: float) -> str

根据当前用时和百分比计算距离结束的时间。

**Parameters**
  - **since** —— 代表当前已经用的时间。
  - **percent** —— 代表当前已经执行的百分比。

**Returns**
  - **str** —— 格式化的输出。

## 7. 视频处理sdk(scepter.utils.video_reader)
用于处理视频读取的api。

### 基础用法

```python
from scepter.utils.video_reader.frame_sampler import do_frame_sample
from scepter.utils.video_reader.video_reader import (
    VideoReaderWrapper, EasyVideoReader, FramesReaderWrapper
)
```
<hr/>

### <font color="#0FB0E4">function **do_frame_sample**</font>
(sampling_type: str, vid_len: int, vid_fps: int, num_frames: int, kwargs) -> list

获取针对于视频的帧采样器。

**Parameters**
  - **sampling_type** —— 采样器类型，目前支持的UniformSampler（均匀采样器）、IntervalSampler（等间隔采样器）、SegmentSampler（切片采样器）。
  - **vid_len** —— 视频长度。
  - **vid_fps** —— 视频的帧率。
  - **num_frames** —— 视频包含的帧数。
  - **kwargs** —— 对应采样器的需要参数，需要参考对应采样器的源码。

**Returns**
  - **list** —— 采样帧结果。

### <font color="#0FB0E4">class **VideoReaderWrapper**</font>
读取视频的标准类，底层解码器为decord

#### <font color="#0FB0E4">function **VideoReaderWrapper.__init__**</font>
(video_path: str)

初始化视频实例

**Parameters**
  - **video_path** —— 视频链接。

#### <font color="#0FB0E4">function **VideoReaderWrapper.len**</font>
() -> int

获取视频帧总数。

**Returns**
  - **int** —— 视频帧数。

#### <font color="#0FB0E4">function **VideoReaderWrapper.fps**</font>
() -> float

获取视频帧率

**Returns**
  - **float** —— 视频帧率。

#### <font color="#0FB0E4">function **VideoReaderWrapper.duration**</font>
() -> float

获取视频时长

**Returns**
  - **float** —— 视频时长。

#### <font color="#0FB0E4">function **VideoReaderWrapper.sample_frames**</font>
(decode_list: torch.Tensor) -> torch.Tensor

根据帧号，获取帧数据

**Parameters**
  - **decode_list** —— 采样帧号列表。
**Returns**
  - **tensor** —— 数据张量。

### <font color="#0FB0E4">class **FramesReaderWrapper**</font>
给定解完帧的文件夹，按顺序读取帧数据

#### <font color="#0FB0E4">function **FramesReaderWrapper.__init__**</font>
(frame_dir: str, extract_fps: float, suffix: str)

初始化视频实例

**Parameters**
  - **frame_dir** —— 帧文件夹。
  - **extract_fps** —— 提取帧的fps。
  - **suffix** —— 帧文件的后缀，默认为jpg。

#### <font color="#0FB0E4">function **FramesReaderWrapper.len**</font>
() -> int

获取视频帧总数。

**Returns**
  - **int** —— 视频帧数。

#### <font color="#0FB0E4">function **FramesReaderWrapper.fps**</font>
() -> float

获取视频帧率

**Returns**
  - **float** —— 视频帧率。

#### <font color="#0FB0E4">function **FramesReaderWrapper.duration**</font>
() -> float

获取视频时长

**Returns**
  - **float** —— 视频时长。

#### <font color="#0FB0E4">function **FramesReaderWrapper.sample_frames**</font>
(decode_list: torch.Tensor) -> torch.Tensor

根据帧号，获取帧数据

**Parameters**
  - **decode_list** —— 采样帧号列表。
**Returns**
  - **tensor** —— 数据张量。

### <font color="#0FB0E4">class **EasyVideoReader**</font>
用于长视频读取、采样和预处理的类。

#### <font color="#0FB0E4">function **EasyVideoReader.__init__**</font>
(video_path: str, num_frames: int, clip_duration: Union[float, Fraction, str],
overlap: Union[float, Fraction, str] = Fraction(0), transforms: Optional[Callable] = None)

初始化视频实例

**Parameters**
  - **video_path** —— 视频链接。
  - **num_frames** —— 视频帧数。
  - **clip_duration** —— 单片段长度。
  - **overlap** —— 片段间重合比例。
  - **transforms** —— 预处理算子。

#### <font color="#0FB0E4">function **EasyVideoReader.__iter__**</font>
() -> int

迭代器

#### <font color="#0FB0E4">function **EasyVideoReader.__next__**</font>
() -> float

迭代器，每迭代一次，返回一个片段的tensor。

**Returns**
  - **tensor** —— 视频片段的tensor。

## 8. 模块注册sdk(scepter.utils.registry)
用于管理各种注册的类。

### 基础用法

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
注册器

#### <font color="#0FB0E4">function **Registry.__init__**</font>
(name: str, build_func: function = None, common_para: Config = None, allow_types: tuple = ("class", "function"))

初始化注册模块实例

**Parameters**
  - **name** —— 模块名。
  - **build_func** —— build模块的时候调用的function。
  - **common_para** —— 该模块下的公共参数。
  - **allow_types** —— 该模块允许注册的类或者函数，默认都允许注册。

#### <font color="#0FB0E4">function **Registry.build**</font>
(cfg: Config, logger: logger = None, kwargs) -> cls_obj

build目标类的实例

**Returns**
  - **cls_obj** —— 特定类的实例。

#### <font color="#0FB0E4">function **Registry.register_class**</font>
(name: str)

注册一个类

**Returns**
  - **name** —— 注册名称。

#### <font color="#0FB0E4">function **Registry.register_function**</font>
(name: str)

注册一个函数

**Returns**
  - **name** —— 注册名称。

## 9. 数据sdk(scepter.utils.data)
用于数据在设备间转移

### 基础用法

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

将数据转移到numpy

**Parameters**
  - **data** —— torch.Tensor并以list/dict形式存储。
**Returns**
  - **data** —— numpy.ndarray并与输入一致的形式存储。

#### <font color="#0FB0E4">function **transfer_data_to_cpu**</font>
(data: list/dict of torch.Tensor(cuda)) -> (data: list/dict of torch.Tensor(cpu))

将gpu数据转移到cpu

**Parameters**
  - **data** —— torch.Tensor[CUDA]并以list/dict形式存储。
**Returns**
  - **data** —— torch.Tensor[CPU]并与输入一致的形式存储。

#### <font color="#0FB0E4">function **transfer_data_to_cuda**</font>
(data: list/dict of torch.Tensor(cpu)) -> (data: list/dict of torch.Tensor(cuda))

将cpu数据转移到gpu

**Parameters**
  - **data** —— torch.Tensor[CPU]并以list/dict形式存储。
**Returns**
  - **data** —— torch.Tensor[CUDA]并与输入一致的形式存储。

## 10. 模型sdk(torch.utils.model)
用于对模型进行加载、评估等操作

### 基础用法

```python
import torch
from scepter.utils.model import move_model_to_cpu, load_pretrained,
    count_params, init_weights
```
<hr/>


#### <font color="#0FB0E4">function **move_model_to_cpu**</font>
(params: list/dict of torch.Tensor[cuda]) -> (data: torch.Tensor[cpu])

将参数数据从gpu转移到cpu上。

**Parameters**
  - **params** —— torch.Tensor[cuda]并以OrderedDict形式存储。
**Returns**
  - **params** —— torch.Tensor[cpu]并与输入一致的形式存储。

#### <font color="#0FB0E4">function **load_pretrained**</font>
(model: torch.nn.Module, path: str, map_location="cpu", logger=None,
                    sub_level=None)

加载参数到模型。

**Parameters**
  - **model** —— torch.nn.Module模型实例。
  - **path** —— 预训练模型参数。
  - **map_location** —— cpu/cuda。
  - **logger** —— 标准日志实例。
  - **sub_level** —— 比如ddp时需要索引子层级。


#### <font color="#0FB0E4">function **count_params**</font>
(model: torch.nn.Module) -> (float)

统计模型的总参数。

**Parameters**
  - **model** —— torch.nn.Module模型实例。
**Returns**
  - **float** —— 模型参数量（浮点数个数）。

#### <font color="#0FB0E4">function **init_weights**</font>
(model: torch.nn.Module)

对模型模块进行参数初始化。

**Parameters**
  - **module** —— torch.nn.Module模型实例。

## 11. 采样器sdk(scepter.utils.sampler)
采样器比较具有通用性，大多数情况下不会进行定制开发，这里提供了几类常用的sampler采样器。

### 基础用法

```python
import torch
from scepter.utils.sampler import MultiFoldDistributedSampler,
    EvalDistributedSampler, MultiLevelBatchSampler, MixtureOfSamplers
```
<hr/>


#### <font color="#0FB0E4">class **MultiFoldDistributedSampler**</font>

多fold采样器，支持在一个epoch中重复多轮数据

#### <font color="#0FB0E4">function **MultiFoldDistributedSampler.__init__**</font>

( dataset: torch.data.dataset, num_folds=1, num_replicas=None, rank=None, shuffle=True)

**Parameters**

- **dataset** —— torch.data.dataset类实例
- **num_folds** —— int，表示数据重复的轮数。
- **num_replicas** —— 表示数据分割的片数，一般和world-size保持一致。
- **rank** —— rank表示当前进程号。
- **shuffle** —— 数据是否要打乱。

#### <font color="#0FB0E4">function **MultiFoldDistributedSampler.__iter__**</font>
()

迭代器，每迭代一次得到一个样本的index

#### <font color="#0FB0E4">function **MultiFoldDistributedSampler.set_epoch**</font>
(epoch: int)

设置当前的epoch

**Parameters**

- **epoch** —— 当前的epoch。


#### <font color="#0FB0E4">class **EvalDistributedSampler**</font>

用于测试时的采样器，当不用padding模式的时候，会发现最后一个rank的数据会少于其他rank。

#### <font color="#0FB0E4">function **EvalDistributedSampler.__init__**</font>

( dataset: torch.data.dataset, num_replicas: Optional[int] =None, rank: Optional[int] =None, padding： bool =False)

**Parameters**

- **dataset** —— torch.data.dataset类实例
- **num_replicas** —— 表示数据分割的片数，一般和world-size保持一致。
- **rank** —— rank表示当前进程号。
- **padding** —— 数据是否需要padding，如果padding则能保证最后一个rank和其他rank数据量一致。

#### <font color="#0FB0E4">function **EvalDistributedSampler.__iter__**</font>
()

迭代器，每迭代一次得到一个样本的index

#### <font color="#0FB0E4">function **EvalDistributedSampler.set_epoch**</font>
(epoch: int)

设置当前的epoch

**Parameters**

- **epoch** —— 当前的epoch。

#### <font color="#0FB0E4">class **MultiLevelBatchSampler**</font>

用于大规模数据的多级索引的sampler

#### <font color="#0FB0E4">function **MultiLevelBatchSampler.__init__**</font>

(index_file: str, batch_size: int, rank: int =0, seed: int = 8888)

**Parameters**

- **index_file** —— 多级数据索引的索引文件。
- **batch_size** —— 一个batch的大小。
- **rank** —— rank表示当前进程号。
- **seed** —— 随机采样的seed，在data.registry中获取全局seed。

#### <font color="#0FB0E4">function **MultiLevelBatchSampler.__iter__**</font>
()

迭代器，每迭代一次得到一个样本的index


#### <font color="#0FB0E4">class **MixtureOfSamplers**</font>

用于大规模数据的多级索引的sampler

#### <font color="#0FB0E4">function **MixtureOfSamplers.__init__**</font>

(samplers: list(sampler), probabilities: list(float), rank: int =0, seed: int = 8888)

**Parameters**

- **samplers** —— 采样器列表，用于混合采样器。
- **probabilities** —— 每个采样器的概率。
- **rank** —— rank表示当前进程号。
- **seed** —— 随机采样的seed，在data.registry中获取全局seed。

#### <font color="#0FB0E4">function **MixtureOfSamplers.__iter__**</font>
()

迭代器，每迭代一次得到一个样本的index

## 12. 探针器sdk(scepter.utils.probe)
用于探针各个组件的变量统计

### 基础用法

```python
import numpy as np
from scepter.model.base_model import BaseModel
from scepter.utils.config import Config
from scepter.utils.file_system import FS
from scepter.utils.probe import ProbeData


class TestModel(BaseModel):
    def forward(self, data):
        self.register_probe(data)
        # 测试ProbeData示例 + view_distribute
        self.register_probe(
            {"data_key_dist": ProbeData(data["data_key"], view_distribute=True),
             "data_folder": ProbeData(data["data_folder"], view_distribute=True)}
        )


class TestModel2(BaseModel):
    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger=logger)
        self.test_model = TestModel(cfg, logger=logger)

    def forward(self, data):
        # 测试list of str done
        # 测试dict of str done
        # 测试dict of number done
        # 测试list of number done
        # 测试number done
        # 测试str done
        # 测试np.array done
        # 测试list of np.ndarray 必须手动建立ProbeData done
        # 测试2D 图 必须手动建立ProbeData
        # 测试3D 图 必须手动建立ProbeData
        # 测试3D 多个2维图 必须手动建立ProbeData
        # 测试3D list 图 必须手动建立ProbeData
        # 测试4D Array 图 done
        # 测试4D Array 图 save_html
        # 测试4D List 图 save_html
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
        # 测试嵌套类型
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
    print(key, probe[key].to_log(prefix=f"xxx/dev_easytorch/{key}"))
```
<hr/>

配合Hook使用如下（其中PROB_INTERVAL探针存储间隔，即调用probe_data()的次数）：

```yaml
-
  NAME: ProbeDataHook
  PROB_INTERVAL: 100
```
#### <font color="#0FB0E4">class **ProbeData**</font>

探针数据的实例。

#### <font color="#0FB0E4">function **ProbeData.__init__**</font>

(data, is_image = False, build_html = False, build_label = None, view_distribute = False)

**Parameters**
- **data** —— 传入的探针数据，目前支持str、Number、list、dict、tensor。
- **is_image** —— 是否要存为图像。
- **build_html** —— 是否存为html。
- **build_label** —— 填入保存html的label html。
- **view_distribute** —— 针对一些值统计频率。
