
# 采样器模块(Sampler)

## Overview
采样器定义了选择训练、验证和测试所需数据的采样方式。
采样器比较具有通用性，大多数情况下不会进行定制开发，这里提供了几类常用的sampler采样器。
其中部分采样器例如MixtureOfSamplers封装有SubSampler。

当然，采样器也支持定制开发并注册，开发者只需要在继承BaseSampler基础上对实现的sampler类进行注册。

## Basic Usage

预定义samplers：
- TorchDefault
- LoopSampler,
- MixtureOfSamplers,
- MultiFoldDistributedSampler,
- EvalDistributedSampler,
- MultiLevelBatchSampler,
- MultiLevelBatchSamplerMultiSource
```python
from scepter.modules.data.sampler import (
    LoopSampler,
    MixtureOfSamplers,
    MultiFoldDistributedSampler,
    EvalDistributedSampler,
    MultiLevelBatchSampler,
    MultiLevelBatchSamplerMultiSource)
```
自定义sampler，以LoopSampler为例：
```python
@SAMPLERS.register_class()
class LoopSampler(BaseSampler):
    para_dict = {}

    def __init__(self, cfg, logger):
        super().__init__(cfg, logger)
        rank = we.rank
        self.rng = np.random.default_rng(self.seed + rank)

    def __iter__(self):
        while True:
            yield self.rng.choice(sys.maxsize)

    def __len__(self):
        return sys.maxsize

    @staticmethod
    def get_config_template():
        return dict_to_yaml('SAMPLERS',
                            __class__.__name__,
                            LoopSampler.para_dict,
                            set_name=True)
```
实现__iter__和__len__方法实现采样功能，在__init__的入参cfg中可以读取相关配置参数。
<hr/>

## LoopSampler
简单数据的无限循环sampler。

#### <font color="#0FB0E4">function **LoopSampler.__init__**</font>
(cfg: scepter.modules.utils.config.Config, logger = None) -> None

#### <font color="#0FB0E4">function **LoopSampler.__iter__**</font>
迭代器，每迭代一次得到一个样本的index


## MixtureOfSamplers

用于大规模数据的多级索引的sampler

#### <font color="#0FB0E4">function **MixtureOfSamplers.__init__**</font>

(samplers: list(sampler), probabilities: list(float), rank: int =0, seed: int = 8888)

**Parameters**

- **samplers** —— 采样器列表，用于混合采样器。
- **probabilities** —— 每个采样器的概率。
- **rank** —— rank表示当前进程号。
- **seed** —— 随机采样的seed，在data.registry中获取全局seed。

#### <font color="#0FB0E4">function **MultiLevelBatchSampler.__iter__**</font>
()

迭代器，每迭代一次得到一个样本的index

## MultiFoldDistributedSampler

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


## EvalDistributedSampler

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

## MultiLevelBatchSampler
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
