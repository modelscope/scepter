# Sampler Module (Sampler)
## Overview
A sampler defines the method for selecting the necessary data for training, validation, and testing.
Samplers are fairly generic and in most cases do not require custom development. Here, several commonly used samplers are provided.
Some samplers, such as MixtureOfSamplers, encapsulate a SubSampler.

Of course, samplers also support custom development and registration. Developers just need to inherit from BaseSampler and register their implemented sampler class.
## Basic Usage
Predefined samplers:
- TorchDefault
- LoopSampler
- MixtureOfSamplers
- MultiFoldDistributedSampler
- EvalDistributedSampler
- MultiLevelBatchSampler
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
Custom sampler, taking LoopSampler as an example:
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
Implement the iter and len methods to perform sampling. Related configuration parameters can be read from the cfg parameter in init.
<hr/>

## LoopSampler
An infinite looping sampler for simple data.
#### <font color="#0FB0E4">function **LoopSampler.__init__**</font>
(cfg: scepter.modules.utils.config.Config, logger=None) -> None

#### <font color="#0FB0E4">function **LoopSampler.__iter__**</font>
Iterator, each iteration yields the index of a sample.

## MixtureOfSamplers
A sampler for large-scale data with multi-level indexing.
#### <font color="#0FB0E4">function **MixtureOfSamplers.__init__**</font>
(samplers: list[sampler], probabilities: list[float], rank: int = 0, seed: int = 8888)

**Parameters**
- **samplers** —— List of samplers for mixing.
- **probabilities** —— Probabilities associated with each sampler.
- **rank** —— The rank representing the current process number.
- **seed** —— The seed for random sampling, obtained globally from data.registry.

#### <font color="#0FB0E4">function **MultiLevelBatchSampler.__iter__**</font>
()
Iterator, each iteration yields the index of a sample.

## MultiFoldDistributedSampler
A multi-fold sampler that supports repeating data several times within one epoch.
#### <font color="#0FB0E4">function **MultiFoldDistributedSampler.__init__**</font>
(dataset: torch.utils.data.Dataset, num_folds=1, num_replicas=None, rank=None, shuffle=True)

**Parameters**
- **dataset** —— An instance of the torch.utils.data.Dataset class.
- **num_folds** —— Integer, indicating the number of times data is repeated.
- **num_replicas** —— The number of data partitions, generally consistent with the world size.
- **rank** —— The rank representing the current process number.
- **shuffle** —— Whether to shuffle the data or not.

#### <font color="#0FB0E4">function **MultiFoldDistributedSampler.__iter__**</font>
()
Iterator, each iteration yields the index of a sample.

#### <font color="#0FB0E4">function **MultiFoldDistributedSampler.set_epoch**</font>
(epoch: int)
Sets the current epoch.
**Parameters**
- **epoch** —— The current epoch number.

## EvalDistributedSampler
A sampler for evaluation during testing, where you may find that the last rank has less data than other ranks when not using padding mode.
#### <font color="#0FB0E4">function **EvalDistributedSampler.__init__**</font>
(dataset: torch.utils.data.Dataset, num_replicas: Optional[int] = None, rank: Optional[int] = None, padding: bool = False)

**Parameters**
- **dataset** —— Instance of the torch.utils.data.Dataset class.
- **num_replicas** —— The number of data partitions, typically consistent with the world size.
- **rank** —— The rank representing the current process number.
- **padding** —— Whether the data needs padding. If true, it can ensure the last rank has the same amount of data as other ranks.

#### <font color="#0FB0E4">function **EvalDistributedSampler.__iter__**</font>
()
Iterator, each iteration yields the index of a sample.

#### <font color="#0FB0E4">function **EvalDistributedSampler.set_epoch**</font>
(epoch: int)
Sets the current epoch.

**Parameters**
- **epoch** —— The current epoch number.

## MultiLevelBatchSampler
A sampler for large-scale data with multi-level indexing.
#### <font color="#0FB0E4">function **MultiLevelBatchSampler.__init__**</font>
(index_file: str, batch_size: int, rank: int = 0, seed: int = 8888)

**Parameters**
- **index_file** —— The index file for multi-level data indexing.
- **batch_size** —— The size of a batch.
- **rank** —— The rank representing the current process number.
- **seed** —— The seed for random sampling, obtained globally from data.registry.

#### <font color="#0FB0E4">function **MultiLevelBatchSampler.__iter__**</font>
()
Iterator, each iteration yields the index of a sample.
