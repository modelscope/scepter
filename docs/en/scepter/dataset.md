# Dataset Module (Dataset)
## Overview
Register individual dataset reading modules for each task by inheriting from BaseDataset, which encapsulates File System operations and a transform pipeline.
***
## **scepter.modules.data.dataset.BaseDataset**
### <font color="#0FB0E4">function **\_\_init\_\_()**</font>
#### Parameters (Input Parameters)
`(cfg: scepter.modules.utils.config.Config, logger = None) -> None`
#### config:
- MODE —— [train, test, eval];
- TRANSFORMS —— For constructing a pipeline to process each sample. See docs.transforms for more details;
- FILE_SYSTEM —— File IO Handler, supports read and write operations for different file types. See docs.utils.file_clients for more details;
#### object (Internal)
- self.pipeline —— Transform pipeline
- self.fs_prefix —— The prefix of the instantiated fs_client
- self.local_we —— Local rank info when is_distributed

### <font color="#0FB0E4">function **\_\_getitem\_\_()**</font>
#### Parameters (Input Parameters): index
The type of index is determined by the sampler, refer to data/registry.py for details;
The default torch dataloader passes an integer type index as the dataset subscript;
Custom samplers can send customized items. Sampler definitions are found in scepter.modules.utils.sampler;
- The specific retrieval of a single data item is implemented by the _get() method with the index parameter;
- Outputs data that has been transformed by the pipeline (if there is a pipeline);

### <font color="#0FB0E4">function **worker_init_fn()**</font>
#### Parameters (Input Parameters):
`(worker_id, num_workers = 1)`
worker_id is the worker node ID in distributed training;
The dataloader instantiates num_workers worker processes at one time;
- For initializing the file reading system and setting parameters corresponding to multi-GPU workers;

### <font color="#0FB0E4">function **\_get()**</font>
#### Parameters (Input Parameters): index (passed in by __getitem__())
An abstract method that must be concretely implemented by the custom dataset. It's used to read each data item in a batch according to the index;
***
## Basic Usage
Subclass registration:
```python
from scepter.modules.data.dataset import BaseDataset
from scepter.modules.data.dataset import DATASETS
@DATASETS.register_class()
class XxxxxDataset(BaseDataset):
    def __init__(self, cfg, logger=None):
        super(XxxxxDataset, self).__init__(cfg, logger=logger)
```
Actual usage of starting a dataset:

```python
if self.cfg.have("TRAIN_DATA"):
    train_data = DATASETS.build(self.cfg.TRAIN_DATA, logger=self.logger)
```
Refer to scepter/modules/solver/diffusion_solver.py for more details. Specific parameter definitions are located under the TRAIN_DATA/EVAL_DATA/TEST_DATA sections in the configuration yaml.
