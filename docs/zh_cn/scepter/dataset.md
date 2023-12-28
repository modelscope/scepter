# 数据集模块 (Dataset)

## 总览
在继承BaseDataset基础上注册每个task各自的数据集读取模块，BaseDataset中封装有File System以及transform pipeline；

<hr/>

## **scepter.modules.data.dataset.BaseDataset**
### <font color="#0FB0E4">function **\_\_init\_\_()**</font>
#### Parameters(输入参数)
(cfg: scepter.modules.utils.config.Config, logger = None) -> None
#### config:
* MODE —— [train, test, eval]；
* TRANSFORMS —— 用于构造pipeline处理每个样本,详见docs.transforms；
* FILE_SYSTEM —— File IO Handler，支持不同类型文件的读写操作，详见docs.utils.file_clients；

#### object(内部)
* self.pipeline —— transform pipeline
* self.fs_prefix —— The prefix of instantiated fs_client
* self.local_we —— local rank info when is_distributed
### <font color="#0FB0E4">function **\_\_getitem\_\_()**</font>
#### Parameters(输入参数)：index
index的类别由sampler确定，详见data/registry.py；

一般默认的torch dataloader传入的index为int型作为dataset下标；

自定义的sampler则可以传入自定义item，sampler定义参照scepter.modules.utils.sampler；

* 具体读取单条数据由_get()方法传入index参数实现；
* 输出经由pipeline转换后的数据（如果有pipeline）；

### <font color="#0FB0E4">function **worker_init_fn()**</font>
#### Parameters(输入参数)：
(worker_id, num_workers = 1)

worker_id为分布式训练中工作节点id；

dataloader一次性创建num_workers个工作进程;
* 用于初始化文件读取系统和设置多卡worker对应参数；

### <font color="#0FB0E4">function **\_get()**</font>
#### Parameters(输入参数)：index（由__getitem__()传入）
抽象方法，需要由自定义dataset继承具体实现，用于根据index读取batch中每条数据；
<hr/>

## 基础用法
子类注册：

```python
from scepter.modules.data.dataset import BaseDataset
from scepter.modules.data.dataset import DATASETS


@DATASETS.register_class()
class XxxxxDataset(BaseDataset):
    def __init__(self, cfg, logger=None):
        super(XxxxxDataset, self).__init__(cfg, logger=logger)
```
实际启动dataset用法：

```python
if self.cfg.have("TRAIN_DATA"):
    train_data = DATASETS.build(self.cfg.TRAIN_DATA, logger=self.logger)
```
详见scepter/modules/solver/diffusion_solver.py，具体参数定义在配置yaml中TRAIN_DATA/EVAL_DATA/TEST_DATA模块下；
