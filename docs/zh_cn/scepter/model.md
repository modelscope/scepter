# 模型模块 (Model)
## Overview
模型模块分为backbone、neck、head、loss、metric、network、tokenizer、tuner；
* backbone/neck：一般为提取feature主要模块（necks非必有）；
* head：根据不同任务类型，输入backbone提取的feature，输出下游任务所需logit；
* loss：用于计算不同类型loss；
* metric：用于计算各类评测指标；
* tokenizer：用于分词；
* tuner：用于创建微调模块；
* <font color="#0FB0E4">network</font>：train和test模块，对数据集输入的batch整合上述模块进行最终loss和指标计算；
<hr/>

## **backbone/neck/head/loss/tuner**
### Basic Usage
子类注册：

```python
from scepter.modules.model.registry import BACKBONES
from scepter.modules.model.base_model import BaseModel


@BACKBONES.register_class("ResNet")
class ResNet(BaseModel):
    def __init__(self, cfg, logger=None):
        super(ResNet, self).__init__(cfg, logger=logger)
```

```python
from scepter.modules.model.registry import NECKS
from scepter.modules.model.base_model import BaseModel


@NECKS.register_class()
class GlobalAveragePooling(BaseModel):
    def __init__(self, cfg, logger=None):
        super(GlobalAveragePooling, self).__init__(cfg, logger=logger)
```

```python
from scepter.modules.model.registry import HEADS
from scepter.modules.model.base_model import BaseModel


@HEADS.register_class()
class ClassifierHead(BaseModel):
    def __init__(self, cfg, logger=None):
        super(ClassifierHead, self).__init__(cfg, logger=logger)
```

```python
from scepter.modules.model.registry import LOSSES
import torch.nn as nn


@LOSSES.register_class()
class CrossEntropy(nn.Module):
    def __init__(self, cfg, logger=None):
        super(CrossEntropy, self).__init__(cfg, logger=logger)
```
实际调用：

```python
from scepter.modules.model.registry import BACKBONES, NECKS, HEADS, LOSSES, TUNERS

backbone = BACKBONES.build(cfg.BACKBONE, logger=logger)
neck = NECKS.build(cfg.NECK, logger=logger)
head = HEADS.build(cfg.HEAD, logger=logger)
loss = LOSSES.build(cfg.LOSS, logger=logger)
tuner = TUNERS.build(cfg.TUNER, logger=logger)
```

### <font color="#0FB0E4">function **\_\_init\_\_()**</font>
#### Parameters(输入参数)
(cfg: scepter.modules.utils.config.Config, logger = None) -> None

主要用于初始化模型各layer；

### <font color="#0FB0E4">function **forward()**</font>
根据需要具体实现；
<hr/>


## **metric**
### Basic Usage
子类注册：

```python
from scepter.modules.model.metrics.registry import METRICS
from scepter.modules.model.metrics.base_metric import BaseMetric


@METRICS.register_class("AccuracyMetric")
class AccuracyMetric(BaseMetric):
    def __init__(self, cfg, logger=None):
        super(CrossEntropy, self).__init__(cfg, logger=logger)
```
实际用法：

```python
from scepter.modules.model.metrics.registry import METRICS

metric = METRICS.build(cfgs, logger)
```

### <font color="#0FB0E4">function **\_\_init\_\_()**</font>
#### Parameters(输入参数)
(cfg: scepter.modules.utils.config.Config, logger = None) -> None

初始化计算metric所需超参，例如topk等系数；

### <font color="#0FB0E4">function **\_\_call\_\_()**</font>
@torch.no_grad()

通常输入logit和label以及其他所需要的变量，输出计算指标；
<hr/>

## **tokenizer**
### Basic Usage
子类注册：

```python
from scepter.modules.model.registry import TOKENIZERS
from scepter.modules.model.tokenizers import BaseTokenizer


@TOKENIZERS.register_class()
class BaseBertTokenizer(BaseTokenizer):
    def __init__(self, cfg, logger=None):
        super(BaseBertTokenizer, self).__init__(cfg, logger=logger)
```
实际用法：

```python
from scepter.modules.model.registry import TOKENIZERS

tokenizer = TOKENIZERS.build(cfgs, logger)
```
### <font color="#0FB0E4">function **\_\_init\_\_()**</font>
#### Parameters(输入参数)
(cfg: scepter.modules.utils.config.Config, logger = None) -> None
用于初始化加载tokenizer对象，例如BertTokenizer；

### <font color="#0FB0E4">function **tokenize()**</font>
输入需要分词的text list，输出分词后转换的token id sequence以及其他所需的attention mask/tpye id list/position id list等；
<hr/>

## **network**
### Basic Usage
子类注册：

```python
from scepter.modules.model.registry import MODELS
from scepter.modules.model.networks.train_module import TrainModule


@MODELS.register_class()
class Classifier(TrainModule):
    def __init__(self, cfg, logger=None):
        super(Classifier, self).__init__(cfg, logger=logger)
```
实际用法：

```python
from scepter.modules.model.registry import MODELS

model = MODELS.build(self.cfg.MODEL, logger=self.logger)
```

### <font color="#0FB0E4">function **\_\_init\_\_()**</font>
#### Parameters(输入参数)
(cfg: scepter.modules.utils.config.Config, logger = None) -> None

结合上述build方法，初始化训练所需的backbone、neck、head、loss、metric、tokenizer模块；

### <font color="#0FB0E4">function **forward_train()**</font>
输入训练batch的数据，经backbone、neck、head、loss计算相关loss；

### <font color="#0FB0E4">function **forward_test()**</font>
输入测试batch的数据，经backbone、neck、head、metrics计算相关指标；

### <font color="#0FB0E4">function **forward**</font>
实际调用接口，用于分发任务至forward_train()/forward_test();

其他训练/测试所需函数可在network下自定义；
