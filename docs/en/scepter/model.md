# Model Modules (Model)
## Overview
Model modules are divided into backbones, necks, heads, loss, metrics, networks, tokenizers;
* backbones/necks：Generally the main modules for feature extraction (necks are not always present);
* heads: According to different task types, they take the features extracted by backbones and output the logits required for downstream tasks;
* loss: Used to calculate different types of loss;
* metric: Used to calculate various evaluation metrics;
* tokenizer: Used for tokenization;
* <font color="#0FB0E4">network</font>：train和test模块，对数据集输入的batch整合上述模块进行最终loss和指标计算；
<hr/>

## **backbones/necks/heads/loss**
### Basic Usage
Subclass registration:

```python
from scepter.model.registry import BACKBONES
from scepter.model.base_model import BaseModel


@BACKBONES.register_class("ResNet")
class ResNet(BaseModel):
    def __init__(self, cfg, logger=None):
        super(ResNet, self).__init__(cfg, logger=logger)
```

```python
from scepter.model.registry import NECKS
from scepter.model.base_model import BaseModel


@NECKS.register_class()
class GlobalAveragePooling(BaseModel):
    def __init__(self, cfg, logger=None):
        super(GlobalAveragePooling, self).__init__(cfg, logger=logger)
```

```python
from scepter.model.registry import HEADS
from scepter.model.base_model import BaseModel


@HEADS.register_class()
class ClassifierHead(BaseModel):
    def __init__(self, cfg, logger=None):
        super(ClassifierHead, self).__init__(cfg, logger=logger)
```

```python
from scepter.model.registry import LOSSES
import torch.nn as nn


@LOSSES.register_class()
class CrossEntropy(nn.Module):
    def __init__(self, cfg, logger=None):
        super(CrossEntropy, self).__init__(cfg, logger=logger)
```
Actual usage：

```python
from scepter.model.registry import BACKBONES, NECKS, HEADS, LOSSES

backbone = BACKBONES.build(cfg.BACKBONE, logger=logger)
neck = NECKS.build(cfg.NECK, logger=logger)
head = HEADS.build(cfg.HEAD, logger=logger)
loss = LOSSES.build(cfg.LOSS, logger=logger)
```

### <font color="#0FB0E4">function **\_\_init\_\_()**</font>
#### Parameters(input parameters)
(cfg: scepter.modules.utils.config.Config, logger = None) -> None

Mainly used for initializing various layers of the model;

### <font color="#0FB0E4">function **forward()**</font>
To be implemented specifically as needed;
<hr/>


## **metrics**
### Basic Usage
Basic Usage Subclass registration:

```python
from scepter.model.metrics.registry import METRICS
from scepter.model.metrics.base_metric import BaseMetric


@METRICS.register_class("AccuracyMetric")
class AccuracyMetric(BaseMetric):
    def __init__(self, cfg, logger=None):
        super(CrossEntropy, self).__init__(cfg, logger=logger)
```
Actual usage:

```python
from scepter.model.metrics.registry import METRICS

metric = METRICS.build(cfgs, logger)
```

### <font color="#0FB0E4">function **\_\_init\_\_()**</font>
#### Parameters(input parameters)
(cfg: scepter.modules.utils.config.Config, logger = None) -> None

Initializes the hyperparameters needed for calculating metrics, such as coefficients like topk;

### <font color="#0FB0E4">function **\_\_call\_\_()**</font>
@torch.no_grad()

Typically takes logits and labels as well as other necessary variables as inputs and outputs calculated metrics;
<hr/>

## **tokenizers**
### Basic Usage
Subclass registration:

```python
from scepter.model.registry import TOKENIZERS
from scepter.model.tokenizers import BaseTokenizer


@TOKENIZERS.register_class()
class BaseBertTokenizer(BaseTokenizer):
    def __init__(self, cfg, logger=None):
        super(BaseBertTokenizer, self).__init__(cfg, logger=logger)
```
Actual usage：

```python
from scepter.model.registry import TOKENIZERS

tokenizer = TOKENIZERS.build(cfgs, logger)
```
### <font color="#0FB0E4">function **\_\_init\_\_()**</font>
#### Parameters(input parameters)
(cfg: scepter.modules.utils.config.Config, logger = None) -> None
None Used for initializing and loading the tokenizer object, such as BertTokenizer;

### <font color="#0FB0E4">function **tokenize()**</font>
Takes a list of texts that need tokenization as input and outputs token id sequences after tokenization, as well as other necessary elements like attention masks, type id lists, position id lists, etc;
<hr/>

## **networks**
### Basic Usage
Subclass registration:

```python
from scepter.model.registry import MODELS
from scepter.model.networks.train_module import TrainModule


@MODELS.register_class()
class Classifier(TrainModule):
    def __init__(self, cfg, logger=None):
        super(Classifier, self).__init__(cfg, logger=logger)
```
Actual usage：

```python
from scepter.model.registry import MODELS

model = MODELS.build(self.cfg.MODEL, logger=self.logger)
```

### <font color="#0FB0E4">function **\_\_init\_\_()**</font>
#### Parameters(input parameters)
(cfg: scepter.modules.utils.config.Config, logger = None) -> None

Combined with the above build method, initializes the backbone, neck, head, loss, metric, tokenizer modules needed for training;

### <font color="#0FB0E4">function **forward_train()**</font>
Takes data from a training batch, processes it through the backbone, neck, head, and loss to calculate the relevant loss;

### <font color="#0FB0E4">function **forward_test()**</font>
Takes data from a test batch, processes it through the backbone, neck, head, and metrics to calculate the relevant metrics;

### <font color="#0FB0E4">function **forward**</font>
The actual calling interface, used to dispatch tasks to forward_train()/forward_test();
Other functions needed for training/testing can be customized under network;"
