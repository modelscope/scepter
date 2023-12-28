# Optimizer (Optimizer)
## Overview
1. lr_schedulers
2. optimizers
<hr/>

## lr_schedulers
### Basic Usage
Usage when subclassing lr_schedulers:

```python
from scepter.opt.lr_schedulers import LR_SCHEDULERS
from scepter.opt.lr_schedulers.base_scheduler import BaseScheduler


@LR_SCHEDULERS.register_class()
class XxxLR(BaseScheduler):
    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger=logger)
```
Actual usage to start lr_scheduler (optimizer is necessary), refer to task/stable_diffusion/impls/solvers/diffusion_solver.py:
```python
if self.cfg.have("LR_SCHEDULER") and not self.optimizer is None:
    self.lr_scheduler = LR_SCHEDULERS.build(self.cfg.LR_SCHEDULER, logger=self.logger,
                                            optimizer=self.optimizer)
```

## **scepter.modules.opt.lr_schedulers.base_scheduler.BaseScheduler**
The base class for lr_schedulers, supports registration operations, can be customized as needed;

### <font color="#0FB0E4">function **\_\_init\_\_()**</font>
#### Parameters(input parameters)
(cfg: scepter.modules.utils.config.Config, logger = None) -> None
#### config(Common parameters, actual needs vary according to different schedulers, taking StepLR as an example):
* STEP_SIZE
* GAMMA
* LAST_EPOCH

### <font color="#0FB0E4">function **\_\_call\_\_()**</font>
#### Parameters(input parameters)
(optimizer: scepter.modules.opt.optimizers.OPTIMIZERS) -> None

Sets up the schedule for the passed-in optimizer object;
<hr/>

## optimizers
### Basic Usage
Usage when subclassing optimizers:

```python
from scepter.opt.optimizers.base_optimizer import BaseOptimize
from scepter.opt.optimizers.registry import OPTIMIZERS


@OPTIMIZERS.register_class()
class Xxx(BaseOptimize):
    def __init__(self, cfg, logger=None):
        super(Xxx, self).__init__(cfg, logger=logger)
```
Actual usage to start optimizers, refer to task/stable_diffusion/impls/solvers/diffusion_solver.py, requires passing in train_parameters:
```python
if self.cfg.have("OPTIMIZER"):
    self.optimizer = OPTIMIZERS.build(self.cfg.OPTIMIZER, logger=self.logger,
                                      parameters=self.train_parameters())
```

## **scepter.modules.opt.optimizers.base_optimizer.BaseOptimize**
The base class for optimizers, supports registration operations, can be customized as needed;

### <font color="#0FB0E4">function **\_\_init\_\_()**</font>
#### Parameters(input parameters)
(cfg: scepter.modules.utils.config.Config, logger = None) -> None
#### config(common parameters, actual needs vary according to different optimizers, taking SGD as an example):
* LEARNING_RATE
* MOMENTUM
* DAMPENING
* WEIGHT_DECAY
* NESTEROV

### <font color="#0FB0E4">function **\_\_call\_\_()**</font>
#### Parameters(input parameters)
(parameters:dict()) -> None
Inputs the train parameters that need gradient updates, in dict format;
