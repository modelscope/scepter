# 优化器 (Optimizer)
## 总览
1. lr_schedulers
2. optimizers
<hr/>

## lr_schedulers
### 基础用法
子lr_schedulers继承时用法：

```python
from scepter.modules.opt.lr_schedulers import LR_SCHEDULERS
from scepter.modules.opt.lr_schedulers.base_scheduler import BaseScheduler


@LR_SCHEDULERS.register_class()
class XxxLR(BaseScheduler):
    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger=logger)
```
实际启动lr_scheduler用法(optimizer必要)，参考task/stable_diffusion/impls/solvers/diffusion_solver.py：
```python
if self.cfg.have("LR_SCHEDULER") and not self.optimizer is None:
    self.lr_scheduler = LR_SCHEDULERS.build(self.cfg.LR_SCHEDULER, logger=self.logger,
                                            optimizer=self.optimizer)
```

## **scepter.modules.opt.lr_schedulers.base_scheduler.BaseScheduler**
lr_schedulers的基类，支持注册操作，可根据需要自定义；

### <font color="#0FB0E4">function **\_\_init\_\_()**</font>
#### Parameters(输入参数)
(cfg: scepter.modules.utils.config.Config, logger = None) -> None
#### config(常用参数，实际需要根据不同schedulers设置，以StepLR为例):
* STEP_SIZE
* GAMMA
* LAST_EPOCH

### <font color="#0FB0E4">function **\_\_call\_\_()**</font>
#### Parameters(输入参数)
(optimizer: scepter.modules.opt.optimizers.OPTIMIZERS) -> None

具体对传入的optimizer对象进行schedule设置；
<hr/>

## optimizers
### 基础用法
子optimizers继承时用法：

```python
from scepter.modules.opt.optimizers.base_optimizer import BaseOptimize
from scepter.modules.opt.optimizers.registry import OPTIMIZERS


@OPTIMIZERS.register_class()
class Xxx(BaseOptimize):
    def __init__(self, cfg, logger=None):
        super(Xxx, self).__init__(cfg, logger=logger)
```
实际启动optimizers用法，参考task/stable_diffusion/impls/solvers/diffusion_solver.py，需要传入train_parameters：
```python
if self.cfg.have("OPTIMIZER"):
    self.optimizer = OPTIMIZERS.build(self.cfg.OPTIMIZER, logger=self.logger,
                                      parameters=self.train_parameters())
```

## **scepter.modules.opt.optimizers.base_optimizer.BaseOptimize**
optimizers的基类，支持注册操作，可根据需要自定义；

### <font color="#0FB0E4">function **\_\_init\_\_()**</font>
#### Parameters(输入参数)
(cfg: scepter.modules.utils.config.Config, logger = None) -> None
#### config(常用参数，实际需要根据不同optimizer设置，以SGD为例):
* LEARNING_RATE
* MOMENTUM
* DAMPENING
* WEIGHT_DECAY
* NESTEROV

### <font color="#0FB0E4">function **\_\_call\_\_()**</font>
#### Parameters(输入参数)
(parameters:dict()) -> None
输入需要梯度更新的train parameters，格式为dict；
