# 训练器（Solvers）

## 总览
Solver是对模型训练、验证和测试过程的一个流程定义。
在Solver中，会根据配置yaml文件的设置，对数据（data），模型（model），优化器（optimizer）和调度（scheduler）等需要的模块进行逐一初始化。
每一个具体的任务的自定义Solver都要继承自BaseSolver。

在某些特殊场景下，还需要初始化一些自定义的模块。例如，在训练过程中记录保存中间结果需要用到Hooks；在训练过程中验证需要定义Metrics等等。

<hr/>

## 基础用法

子Solver继承时的用法：

```python
from scepter.modules.solver.registry import SOLVERS
from scepter.modules.solver import BaseSolver


@SOLVERS.register_class()
class XxxSolver(BaseSolver):
    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger=logger)
```

实际启动Solver的使用方式见scepter.modules/task/cate_recognition中run_task.py和run_inference.py:

```python
solver = SOLVERS.build(cfg.SOLVER, logger=std_logger)
```

<hr/>

## **scepter.modules.solver.BaseSolver**
Solver的基类，是通过元类ABCMeta定义的抽象基类，支持注册操作。自定义的solver均应该是该类的子类，并且进行注册。

BaseSolver是一个具体实现Solver的案例，展示了使用pytorch_lightning和不使用时两种Solver的写法。
在实际使用中情况各异，因此Solver的使用也比较灵活，其中大部分的成员函数均可以按照需求在子类中，被复写或者新加好用的功能，甚至直接写新的函数代替。

<br>

### <font color="#0FB0E4">function **\_\_init\_\_**</font>

(cfg: *scepter.modules.utils.config.Config*, logger = None) -> None

初始化Solver类，从cfg中获取并定义一些必要的参数。

部分在__init__中初始化的参数，详情见代码：

**Configs**

- **FILE_SYSTEM** —— (cfg) 文件系统配置，默认None.
- **WORK_DIR** —— (str) 工作路径定义.
- **LOG_FILE** —— (str) Log文件的位置.
- **RESUME_FROM** —— (str) 恢复训练模型保存的中间结果.
- **MAX_EPOCHS** —— （int) 训练最大epochs数.
- **ACCU_STEP** —— (int) When use ddp, the grad accumulate steps for each process，默认1.
- **NUM_FOLDS** —— (int) Num folds for training，默认1.
- **EVAL_INTERVAL** —— (int) Eval the model interval，默认1.
- **EXTRA_KEYS** —— (List) The extra keys for metrics，默认[].
- **TRAIN_DATA** —— (cfg) 训练数据配置.
- **EVAL_DATA** —— (cfg) 验证数据配置.
- **TEST_DATA** —— (cfg) 测试数据配置.
- **TRAIN_HOOKS** —— (List) 训练HOOKS.
- **EVAL_HOOKS** —— (List) 验证HOOKS.
- **TEST_HOOKS** —— (List) 测试HOOKS.
- **MODEL** —— (cfg) 模型配置.
- **OPTIMIZER** —— (cfg) 优化器配置.
- **LR_SCHEDULER** —— (cfg) 学习率Scheduler配置.
- **METRICS** —— (List) Metrics.

**Parameters**

- **cfg** —— The Config used to build solver.
- **logger** —— Instantiated Logger to print or save log.

<br>

### <font color="#0FB0E4">function **set_up_pre**</font>

() -> None

配置环境、日志路径，调用construct_hook来初始化hook等等。

与pytorch_lightning二选一，在不使用pytorch_lightning（use_pl=Flase）时调用。需要在启动其他所有操作之前调用。

<br>

### <font color="#0FB0E4">function **set_up**</font>

() -> None

配置数据、模型、metrics、优化器及pytorch_lightning环境（如果使用的话）。

<br>

### <font color="#0FB0E4">function **construct_data**</font>

() -> None

实际数据的构建方法，默认会在self.set_up中被调用，包括TRAIN_DATA、EVAL_DATA、TEST_DATA。将实例化的结果写入self.datas中。

<br>

### <font color="#0FB0E4">function **construct_hook**</font>

() -> None

实际Hook的构建方法，默认会在self.set_up_pre中被调用，包括TRAIN_HOOKS、EVAL_HOOKS、TEST_HOOKS。将实例化的结果写入self.hooks_dict中。

<br>

### <font color="#0FB0E4">function **construct_model**</font>

() -> None

实际Hook的构建方法，默认会在self.set_up中被调用，将实例化的结果作为self.model。

<br>

### <font color="#0FB0E4">function **model_to_device**</font>

() -> None

实际Metrics的构建方法，默认会在self.set_up中被调用，将实例化的结果写入self.metrics。

<br>

### <font color="#0FB0E4">function **model_to_device**</font>

(tg_model_ins=None) -> None or MODELS

模型的配置方法，包括模型分片等分布式配置，默认会在self.set_up中被调用。

**Parameters**

- **tg_model_ins** —— 待配置的model，如果为None，默认使用self.model。

**Returns**

- **tg_model_ins** —— 配置好的model。如果tg_model_ins为None，则无返回值。

<br>

### <font color="#0FB0E4">function **init_opti**</font>

() -> None

优化器的配置方法，默认会在self.set_up中被调用。将实例化的optimizer和lr_scheduler分别作为self.optimizer和self.lr_scheduler。

<br>

### <font color="#0FB0E4">function **solve**</font>

(epoch = None, every_epoch = False) -> None

执行实际的训练、验证或者测试操作，执行self.solve_train、self.solve_eval、self.solve_test等。
并且在执行前后分别调用self.before_solve、self.after_solve来进行Hook的记录。

**Parameters**

- **epoch** —— 设定epoch数量。
- **every_epoch** —— 与self.solve_train、self.solve_eval、self.solve_test的实现方式及Data的配置有关，标记是否每个epoch都需要重新调用一遍。
以self.solve_train为例，如果实现方式是调用一次执行一个epoch，则every_epoch应为True；如果调用一次调用会执行到所有epoch都结束，则应为False。

<br>

### <font color="#0FB0E4">function **solve_train**</font>

() -> None

调用self.run_train，执行train。
并且在执行前后分别调用self.before_epoch、self.after_epoch来进行Hook的记录。

<br>

### <font color="#0FB0E4">function **solve_eval**</font>

() -> None

调用self.run_eval，执行eval。
并且在执行前后分别调用self.before_epoch、self.after_epoch来进行Hook的记录。

<br>

### <font color="#0FB0E4">function **solve_test**</font>

() -> None

调用self.run_test，执行test。
并且在执行前后分别调用self.before_epoch、self.after_epoch来进行Hook的记录。

<br>

### <font color="#0FB0E4">function **before_solve**</font>

() -> None

在实际执行run_xxx之前，进行Hook的记录。

<br>

### <font color="#0FB0E4">function **after_solve**</font>

() -> None

在执行run_xxx之后，再次进行Hook的记录。

<br>

### <font color="#0FB0E4">function **run_train**</font>

() -> None

循环调用self.run_step_train，执行一个epoch或者所有epoch的训练过程。
并且在循环开始及结束后分别调用self.before_all_iter、self.after_all_iter来进行Hook的记录。
并且在每次循环调用self.run_step_train前后，分别调用self.before_iter、self.after_iter来进行Hook的记录。

<br>

### <font color="#0FB0E4">function **run_eval**</font>

() -> None

循环调用self.run_step_eval，执行一个epoch或者所有epoch的验证过程。
并且在循环开始及结束后分别调用self.before_all_iter、self.after_all_iter来进行Hook的记录。
并且在每次循环调用self.run_step_eval前后，分别调用self.before_iter、self.after_iter来进行Hook的记录。

<br>

### <font color="#0FB0E4">function **run_test**</font>

() -> None

循环调用self.run_step_test，执行一个epoch或者所有epoch的验证过程。
并且在循环开始及结束后分别调用self.before_all_iter、self.after_all_iter来进行Hook的记录。
并且在每次循环调用self.run_step_test前后，分别调用self.before_iter、self.after_iter来进行Hook的记录。

<br>

### <font color="#0FB0E4">function **run_step_train**</font>

(batch_data, batch_idx = 0, step = None, rank = None) -> None

执行一个batch的模型推理。

<br>

### <font color="#0FB0E4">function **run_step_eval**</font>

(batch_data, batch_idx = 0, step = None, rank = None) -> None

执行一个batch的模型推理。

<br>

### <font color="#0FB0E4">function **run_step_test**</font>

(batch_data, batch_idx = 0, step = None, rank = None) -> None

执行一个batch的模型推理。

<br>

### <font color="#0FB0E4">function **register_flops**</font>

(Dict data, keys = []) -> None

使用fvcore进行flops计算，结果保存在self._model_flops中。

**Parameters**

- **data** —— 输入模型的data，格式为key-value结构，value为Tensor或者List，默认只取batch维度的第一个元素来保证batchsize=1。
- **keys** —— 指定使用keys中的元素所对应的value作为模型输入。如果keys为空，则去data中所有的value。

<br>

### <font color="#0FB0E4">function **before_epoch**</font>

() -> None

在每个epoch开始前执行。run_xxx前执行。

<br>

### <font color="#0FB0E4">function **before_all_iter**</font>

() -> None

在每个epoch开始前执行。循环调用run_step_xxx的循环前执行。

<br>

### <font color="#0FB0E4">function **before_iter**</font>

() -> None

在每个step开始前执行。run_step_xxx前执行。

<br>

### <font color="#0FB0E4">function **after_epoch**</font>

() -> None

在每个epoch开始后执行。run_xxx后执行。

<br>

### <font color="#0FB0E4">function **after_all_iter**</font>

() -> None

在每个epoch开始后执行。循环调用run_step_xxx的循环后执行。

<br>

### <font color="#0FB0E4">function **after_iter**</font>

() -> None

在每个step开始后执行。run_step_xxx后执行。

<br>

### <font color="#0FB0E4">function **collect_log_vars**</font>

() -> OrderedDict

获取需要在log中保存的变量。

**Returns**

- **ret** —— 返回需要的变量。

<br>


# Hooks

## 总览

在Solver执行过程中，需要打印日志、记录Tensorboard、梯度计算和更新、保存中间模型参数、保存测试结果等，这些都需要Hook去执行。

<hr/>

## 基础用法

新建Hook的方法：

```python
from scepter.modules.solver.hooks.hook import Hook
from scepter.modules.solver.hooks.registry import HOOKS


@HOOKS.register_class()
class XxxHook(Hook):
    def __init__(self, cfg, logger=None):
        super(XxxHook, self).__init__(cfg, logger=logger)
```

<hr/>


## **scepter.modules.solvers.hooks.Hook**
定义了一个标准的Hook类的基类，定义了详细的成员函数列表。继承自该类的子类均需要在这些成员函数中挑选部分进行实现。
成员函数包括：

### <font color="#0FB0E4">function **__init__**</font>
(cfg: *scepter.modules.utils.config.Config*, logger = None) -> None
初始化logger。

### <font color="#0FB0E4">function **before_solve**</font>
(solver) -> None

开始solve之前执行。

### <font color="#0FB0E4">function **after_solve**</font>
(solver) -> None

结束solve之后执行。

### <font color="#0FB0E4">function **before_epoch**</font>
(solver) -> None

每个epoch之前执行。

### <font color="#0FB0E4">function **after_epoch**</font>
(solver) -> None

每个epoch之后执行。

### <font color="#0FB0E4">function **before_all_iter**</font>
(solver) -> None

迭代开始之前执行。

### <font color="#0FB0E4">function **after_all_iter**</font>
(solver) -> None

迭代结束之后执行。

### <font color="#0FB0E4">function **before_iter**</font>
(solver) -> None

每个step之前执行。

### <font color="#0FB0E4">function **after_iter**</font>
(solver) -> None

每个step之后执行。

## **scepter.modules.solver.hooks.CheckpointHook**
在solve开始之前，加载checkpoint。加载模型的路径来自Solver的RESUME_FROM参数。依赖于Solver中实现的load_checkpoint成员函数。

每个epoch结束之后，保存checkpoint。

**Configs**

- **PRIORITY** —— （int）默认为_DEFAULT_CHECKPOINT_PRIORITY=300
- **INTERVAL** —— （int）保存checkpoint的epoch间隔，默认为1
- **SAVE_NAME_PREFIX** —— （str）保存checkpoint的前缀名，默认为'ldm_step'
- **SAVE_LAST** —— （bool）是否保存最后iter的checkpoint，默认为False，作用于after_iter中。
- **SAVE_BEST** —— （bool）是否保存最好的checkpoint，默认为True，作用于after_epoch中。需要同时设置SAVE_BEST_BY，否则退回默认False。
- **SAVE_BEST_BY** —— （str）判断最好的指标，默认越大越好

## **scepter.modules.solver.hooks.BackwardHook**
在每步迭代之后执行的内容。包括loss的反传，optimizer的步数配置等等。

**Configs**

- **PRIORITY** —— （int）_DEFAULT_BACKWARD_PRIORITY=0
- **GRADIENT_CLIP** —— （int）torch.nn.utils.clip_grad_norm_中max_norm参数，默认为-1。小于等于0时不生效。
- **ACCUMULATE_STEP** —— （int）用于设置梯度累计的步数
- **EMPTY_CACHE_STEP** —— （int）torch.cuda.empty_cache每多少步清除一下memory，默认为-1。小于等于0时不生效。
-
## **scepter.modules.solver.hooks.LogHook**
日志Hook。

**Configs**

- **PRIORITY** —— （int）__DEFAULT_LOG_PRIORITY=100
- **SHOW_GPU_MEM** —— （bool）判断是否打印内存使用情况
- **LOG_INTERVAL** —— （int）打印log的步数间隔，默认为10。小于等于0时不生效。

## **scepter.modules.solver.hooks.TensorboardLogHook**
Tensorboard日志Hook。

**Configs**

- **PRIORITY** —— （int）__DEFAULT_LOG_PRIORITY=100
- **LOG_DIR** —— （str）存储Tensorboard log的路径。
- **LOG_INTERVAL** —— （int）打印log的步数间隔，默认为1000。

## **scepter.modules.solver.hooks.LrHook**
学习率变化Hook。

**Configs**

- **PRIORITY** —— （int）_DEFAULT_LR_PRIORITY=200
- **WARMUP_FUNC** —— （str）warmup的方式：仅支持"linear"，默认为"linear"。
- **WARMUP_EPOCHS** —— （int）warmup epoch数，默认为1。
- **WARMUP_START_LR** —— （float）warmup初始学习率，默认为0.0001。
- **SET_BY_EPOCH** —— （bool）是否每个epoch设置一次学习率，默认为True。False则每个step设置一次。

## **scepter.modules.solver.hooks.DistSamplerHook**
每个epoch开始之前，采样一个epoch的数据。

**Configs**

- **PRIORITY** —— （int）__DEFAULT_SAMPLER_PRIORITY=400

## **scepter.modules.solver.hooks.ProbeDataHook**
用于打印probe存储的train/eval中间（可视化）结果。

**Configs**

- **PRIORITY** —— （int）_DEFAULT_LR_PRIORITY=200
- **PROB_INTERVAL** —— （int）打印probe的步数间隔，默认为1000。

## **scepter.modules.solver.hooks.SafetensorsHook**
用于存储.safetensors格式的模型文件。

**Configs**

- **PRIORITY** —— （int）_DEFAULT_LR_PRIORITY=200
- **INTERVAL** —— （int）存储的步数间隔，默认为1000。
- **SAVE_NAME_PREFIX** —— （str）保存文件的前缀名。
