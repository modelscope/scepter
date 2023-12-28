# Solvers

## Overview
The Solver defines a workflow for model training, validation, and testing processes.
Within the Solver, based on the settings in the configuration YAML file, modules required such as data, model, optimizer, and scheduler are initialized one by one.
Each specific task's custom Solver must inherit from BaseSolver.

In some special scenarios, it is also necessary to initialize some custom modules. For example, Hooks are needed to record and save intermediate results during the training process; Metrics need to be defined for validation during training, and so on.


<hr/>

## Basic Usage

Usage of subclass Solver upon inheritance：

```python
from scepter.modules.solver.registry import SOLVERS
from scepter.modules.solver import BaseSolver


@SOLVERS.register_class()
class XxxSolver(BaseSolver):
    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger=logger)
```

The actual usage of starting the Solver can be found in ***scepter.modules/task/cate_recognition*** within ***run_train.py*** and ***run_inference.py***:

```python
solver = SOLVERS.build(cfg.SOLVER, logger=std_logger)
```

<hr/>

## **scepter.modules.solver.BaseSolver**
The base class for Solvers is an abstract base class defined through the ABCMeta metaclass, which supports registration operations. Custom solvers should all be subclasses of this class and be registered accordingly.

BaseSolver is a concrete example of a Solver implementation, showcasing how to write a Solver with and without the use of pytorch_lightning.

In practice, usage varies widely, so the application of Solvers is quite flexible. Most of its member functions can be overwritten in subclasses or enhanced with additional useful features to meet specific needs. Alternatively, completely new functions can be written to replace existing ones.
<br>

### <font color="#0FB0E4">function **\_\_init\_\_**</font>

(cfg: *scepter.modules.utils.config.Config*, logger = None) -> None

Initialize the Solver class, obtaining and defining some necessary parameters from the configuration (***cfg***).
Details of some parameters initialized in the **\_\_init\_\_** method can be seen in the code:

**Configs**

- **FILE_SYSTEM** —— (cfg) File system configuration, default None.
- **WORK_DIR** —— (str) Definition of the work directory.
- **LOG_FILE** —— (str) Location of the log file.
- **RESUME_FROM** —— (str) Path to intermediate results of the model from a previous training session.
- **MAX_EPOCHS** —— （int) Maximum number of training epochs.
- **ACCU_STEP** —— (int) When using ddp (distributed data parallel), the gradient accumulation steps for each process, default 1.
- **NUM_FOLDS** —— (int) Number of folds for training, default 1.
- **EVAL_INTERVAL** —— (int) Interval for evaluating the model, default 1.
- **EXTRA_KEYS** —— (List) The extra keys for metrics, default [].
- **TRAIN_DATA** —— (cfg) Training data configuration.
- **EVAL_DATA** —— (cfg) Validation data configuration.
- **TEST_DATA** —— (cfg) Test data configuration.
- **TRAIN_HOOKS** —— (List) Training hooks.
- **EVAL_HOOKS** —— (List) Validation hooks.
- **TEST_HOOKS** —— (List) Test hooks.
- **MODEL** —— (cfg) Model configuration.
- **OPTIMIZER** —— (cfg) Optimizer configuration.
- **LR_SCHEDULER** —— (cfg) Learning rate scheduler configuration.
- **METRICS** —— (List) Test Metrics.

**Parameters**

- **cfg** —— The Config used to build solver.
- **logger** —— Instantiated Logger to print or save log.

<br>

### <font color="#0FB0E4">function **set_up_pre**</font>

() -> None

Configure the environment, log path, and call construct_hook to initialize hooks, among other things.
Choose between this and pytorch_lightning; it is called when not using pytorch_lightning (use_pl=False). It needs to be called before initiating any other operations.

<br>

### <font color="#0FB0E4">function **set_up**</font>

() -> None

Configure data, model, metrics, optimizer, and the pytorch_lightning environment (if used).


<br>

### <font color="#0FB0E4">function **construct_data**</font>

() -> None

The actual method for constructing data, which by default is called within self.set_up, including TRAIN_DATA, EVAL_DATA, TEST_DATA. The instantiated results are written into self.datas.

<br>

### <font color="#0FB0E4">function **construct_hook**</font>

() -> None

The actual method for constructing hooks, which by default is called within self.set_up_pre, includes TRAIN_HOOKS, EVAL_HOOKS, TEST_HOOKS. The instantiated results are stored in self.hooks_dict.

<br>

### <font color="#0FB0E4">function **construct_model**</font>

() -> None

The actual method for constructing the Model, which by default is called within self.set_up, resulting in the instantiated model being assigned to self.model.

<br>

### <font color="#0FB0E4">function **model_to_device**</font>

() -> None

The actual method for constructing Metrics, which by default is called within self.set_up, and the instantiated results are stored in self.metrics.

<br>

### <font color="#0FB0E4">function **model_to_device**</font>

() -> None or MODELS

The method for configuring the model, including model sharding and other distributed settings, is typically called within self.set_up by default.

**Parameters**

- **tg_model_ins** —— The model to be configured. If it is None, self.model will be used by default.

**Returns**

- **tg_model_ins** —— The configured model. If tg_model_ins is None, there is no return value.

<br>

### <font color="#0FB0E4">function **init_opti**</font>

() -> None

The method for configuring the optimizer, which by default is called within self.set_up. The instantiated optimizer and learning rate scheduler are assigned to self.optimizer and self.lr_scheduler, respectively.

<br>

### <font color="#0FB0E4">function **solve**</font>

(epoch = None, every_epoch = False) -> None

Execute the actual training, validation, or testing operations by calling self.solve_train, self.solve_eval, self.solve_test, etc. Additionally, call self.before_solve and self.after_solve before and after execution to perform ***Hook*** logging.

**Parameters**

- **epoch** —— The number of epochs to set.
- **every_epoch** —— Related to the implementation of self.solve_train, self.solve_eval, self.solve_test, and the configuration of the data. It indicates whether it is necessary to call them again for each epoch. For example, with self.solve_train, if the implementation involves calling it once to execute one epoch, then every_epoch should be True. If a single call will execute until all epochs are finished, then it should be False.

<br>

### <font color="#0FB0E4">function **solve_train**</font>

() -> None

To execute the training process, you would call self.run_train. Additionally, you would invoke self.before_epoch and self.after_epoch before and after the training execution to carry out ***Hook*** logging.

<br>

### <font color="#0FB0E4">function **solve_eval**</font>

() -> None

Invoke self.run_eval to execute evaluation.
Additionally, call self.before_epoch and self.after_epoch respectively before and after the execution to log ***Hook***.

<br>

### <font color="#0FB0E4">function **solve_test**</font>

() -> None

Invoke self.run_test to execute testing.
Additionally, call self.before_epoch and self.after_epoch respectively before and after the execution to log ***Hook***.

<br>

### <font color="#0FB0E4">function **before_solve**</font>

() -> None

Before actually executing run_xxx, perform ***Hook*** logging.

<br>

### <font color="#0FB0E4">function **after_solve**</font>

() -> None

After executing run_xxx, perform ***Hook*** logging again.

<br>

### <font color="#0FB0E4">function **run_train**</font>

() -> None

Iteratively call self.run_step_train to execute the training process for one epoch or all epochs.
Additionally, call self.before_all_iter and self.after_all_iter respectively at the beginning and end of the loop to log ***Hook***.
Moreover, call self.before_iter and self.after_iter respectively before and after each iteration when calling self.run_step_train to log ***Hook***.

<br>

### <font color="#0FB0E4">function **run_eval**</font>

() -> None

Iteratively call self.run_step_eval to execute the validating process for one epoch or all epochs.
Additionally, call self.before_all_iter and self.after_all_iter respectively at the beginning and end of the loop to log ***Hook***.
Moreover, call self.before_iter and self.after_iter respectively before and after each iteration when calling self.run_step_eval to log ***Hook***.

<br>

### <font color="#0FB0E4">function **run_test**</font>

() -> None

Iteratively call self.run_step_test to execute the testing process for one epoch or all epochs.
Additionally, call self.before_all_iter and self.after_all_iter respectively at the beginning and end of the loop to log ***Hook***.
Moreover, call self.before_iter and self.after_iter respectively before and after each iteration when calling self.run_step_test to log ***Hook***.

<br>

### <font color="#0FB0E4">function **run_step_train**</font>

(batch_data, batch_idx = 0, step = None, rank = None) -> None

Perform model training for a single batch.

<br>

### <font color="#0FB0E4">function **run_step_eval**</font>

(batch_data, batch_idx = 0, step = None, rank = None) -> None

Perform model inference for a single batch.

<br>

### <font color="#0FB0E4">function **run_step_test**</font>

(batch_data, batch_idx = 0, step = None, rank = None) -> None

Perform model testing for a single batch.

<br>

### <font color="#0FB0E4">function **register_flops**</font>

(Dict data, keys = []) -> None

Use fvcore to calculate the FLOPs (floating-point operations per second), and save the result in self._model_flops.

**Parameters**

- **data** —— The input data for the model, formatted as a key-value structure, where the value is either a Tensor or a List. By default, only the first element along the batch dimension is taken to ensure batch_size = 1.
- **keys** —— Specifies the use of the values corresponding to the elements in keys as inputs to the model. If keys are empty, then all values in data are taken.

<br>

### <font color="#0FB0E4">function **before_epoch**</font>

() -> None

Execute before the start of each epoch. Perform before running run_xxx.

<br>

### <font color="#0FB0E4">function **before_all_iter**</font>

() -> None

Execute before the start of each epoch. Perform before the loop that iteratively calls run_step_xxx.

<br>

### <font color="#0FB0E4">function **before_iter**</font>

() -> None

Execute before the start of each step. Perform before running run_step_xxx.

<br>

### <font color="#0FB0E4">function **after_epoch**</font>

() -> None

Execute after the start of each epoch. Perform after running run_xxx.

<br>

### <font color="#0FB0E4">function **after_all_iter**</font>

() -> None

Execute after the end of each epoch. Perform after the loop that iteratively calls run_step_xxx.


<br>

### <font color="#0FB0E4">function **after_iter**</font>

() -> None

Execute after the start of each step. Perform after running run_step_xxx.

<br>

### <font color="#0FB0E4">function **collect_log_vars**</font>

() -> OrderedDict

Obtain the variables you need to save in logs

**Returns**

- **ret** —— Return the required variables.

<br>


# Hooks

## Overview

During the execution of the Solver, it is necessary to perform tasks such as printing logs, recording to Tensorboard, computing and updating gradients, saving intermediate model parameters, and saving test results, all of which require Hooks to execute.

<hr/>

## Basic Usage

Method for creating a new Hook:

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
A standard Hook base class is defined with a detailed list of member functions. Subclasses that inherit from this class need to select and implement some of these member functions.

The member functions include:

### <font color="#0FB0E4">function **__init__**</font>
(cfg: *scepter.modules.utils.config.Config*, logger = None) -> None
initialize logger。

### <font color="#0FB0E4">function **before_solve**</font>
(solver) -> None

Execute before starting to solve.

### <font color="#0FB0E4">function **after_solve**</font>
(solver) -> None

Execute after starting to solve.

### <font color="#0FB0E4">function **before_epoch**</font>
(solver) -> None

Execute before each epoch.

### <font color="#0FB0E4">function **after_epoch**</font>
(solver) -> None

Execute after each epoch.

### <font color="#0FB0E4">function **before_all_iter**</font>
(solver) -> None

Execute before each iteration.

### <font color="#0FB0E4">function **after_all_iter**</font>
(solver) -> None

Execute after each iteration.

### <font color="#0FB0E4">function **before_iter**</font>
(solver) -> None

Execute before each step.

### <font color="#0FB0E4">function **after_iter**</font>
(solver) -> None

Execute after each step.

## **scepter.modules.solver.hooks.CheckpointHook**
Before starting the solve, load the checkpoint. The path to load the model comes from the Solver's RESUME_FROM parameter. It depends on the load_checkpoint member function implemented in the Solver.

After the end of each epoch, save the checkpoint.

**Configs**

- **PRIORITY** —— （int）Default is _DEFAULT_CHECKPOINT_PRIORITY = 300
- **INTERVAL** —— （int）The interval of epochs between saving checkpoints, default is 1
- **SAVE_NAME_PREFIX** —— （str）The prefix name for saving checkpoints, default is 'ldm_step'
- **SAVE_LAST** —— （bool）Whether to save the checkpoint of the last iteration, default is False, applied in after_iter.
- **SAVE_BEST** —— （bool）Whether to save the best checkpoint, default is True, applied in after_epoch. SAVE_BEST_BY must also be set, otherwise it defaults back to False.
- **SAVE_BEST_BY** —— （str）The metric used to judge the best checkpoint, by default, the larger the better.

## **scepter.modules.solver.hooks.BackwardHook**
The actions executed after each iteration step. This includes backpropagation of the loss, configuration of the optimizer's steps, and so on.

**Configs**

- **PRIORITY** —— （int）_DEFAULT_BACKWARD_PRIORITY=0
- **GRADIENT_CLIP** —— （int）In torch.nn.utils.clip_grad_norm_, the max_norm parameter defaults to -1. It does not take effect when it is less than or equal to 0.
- **ACCUMULATE_STEP** —— （int）The gradient accumulation step count is used to specify how many forward/backward passes to accumulate gradients over before performing an optimizer step (gradient descent update).
- **EMPTY_CACHE_STEP** —— （int）The max_steps parameter for torch.cuda.empty_cache() defaults to -1. When set to a value less than or equal to 0, the cache clearing operation will not be performed.
-
## **scepter.modules.solver.hooks.LogHook**
Log hook。

**Configs**

- **PRIORITY** —— （int）__DEFAULT_LOG_PRIORITY=100
- **SHOW_GPU_MEM** —— （bool）To check whether to print memory usage information.
- **LOG_INTERVAL** —— （int）The interval of steps for printing logs, default is 10. It does not take effect when it is less than or equal to 0.

## **scepter.modules.solver.hooks.TensorboardLogHook**
Tensorboard log hook。

**Configs**

- **PRIORITY** —— （int）__DEFAULT_LOG_PRIORITY=100
- **LOG_DIR** —— （str）The path to store Tensorboard logs.
- **LOG_INTERVAL** —— （int）The interval of steps for printing logs, default is 1000.

## **scepter.modules.solver.hooks.LrHook**
The hook for changing the learning rate.

**Configs**

- **PRIORITY** —— （int）_DEFAULT_LR_PRIORITY=200
- **WARMUP_FUNC** —— （str）The method of warmup: only "linear" is supported, default is "linear".
- **WARMUP_EPOCHS** —— （int）The number of warmup epochs, default is 1.
- **WARMUP_START_LR** —— （float）the initial learning rate for warmup, default is 0.0001.
- **SET_BY_EPOCH** —— （bool）whether to set the learning rate once per epoch, default is True. If False, the learning rate is set at each step.

## **scepter.modules.solver.hooks.DistSamplerHook**
Before each epoch begins, sample data for one epoch.

**Configs**

- **PRIORITY** —— （int）__DEFAULT_SAMPLER_PRIORITY=400

## **scepter.modules.solver.hooks.ProbeDataHook**
Used to print the intermediate (visualization) results of train/eval stored by the probe.

**Configs**

- **PRIORITY** —— （int）_DEFAULT_LR_PRIORITY=200
- **PROB_INTERVAL** —— （int）The interval of steps for printing the probe data, default is 1000.

## **scepter.modules.solver.hooks.SafetensorsHook**
The ***.safetensors*** format is used to store model files

**Configs**

- **PRIORITY** —— （int）_DEFAULT_LR_PRIORITY=200
- **INTERVAL** —— （int）The interval of steps for saving .safetensors file, default is 1000.
- **SAVE_NAME_PREFIX** —— （str）The prefix name for saved files。
