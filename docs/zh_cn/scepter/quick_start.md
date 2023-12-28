# 快速开始 （Quick Start）

本章节以stable diffusion v1.5为例，介绍如何从零开始构建一个网络结构，以及基于该结构训练一个模型和测试该模型。

# 1. 使用network类定义模型结构

network类包含了定义、训练和测试模型的方法，我们首先初始化一个network类，然后在里面定义所需的autoencoder, unet, embedder, 以及loss子模块。

```python
@MODELS.register_class()
class LatentDiffusion(TrainModule):
    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger=logger)
        self.model_config = cfg.DIFFUSION_MODEL
        self.first_stage_config = cfg.FIRST_STAGE_MODEL
        self.cond_stage_config = cfg.COND_STAGE_MODEL
        self.loss_config = cfg.get('LOSS', None)

        self.model = BACKBONES.build(self.model_config, logger=self.logger)
        self.first_stage_model = MODELS.build(self.first_stage_config,
                                              logger=self.logger)
        self.cond_stage_model = EMBEDDERS.build(self.cond_stage_config,
                                                logger=self.logger)
        if self.loss_config:
            self.loss = LOSSES.build(self.loss_config, logger=self.logger)

        # 其他变量和模块定义
```

# 2. 实现自定义network类的训练和测试方法

每个network类依赖forward_train和forward_test方法定义自己的训练和测试流程，在sd1.5中，forward_train对采样时刻t进行噪声预测以及进行loss计算

```python
 def forward_train(self, image=None, noise=None, prompt=None, **kwargs):
    x_start = self.encode_first_stage(image, **kwargs)
    t = torch.randint(0,
                        self.num_timesteps, (x_start.shape[0], ),
                        device=x_start.device).long()
    context = {}
    if prompt and self.cond_stage_model:
        zeros = (torch.rand(len(prompt)) < self.p_zero).numpy().tolist()
        prompt = [
            self.train_n_prompt if zeros[idx] else p
            for idx, p in enumerate(prompt)
        ]

        with torch.autocast(device_type='cuda', enabled=False):
            context = self.encode_condition(
                self.tokenizer(prompt).to(we.device_id))

    loss = self.diffusion.loss(x0=x_start,
                                t=t,
                                model=self.model,
                                model_kwargs={'cond': context},
                                noise=noise)
    loss = loss.mean()
    ret = {'loss': loss, 'probe_data': {'prompt': prompt}}
    return ret
```

forward_test函数用于推理阶段执行完整的图像去噪过程

```python
@torch.no_grad()
@torch.autocast('cuda', dtype=torch.float16)
def forward_test(self,
                    prompt=None,
                    n_prompt=None,
                    sampler='ddim',
                    sample_steps=50,
                    seed=2023,
                    guide_scale=7.5,
                    guide_rescale=0.5,
                    discretization='trailing',
                    run_train_n=True,
                    **kwargs):
    g = torch.Generator(device=we.device_id)
    seed = seed if seed >= 0 else random.randint(0, 2**32 - 1)
    g.manual_seed(seed)
    num_samples = len(prompt)

    n_prompt = default(n_prompt, [self.default_n_prompt] * len(prompt))
    assert isinstance(prompt, list) and \
            isinstance(n_prompt, list) and \
            len(prompt) == len(n_prompt)

    context = self.encode_condition(self.tokenizer(prompt).to(
        we.device_id),  method='encode_text')
    null_context = self.encode_condition(self.tokenizer(n_prompt).to(
        we.device_id), method='encode_text')

    width, height = 512, 512
    noise = self.noise_sample(num_samples, width // self.size_factor,
                                height // self.size_factor, g)
    # UNet use input n_prompt
    samples = self.diffusion.sample(solver=sampler,
                                    noise=noise,
                                    model=self.model,
                                    model_kwargs=[{
                                        'cond': context
                                    }, {
                                        'cond': null_context
                                    }],
                                    steps=sample_steps,
                                    guide_scale=guide_scale,
                                    guide_rescale=guide_rescale,
                                    discretization=discretization,
                                    show_progress=True,
                                    seed=seed,
                                    condition_fn=None,
                                    clamp=None,
                                    percentile=None,
                                    t_max=None,
                                    t_min=None,
                                    discard_penultimate_step=None,
                                    return_intermediate=None,
                                    **kwargs)
    x_samples = self.decode_first_stage(samples).float()
    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

    outputs = list()
    for p, np, img in zip(prompt, n_prompt, x_samples):
        one_tup = {'prompt': p, 'n_prompt': np, 'image': img}
        outputs.append(one_tup)

    return outputs
```

# 3. 子模块注册

在实现完network类之后，需要确保network类中用到的所有子模块都已完成注册。以sd1.5中的embedder为例，为了能在network的初始化方法中实例化该embedder，我们需要先实现该embedder类，并注册到scepter中

```python
@EMBEDDERS.register_class()
class FrozenCLIPEmbedder(BaseEmbedder):
    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger=logger)

        with FS.get_dir_to_local_dir(cfg.PRETRAINED_MODEL,
                                     wait_finish=True) as local_path:
            self.tokenizer = CLIPTokenizer.from_pretrained(local_path)
            self.transformer = CLIPTextModel.from_pretrained(local_path)

        self.use_grad = cfg.get('USE_GRAD', False)
        self.freeze_flag = cfg.get('FREEZE', True)
        if self.freeze_flag:
            self.freeze()

        self.max_length = cfg.get('MAX_LENGTH', 77)
        self.layer = cfg.get('LAYER', 'last')
        self.layer_idx = cfg.get('LAYER_IDX', None)
        self.use_final_layer_norm = cfg.get('USE_FINAL_LAYER_NORM', False)
        assert self.layer in self.LAYERS
        if self.layer == 'hidden':
            assert self.layer_idx is not None
            assert 0 <= abs(self.layer_idx) <= 12


    def encode_text(self,
                    tokens,
                    tokenizer=None,
                    append_sentence_embedding=False):
        # 定义一些需要的方法
        pass

    @staticmethod
    def get_config_template():
        return dict_to_yaml('MODELS',
                            __class__.__name__,
                            FrozenCLIPEmbedder.para_dict,
                            set_name=True)
```

# 4. Solver注册

solver类中封装了训练和测试一个network所需要的完整流程。以sd1.5为例，注册一个训练sd1.5模型的solver需要:
1. 创建一个或多个数据加载器(data_loader), 对应solver中的construct_data方法；
2. 实例化一个模型，对应solver中的construct_model方法
3. （可选）定义度量指标，对应solver中的construct_metrics方法
4. （可选）定义训练和测试用到的一些钩子(HOOKS)，比如模型保存，预训练参数加载，日志打印，等等。

```python
@SOLVERS.register_class()
class LatentDiffusionSolver(BaseSolver):
    def set_up(self):
        self.construct_data()
        self.construct_model()
        self.construct_metrics()
        self.model_to_device()
        self.init_opti()

    def load_checkpoint(self, checkpoint):
        # 这里定义加载模型的指令

    def save_checkpoint(self):
        # 这里定义保存模型的指令

    def solve(self):
        # 入口函数，根据数据类型选择执行训练或测试
        self.before_solve()
        if 'train' in self._mode_set:
            self.run_train()
        if 'test' in self._mode_set:
            self.run_test()
        self.after_solve()

    def run_train(self):
        # 模型训练

    def run_eval(self):
        # 模型验证

    def run_test(self):
        # 模型测试
```

# 5. 训练/测试超参数定义

在注册完所需的各个组件（包括但不限于BACKBONE, NETWORK, EMBEDDER, SOLVER, METRIC）后，需要对其中用到的一些超参数进行设置，scepter使用yaml文件定义各模块超参数，具体参考scepter/modules/examples/sd15/sd15_512_full.yaml

# 6. 模型训练

通过指定--cfg参数加载所需的yaml文件，完成训练或批量测试的操作
```shell
多机多卡训练
# 基于spawn方式，为默认模式
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WORLD_SIZE=1
python  -W ignore  scepter/run_train.py --cfg scepter/examples/sd15/sd15_512_full.yaml
# 基于原生的pytorch引擎 torchrun模式
torchrun --nproc_per_node 4  scepter/run_train.py --cfg scepter/examples/sd15/sd15_512_full.yaml --launcher torchrun
# 基于pytorch_lightning引擎， ENV.USE_PL需要设置为true
python -W ignore scepter/run_train.py --cfg scepter/examples/sd15/sd15_512_full.yaml
# 单卡训练
# 基于spawn方式，为默认模式
export CUDA_VISIBLE_DEVICES=0
export WORLD_SIZE=1
python -W ignore  scepter/run_train.py --cfg scepter/examples/sd15/sd15_512_full.yaml
# 基于原生的pytorch引擎
python -W ignore scepter/run_train.py --cfg scepter/examples/sd15/sd15_512_full.yaml --launcher torchrun
# 基于pytorch_lightning引擎， ENV.USE_PL需要设置为true
python scepter/run_train.py --cfg scepter/examples/sd15/sd15_512_full.yaml
```

# 7. 模型推理

单次推理可以通过自定义run_inference.py文件来实现，参照sd1.5的推理方法
```shell
python -W ignore scepter/run_inference.py --prompt "a woman" --n_prompt "" --num_samples 4 --pretrained_model "path/to/your/pretrained/model"
```
