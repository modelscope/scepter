# Quick Start

This chapter takes stable diffusion v1.5 as an example, demonstrating how to build a neural network architecture from scratch, as well as how to train and test a model based on that structure.

# 1. Defining Model Structure with the Network Class

The network class encompasses methods for defining, training, and testing the model. First, we initialize a network class and then define the required submodules: autoencoder, UNet, embedder, and loss.



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

        # Other module definition.
```

# 2. Implementing the Training and Testing Code for the Network class

Each network class relies on forward_train and forward_test functions to define its training and testing processes. In SD1.5, forward_train predicts the noise for each timestep t and conducts loss calculation.

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

The forward_test function is used during the inference stage to perform the full image denoising process.

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

# 3. Submodule Registration

After the network class is fully implemented, it is necessary to ensure that all submodules used within the class are registered. For instance, with the embedder in SD1.5, to instantiate this embedder within the network's initialization method, we must first implement the embedder class and register it with Scepter.

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

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    # @torch.no_grad()
    def _encode_text(self,
                     tokens,
                     tokenizer=None,
                     append_sentence_embedding=False):
        outputs = self.transformer(input_ids=tokens,
                                   output_hidden_states=self.layer == 'hidden')
        if self.layer == 'last':
            z = outputs.last_hidden_state
        elif self.layer == 'pooled':
            z = outputs.pooler_output[:, None, :]
            if self.use_final_layer_norm:
                z = self.transformer.text_model.final_layer_norm(z)
        else:
            z = outputs.hidden_states[self.layer_idx]
            if self.use_final_layer_norm:
                z = self.transformer.text_model.final_layer_norm(z)
        return z

    def encode_text(self,
                    tokens,
                    tokenizer=None,
                    append_sentence_embedding=False):
        if not self.use_grad:
            with torch.no_grad():
                output = self._encode_text(tokens, tokenizer,
                                           append_sentence_embedding)
        else:
            output = self._encode_text(tokens, tokenizer,
                                       append_sentence_embedding)
        return output

    @staticmethod
    def get_config_template():
        return dict_to_yaml('MODELS',
                            __class__.__name__,
                            FrozenCLIPEmbedder.para_dict,
                            set_name=True)
```

# 4. Solver Registration

The solver class encapsulates the complete process needed to train and test a network. For instance, to register a solver for training the SD1.5 model, you'd need to:

1. Create one or more data loaders, corresponding to the construct_data method in the solver;
2. Instantiate a model, corresponding to the construct_model method in the solver;
3. (Optional) Define metrics, corresponding to the construct_metrics method in the solver;
4. (Optional) Define various hooks (HOOKS) used in training and testing, such as model saving, pre-trained parameter loading, logging, etc.

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
        pass

    def save_checkpoint(self):
        pass

    def solve(self):
        self.before_solve()
        if 'train' in self._mode_set:
            self.run_train()
        if 'test' in self._mode_set:
            self.run_test()
        self.after_solve()

    def run_train(self):
        pass

    def run_eval(self):
        pass

    def run_test(self):
        pass
```

# 5. Training/Testing Hyperparameter Definitions

After registering all required components, which include but are not limited to BACKBONE, NETWORK, EMBEDDER, SOLVER, and METRIC, it is necessary to configure some of the hyperparameters used. Scepter employs YAML files to define hyperparameters for each module; for specifics, refer to scepter/modules/examples/sd15/sd15_512_full.yaml.

# 6. Model Training

Execute training or batch testing by loading the required YAML file with the --cfg parameter.
```shell
# Multi-node Multi-GPU Training
# Spawn mode (default)
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WORLD_SIZE=1
python  -W ignore  scepter/run_train.py --cfg scepter/examples/sd15/sd15_512_full.yaml
# torchrun mode
torchrun --nproc_per_node 4  scepter/run_train.py --cfg scepter/examples/sd15/sd15_512_full.yaml --launcher torchrun
# pytorch_lightning mode， ENV.USE_PL=true
python -W ignore scepter/run_train.py --cfg scepter/examples/sd15/sd15_512_full.yaml
# Single-GPU Training
# Spawn mode (default)
export CUDA_VISIBLE_DEVICES=0
export WORLD_SIZE=1
python -W ignore  scepter/run_train.py --cfg scepter/examples/sd15/sd15_512_full.yaml
# torchrun mode
python -W ignore scepter/run_train.py --cfg scepter/examples/sd15/sd15_512_full.yaml --launcher torchrun
# pytorch_lightning mode， ENV.USE_PL=true
python scepter/run_train.py --cfg scepter/examples/sd15/sd15_512_full.yaml
```

# 7. Model Inference

Single inference can be realized by customizing the run_inference.py file, with reference to the inference method of SD1.5.
```shell
python -W ignore scepter/run_inference.py --prompt "a woman" --n_prompt "" --num_samples 4 --pretrained_model "path/to/your/pretrained/model"
```
