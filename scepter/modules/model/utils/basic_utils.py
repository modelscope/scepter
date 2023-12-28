# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from inspect import isfunction

import torch


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        ctx.gpu_autocast_kwargs = {
            'enabled': torch.is_autocast_enabled(),
            'dtype': torch.get_autocast_gpu_dtype(),
            'cache_enabled': torch.is_autocast_cache_enabled()
        }
        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [
            x.detach().requires_grad_(True) for x in ctx.input_tensors
        ]
        with torch.enable_grad(), \
                torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs):
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def transfer_size(para_num):
    if para_num > 1000 * 1000 * 1000 * 1000:
        bill = para_num / (1000 * 1000 * 1000 * 1000)
        return '{:.2f}T'.format(bill)
    elif para_num > 1000 * 1000 * 1000:
        gyte = para_num / (1000 * 1000 * 1000)
        return '{:.2f}B'.format(gyte)
    elif para_num > (1000 * 1000):
        meta = para_num / (1000 * 1000)
        return '{:.2f}M'.format(meta)
    elif para_num > 1000:
        kelo = para_num / 1000
        return '{:.2f}K'.format(kelo)
    else:
        return para_num


def count_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    return transfer_size(total_params)


def expand_dims_like(x, y):
    while x.dim() != y.dim():
        x = x.unsqueeze(-1)
    return x
