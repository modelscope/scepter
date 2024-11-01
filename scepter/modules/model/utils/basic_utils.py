# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from inspect import isfunction

import torch
from torch.nn.utils.rnn import pad_sequence

from scepter.modules.utils.distribute import we


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


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


def unpack_tensor_into_imagelist(image_tensor, shapes):
    image_list = []
    for img, shape in zip(image_tensor, shapes):
        h, w = shape[0], shape[1]
        image_list.append(img[:, :h * w].view(1, -1, h, w))

    return image_list


def find_example(tensor_list, image_list):
    for i in tensor_list:
        if isinstance(i, torch.Tensor):
            return torch.zeros_like(i)
    for i in image_list:
        if isinstance(i, torch.Tensor):
            _, c, h, w = i.size()
            return torch.zeros_like(i.view(c, h * w).transpose(1, 0))
    return None


def pack_imagelist_into_tensor_v2(image_list):
    # allow None
    example = None
    image_tensor, shapes = [], []
    for img in image_list:
        if img is None:
            example = find_example(image_tensor,
                                   image_list) if example is None else example
            image_tensor.append(example)
            shapes.append(None)
            continue
        _, c, h, w = img.size()
        image_tensor.append(img.view(c, h * w).transpose(1, 0))  # h*w, c
        shapes.append((h, w))

    image_tensor = pad_sequence(image_tensor,
                                batch_first=True).permute(0, 2, 1)  # b, c, l
    return image_tensor, shapes


def to_device(inputs, strict=True):
    if inputs is None:
        return None
    if strict:
        assert all(isinstance(i, torch.Tensor) for i in inputs)
    return [i.to(we.device_id) if i is not None else None for i in inputs]


def check_list_of_list(ll):
    return isinstance(ll, list) and all(isinstance(i, list) for i in ll)
