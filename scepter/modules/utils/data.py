# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

from collections import OrderedDict

import torch


def transfer_data_to_numpy(data_map: dict) -> dict:
    """ Transfer tensors in data_map to numpy type.
    Will recursively walk through inner list, tuple and dict values.

    Args:
        data_map (dict): a dictionary which contains tensors to be transferred

    Returns:
        A dict which has same structure with input `data_map`.
    """
    if not isinstance(data_map, dict):
        return data_map
    ret = OrderedDict()
    for key, value in data_map.items():
        if isinstance(value, torch.Tensor):
            ret[key] = value.detach().cpu().numpy()
        elif isinstance(value, dict):
            ret[key] = transfer_data_to_numpy(value)
        elif isinstance(value, (list, tuple)):
            ret[key] = type(value)([transfer_data_to_numpy(t) for t in value])
        else:
            ret[key] = value
    return ret


def transfer_data_to_cpu(data_map: dict) -> dict:
    """ Transfer tensors in data_map to cpu device.
    Will recursively walk through inner list, tuple and dict values.

    Args:
        data_map (dict): a dictionary which contains tensors to be transferred

    Returns:
        A dict which has same structure with input `data_map`.
    """
    if not isinstance(data_map, dict):
        return data_map
    ret = OrderedDict()
    for key, value in data_map.items():
        if isinstance(value, torch.Tensor):
            ret[key] = value.detach().cpu()
        elif isinstance(value, dict):
            ret[key] = transfer_data_to_cpu(value)
        elif isinstance(value, (list, tuple)):
            ret[key] = type(value)([transfer_data_to_cpu(t) for t in value])
        else:
            ret[key] = value
    torch.cuda.empty_cache()
    return ret


def transfer_data_to_cuda(data_map: dict) -> dict:
    """ Transfer tensors in data_map to current default gpu device.
    Will recursively walk through inner list, tuple and dict values.

    Args:
        data_map (dict): a dictionary which contains tensors to be transferred

    Returns:
        A dict which has same structure with input `data_map`.
    """
    import platform
    if platform.system() == 'Darwin':
        return data_map
    if not isinstance(data_map, dict):
        return data_map
    ret = OrderedDict()
    for key, value in data_map.items():
        if isinstance(value, torch.Tensor):
            if value.is_cuda:
                ret[key] = value
            else:
                ret[key] = value.cuda(non_blocking=True)
        elif isinstance(value, dict):
            ret[key] = transfer_data_to_cuda(value)
        elif isinstance(value, (list, tuple)):
            ret[key] = type(value)([transfer_data_to_cuda(t) for t in value])
        else:
            ret[key] = value
    return ret
