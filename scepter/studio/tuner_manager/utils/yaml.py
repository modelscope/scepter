# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import yaml

from scepter.modules.utils.file_system import IoString


def check_data(data):
    if isinstance(data, dict):
        for k, v in data.items():
            data[k] = check_data(v)
        return data
    elif isinstance(data, list):
        for idx, v in enumerate(data):
            data[idx] = check_data(v)
        return data
    else:
        if isinstance(data, IoString):
            return str(data)
        return data


def save_yaml(data, file_path):
    with open(file_path, 'w') as f_out:
        yaml.dump(check_data(data),
                  f_out,
                  encoding='utf-8',
                  allow_unicode=True,
                  default_flow_style=False)
