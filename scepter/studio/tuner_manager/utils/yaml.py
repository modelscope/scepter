# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import yaml


def save_yaml(data, file_path):
    with open(file_path, 'w') as f_out:
        yaml.dump(data,
                  f_out,
                  encoding='utf-8',
                  allow_unicode=True,
                  default_flow_style=False)
