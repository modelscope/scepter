# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import os


def get_tuner_choices():
    tuner_choices = []
    current_dir = os.path.dirname(__file__)
    base_path = os.sep.join(current_dir.split(os.sep)[:-3])
    base_path = os.path.join(base_path, 'config')
    for file in os.listdir(base_path):
        if 'ipynb' in file:
            continue
        import json

        with open(os.path.join(base_path, file), 'r') as f:
            content = json.load(f)
            tuner_choices.append(content['name'])
    return tuner_choices
