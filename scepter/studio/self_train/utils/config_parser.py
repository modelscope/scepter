# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from glob import glob

import yaml

paras_keys = [
    'TRAIN_BATCH_SIZE', 'TRAIN_PREFIX', 'TRAIN_N_PROMPT', 'RESOLUTION',
    'MEMORY', 'EPOCHS', 'SAVE_INTERVAL', 'EPSEC', 'LEARNING_RATE',
    'IS_DEFAULT', 'TUNER'
]
control_paras_keys = ['CONTROL_MODE', 'RESOLUTION', 'IS_DEFAULT']


def build_meta_index(meta_cfg, config_file):
    tuner_type = {}
    paras = meta_cfg.get('PARAS', None)
    if paras:
        for idx, para in enumerate(paras):
            for key in paras_keys:
                if key not in para:
                    print(
                        f'Para key {key} not defined in {config_file} META/RARAS[{idx}]'
                    )
                assert key in para
            tuner_type[para['TUNER']] = para
            if para['IS_DEFAULT']:
                tuner_type['default'] = para['TUNER']

    tuner_type['choices'] = list(tuner_type.keys())
    if 'default' in tuner_type['choices']:
        tuner_type['choices'].remove('default')
    if 'default' not in tuner_type:
        tuner_type['default'] = tuner_type['choices'][0] if len(
            tuner_type['choices']) > 0 else ''

    tuner_paras = meta_cfg.get('TUNERS', None)
    return tuner_type, tuner_paras


def build_meta_index_control(meta_cfg, config_file):
    control_type = {}
    paras = meta_cfg.get('CONTROL_PARAS', None)
    if paras:
        for idx, para in enumerate(paras):
            for key in control_paras_keys:
                if key not in para:
                    print(
                        f'Para key {key} not defined in {config_file} META/RARAS[{idx}]'
                    )
                assert key in para
            control_type[para['CONTROL_MODE']] = para
            # control_type[para["CONTROL_MODE"]] = para
            if para['IS_DEFAULT']:
                control_type['default'] = para['CONTROL_MODE']

    control_type['choices'] = list(control_type.keys())
    if 'default' in control_type['choices']:
        control_type['choices'].remove('default')
    if 'default' not in control_type:
        control_type['default'] = control_type['choices'][0] if len(
            control_type['choices']) > 0 else ''

    return control_type, paras


def get_all_config(config_root, global_meta):
    config_dict = {}
    config_list = glob(os.path.join(config_root, '*/*_pro.yaml'),
                       recursive=True)
    for config_file in config_list:
        base_model_name = config_file.split('/')[-2]
        with open(config_file, 'r') as f:
            cfg = yaml.load(f.read(), Loader=yaml.SafeLoader)
        if base_model_name not in config_dict:
            config_dict[base_model_name] = {}

        meta_cfg = cfg.pop('META')
        inference_paras = meta_cfg.pop('INFERENCE_PARAS')
        if 'MODIFY_PARAS' in meta_cfg:
            modify_para = meta_cfg['MODIFY_PARAS']
        else:
            modify_para = {}
        version = meta_cfg['VERSION']
        if version in config_dict[base_model_name]:
            ori_config = config_dict[base_model_name][version]['config_file']
            print(
                f'Current config {config_file} for {base_model_name}_{version} will be replaced by {ori_config}.'
            )

        tuner_type, tuner_para = build_meta_index(meta_cfg, config_file)
        config_dict[base_model_name][version] = {
            'config_file': config_file,
            'config_value': cfg,
            'inference_para': inference_paras,
            'tuner_type': tuner_type,
            'tuner_para': tuner_para,
            'modify_para': modify_para,
            'is_share': 'IS_SHARE' in meta_cfg and meta_cfg['IS_SHARE']
        }
        if 'CONTROL_PARAS' in meta_cfg:
            control_type, control_para = build_meta_index_control(
                meta_cfg, config_file)
            config_dict[base_model_name][version].update({
                'control_type':
                control_type,
                'control_para':
                control_para
            })
        if meta_cfg['IS_DEFAULT']:
            config_dict[base_model_name]['default'] = version

    for base_model_name in config_dict:
        config_dict[base_model_name]['choices'] = list(
            config_dict[base_model_name].keys())
        if 'default' in config_dict[base_model_name]['choices']:
            config_dict[base_model_name]['choices'].remove('default')
        if 'default' not in config_dict[base_model_name]:
            config_dict[base_model_name][
                'default'] = config_dict[base_model_name]['choices'][0] if len(
                    config_dict[base_model_name]['choices']) > 0 else ''

    config_dict['choices'] = list(config_dict.keys())
    config_dict['default'] = config_dict['choices'][0] if len(
        config_dict['choices']) > 0 else ''
    default_base_model = global_meta.DEFAULT_FOLDER
    if default_base_model in config_dict:
        config_dict['default'] = default_base_model
    config_dict['samplers'] = {
        sampler['NAME']: sampler
        for sampler in global_meta.SAMPLERS
    }
    config_dict.update(cfg)
    return config_dict


def get_default(config_dict):
    ret_data = {}
    # 默认的模型
    ret_data['model_choices'] = config_dict['choices']
    ret_data['model_default'] = config_dict['default']
    default_version_cfg = config_dict.get(config_dict['default'], None)
    # 默认的版本
    if default_version_cfg is None:
        return ret_data
    ret_data['version_choices'] = default_version_cfg['choices']
    ret_data['version_default'] = default_version_cfg['default']
    default_model_cfg = default_version_cfg.get(default_version_cfg['default'],
                                                None)
    if default_model_cfg is None:
        return ret_data
    if 'tuner_type' in default_model_cfg and default_model_cfg['tuner_type'][
            'default'] != '':
        default_tuner_cfg = default_model_cfg['tuner_type']
    else:
        return ret_data
    ret_data['tuner_choices'] = default_tuner_cfg['choices']
    defalt_t_type = default_tuner_cfg['default']
    ret_data['tuner_default'] = defalt_t_type
    type_paras = default_tuner_cfg.get(defalt_t_type, None)
    if type_paras is not None:
        ret_data.update(type_paras)
    return ret_data


def get_values_by_model(config_dict, model_name):
    ret_data = {}
    version_cfg = config_dict.get(model_name, None)
    if version_cfg is None:
        return ret_data
    ret_data['version_choices'] = version_cfg['choices']
    ret_data['version_default'] = version_cfg['default']
    default_model_cfg = version_cfg.get(version_cfg['default'], None)
    if default_model_cfg is None:
        return ret_data

    default_tuner_cfg = default_model_cfg['tuner_type']
    ret_data['tuner_choices'] = default_tuner_cfg['choices']
    defalt_t_type = default_tuner_cfg['default']
    ret_data['tuner_default'] = defalt_t_type
    type_paras = default_tuner_cfg.get(defalt_t_type, None)
    if type_paras is not None:
        ret_data.update(type_paras)
    return ret_data


def get_values_by_model_version(config_dict, model_name, version):
    ret_data = {}
    version_cfg = config_dict.get(model_name, None)
    if version_cfg is None:
        return ret_data
    tuner_cfg = version_cfg.get(version, None)
    if tuner_cfg is None:
        return ret_data

    default_tuner_cfg = tuner_cfg['tuner_type']
    ret_data['tuner_choices'] = default_tuner_cfg['choices']
    defalt_t_type = default_tuner_cfg['default']
    ret_data['tuner_default'] = defalt_t_type
    type_paras = default_tuner_cfg.get(defalt_t_type, None)
    if type_paras is not None:
        ret_data.update(type_paras)
    return ret_data


def get_values_by_model_version_tuner(config_dict, model_name, version,
                                      tuner_name):
    ret_data = {}
    version_cfg = config_dict.get(model_name, None)
    if version_cfg is None:
        return ret_data
    model_cfg = version_cfg.get(version, None)
    if model_cfg is None:
        return ret_data
    tuner_cfg = model_cfg['tuner_type']
    type_paras = tuner_cfg.get(tuner_name, None)
    if type_paras is not None:
        ret_data.update(type_paras)
    return ret_data


def get_inference_para_by_model_version(config_dict, model_name, version):
    ret_data = {}
    version_cfg = config_dict.get(model_name, None)
    if version_cfg is None:
        return ret_data
    tuner_cfg = version_cfg.get(version, None)
    if tuner_cfg is None:
        return ret_data
    samplers_list = list(config_dict['samplers'].keys())
    ret_data.update(tuner_cfg['inference_para'])
    ret_data['sampler_default'] = ret_data['DEFAULT_SAMPLER']
    if ret_data['DEFAULT_SAMPLER'] in samplers_list:
        samplers_list.remove(ret_data['DEFAULT_SAMPLER'])
    ret_data['sampler_choices'] = [ret_data['DEFAULT_SAMPLER']] + samplers_list
    return ret_data


def get_base_model_list(config_dict):
    ret_data = {'model_choices': [], 'model_default': '@'}
    default_model = config_dict['default']
    if not default_model == '':
        default_version = config_dict[default_model]['default']
    else:
        default_version = ''
    ret_data['model_default'] = f'{default_model}@{default_version}'
    for base_model_name in config_dict:
        if base_model_name in ['default', 'choices', 'samplers']:
            continue
        for version in config_dict[base_model_name]:
            if version in ['default', 'choices', 'samplers']:
                continue
            ret_data['model_choices'].append(f'{base_model_name}@{version}')
    if not ret_data['model_default'] == '@':
        ret_data['model_choices'].remove(ret_data['model_default'])
        ret_data['model_choices'] = [ret_data['model_default']
                                     ] + ret_data['model_choices']
    else:
        ret_data['model_default'] = ret_data['model_choices'][0] if len(
            ret_data['model_choices']) > 0 else ''

    if not ret_data['model_default'] == '':
        ret_data['model_name'] = ret_data['model_default'].split('@')[0]
        ret_data['version_name'] = ret_data['model_default'].split('@')[1]
    else:
        ret_data['model_name'] = ''
        ret_data['version_name'] = ''
    return ret_data


def get_control_para_by_model_version(config_dict, model_name, version):
    ret_data = {}
    version_cfg = config_dict.get(model_name, None)
    if version_cfg is None:
        return ret_data
    tuner_cfg = version_cfg.get(version, None)
    if tuner_cfg is None:
        return ret_data
    samplers_list = list(config_dict['samplers'].keys())
    ret_data.update(tuner_cfg['inference_para'])
    ret_data['sampler_default'] = ret_data['DEFAULT_SAMPLER']
    if ret_data['DEFAULT_SAMPLER'] in samplers_list:
        samplers_list.remove(ret_data['DEFAULT_SAMPLER'])
    ret_data['sampler_choices'] = [ret_data['DEFAULT_SAMPLER']] + samplers_list
    return ret_data


def get_control_default(config_dict):
    ret_data = {}
    # 默认的模型
    ret_data['model_choices'] = config_dict['choices']
    ret_data['model_default'] = config_dict['default']
    default_version_cfg = config_dict.get(config_dict['default'], None)
    # 默认的版本
    if default_version_cfg is None:
        return ret_data
    ret_data['version_choices'] = default_version_cfg['choices']
    ret_data['version_default'] = default_version_cfg['default']
    default_control_cfg = default_version_cfg.get(
        default_version_cfg['default'], None)
    if default_control_cfg is None:
        return ret_data
    if 'control_type' in default_control_cfg and default_control_cfg[
            'control_type']['default'] != '':
        default_control_cfg = default_control_cfg['control_type']
    else:
        return ret_data

    ret_data['control_choices'] = list(default_control_cfg['choices'].keys())
    defalt_t_type = default_control_cfg['default']
    ret_data['control_default'] = defalt_t_type
    type_paras = default_control_cfg.get(defalt_t_type, None)
    if type_paras is not None:
        ret_data.update(type_paras)
    return ret_data
