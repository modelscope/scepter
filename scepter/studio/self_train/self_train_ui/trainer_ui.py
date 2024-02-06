# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import copy
import datetime
import os
import random
import time

import gradio as gr
import torch
import yaml

import scepter
from scepter.modules.utils.file_system import FS
from scepter.studio.self_train.self_train_ui.component_names import \
    TrainerUIName
from scepter.studio.self_train.utils.config_parser import (
    get_default, get_values_by_model, get_values_by_model_version,
    get_values_by_model_version_tuner,
    get_values_by_model_version_tuner_resolution)
from scepter.studio.utils.uibase import UIBase


def print_memory_status(is_debug):
    if not is_debug:
        nvi_info = os.popen('nvidia-smi').read()
        gpu_mem = nvi_info.split('\n')[9].split('|')[2].split('/')[0].strip()
    else:
        gpu_mem = 0
    return gpu_mem


refresh_symbol = '\U0001f504'  # 🔄


def get_work_name(model, version, tuner, resolution):
    model_prefix = f'Swift@{model}@{version}@{tuner}@{resolution}'
    return model_prefix + '@' + '{0:%Y%m%d%H%M%S%f}'.format(
        datetime.datetime.now()) + ''.join(
            [str(random.randint(1, 10)) for i in range(3)])


class TrainerUI(UIBase):
    def __init__(self, cfg, all_cfg_value, is_debug=False, language='en'):
        self.BASE_CFG_VALUE = all_cfg_value
        self.para_data = get_default(self.BASE_CFG_VALUE)
        self.run_script = os.path.join(os.path.dirname(scepter.dirname),
                                       cfg.SCRIPT_DIR, 'run_task.py')
        self.work_dir_pre, _ = FS.map_to_local(cfg.WORK_DIR)
        self.is_debug = is_debug
        self.component_names = TrainerUIName(language=language)

    def create_ui(self):
        with gr.Box():
            with gr.Row(variant='panel', equal_height=True):
                with gr.Column(variant='panel'):
                    gr.Markdown(self.component_names.user_direction)
                    self.data_type = gr.Dropdown(
                        choices=self.component_names.data_type_choices,
                        value=self.component_names.data_type_value,
                        label=self.component_names.data_type_name,
                        interactive=True)
                    self.ms_data_name = gr.Textbox(
                        label=' or '.join(
                            self.component_names.data_type_choices),
                        max_lines=1,
                        placeholder=self.component_names.
                        ms_data_name_place_hold,
                        interactive=True)
                    with gr.Box(visible=False) as self.ms_data_box:
                        with gr.Row():
                            self.ms_data_space = gr.Textbox(
                                label=self.component_names.ms_data_space,
                                max_lines=1)
                            self.ms_data_subname = gr.Textbox(
                                label=self.component_names.ms_data_subname,
                                value='default',
                                max_lines=1)
                with gr.Column(variant='panel'):
                    with gr.Box():
                        gr.Markdown(self.component_names.training_block)
                        with gr.Row():
                            with gr.Column(scale=1, min_width=0):
                                self.base_model = gr.Dropdown(
                                    choices=self.para_data.get(
                                        'model_choices', []),
                                    value=self.para_data.get(
                                        'model_default', ''),
                                    label=self.component_names.base_model,
                                    interactive=True)
                            with gr.Column(scale=1, min_width=0):
                                self.tuner_name = gr.Dropdown(
                                    choices=self.para_data.get(
                                        'tuner_choices', []),
                                    value=self.para_data.get(
                                        'tuner_default', ''),
                                    label=self.component_names.tuner_name,
                                    interactive=True)
                        with gr.Row():
                            with gr.Column(scale=1, min_width=0):
                                self.base_model_revision = gr.Dropdown(
                                    choices=self.para_data.get(
                                        'version_choices', []),
                                    value=self.para_data.get(
                                        'version_default', ''),
                                    label=self.component_names.
                                    base_model_revision,
                                    interactive=True)

                            with gr.Column(scale=1, min_width=0):
                                self.resolution = gr.Dropdown(
                                    choices=self.para_data.get(
                                        'resolution_choices', []),
                                    value=self.para_data.get(
                                        'resolution_default', 1024),
                                    label=self.component_names.resolution,
                                    allow_custom_value=True,
                                    interactive=True)

                        with gr.Row():
                            with gr.Column(scale=1, min_width=0):
                                self.train_epoch = gr.Number(
                                    label=self.component_names.train_epoch,
                                    value=self.para_data.get('EPOCHS', 10),
                                    precision=0,
                                    interactive=True)
                            with gr.Column(scale=1, min_width=0):
                                self.learning_rate = gr.Number(
                                    label=self.component_names.learning_rate,
                                    value=self.para_data.get(
                                        'LEARNING_RATE', 0.0001),
                                    interactive=True)

                        with gr.Row():
                            with gr.Column(scale=1, min_width=0):
                                self.save_interval = gr.Number(
                                    label=self.component_names.save_interval,
                                    value=self.para_data.get(
                                        'SAVE_INTERVAL', 10),
                                    precision=0,
                                    interactive=True)
                            with gr.Column(scale=1, min_width=0):
                                self.train_batch_size = gr.Number(
                                    label=self.component_names.
                                    train_batch_size,
                                    value=self.para_data.get(
                                        'TRAIN_BATCH_SIZE', 4),
                                    precision=0,
                                    interactive=True)

                        with gr.Row():
                            with gr.Column(scale=1, min_width=0):
                                self.prompt_prefix = gr.Text(
                                    label=self.component_names.prompt_prefix,
                                    value=self.para_data.get(
                                        'TRAIN_PREFIX', ''))
                            with gr.Column(scale=1, min_width=0):
                                self.replace_keywords = gr.Text(
                                    label=self.component_names.
                                    replace_keywords,
                                    value='')

                        with gr.Row():
                            with gr.Column(scale=5, min_width=0):
                                self.work_name = gr.Text(
                                    label=self.component_names.work_name,
                                    value=None,
                                    interactive=False)
                            with gr.Column(scale=1, min_width=0):
                                self.work_name_button = gr.Button(
                                    value=refresh_symbol)
                            with gr.Column(scale=2, min_width=0):
                                self.push_to_hub = gr.Checkbox(
                                    label=self.component_names.push_to_hub,
                                    value=False,
                                    visible=False)

            with gr.Row(variant='panel', equal_height=True):
                self.examples = gr.Examples(
                    examples=[
                        [
                            self.component_names.data_type_choices[0],
                            '',
                            'https://modelscope.cn/api/v1/models/damo/scepter/repo?Revision=master&FilePath=datasets/3D_example_csv.zip',  # noqa
                            ''
                        ],
                        [
                            self.component_names.data_type_choices[1], 'damo',
                            'style_custom_dataset', '3D'
                        ]
                    ],
                    inputs=[
                        self.data_type, self.ms_data_space, self.ms_data_name,
                        self.ms_data_subname
                    ])

            with gr.Row(variant='panel', equal_height=True):
                with gr.Box():
                    gr.Markdown(self.component_names.log_block)
                    self.training_message = gr.Markdown()
            with gr.Row(variant='panel', equal_height=True):
                self.training_button = gr.Button()

    def set_callbacks(self, inference_ui):
        def change_data_type(data_type):
            if data_type == self.component_names.data_type_choices[0]:
                return gr.Box(visible=False)
            elif data_type == self.component_names.data_type_choices[1]:
                return gr.Box(visible=True)

        self.data_type.change(fn=change_data_type,
                              inputs=[self.data_type],
                              outputs=[self.ms_data_box],
                              queue=False)

        self.work_name_button.click(fn=get_work_name,
                                    inputs=[
                                        self.base_model,
                                        self.base_model_revision,
                                        self.tuner_name, self.resolution
                                    ],
                                    outputs=[self.work_name],
                                    queue=False)

        def change_train_value_by_model(base_model):
            '''
                Changes to the base model will affect the training parameters,
                and it is best to define the related default values in the YAML.
                Training Iterations:
                Learning Rate:
                Training Prefix Used:
                Training Batch Size:
                Supported Resolutions:
                Supported Fine-tuning Methods:
                Supported Fine-tuning Methods:
                Prefix for Saved Model:
            '''
            ret_data = get_values_by_model(self.BASE_CFG_VALUE, base_model)
            return ret_data.get('EPOCHS', 10), \
                ret_data.get('LEARNING_RATE', 0.0001), \
                ret_data.get('SAVE_INTERVAL', 10), \
                ret_data.get('TRAIN_BATCH_SIZE', 4), \
                ret_data.get('TRAIN_PREFIX', ''), \
                gr.Dropdown(value=ret_data.get('version_default', ''), choices=ret_data.get('version_choices', []),
                            interactive=True), \
                gr.Dropdown(value=ret_data.get('tuner_default', ''), choices=ret_data.get('tuner_choices', []),
                            interactive=True), \
                gr.Dropdown(value=ret_data.get('resolution_default', 1024),
                            choices=ret_data.get('resolution_choices', []), interactive=True)

        self.base_model.change(fn=change_train_value_by_model,
                               inputs=[self.base_model],
                               outputs=[
                                   self.train_epoch, self.learning_rate,
                                   self.save_interval, self.train_batch_size,
                                   self.prompt_prefix,
                                   self.base_model_revision, self.tuner_name,
                                   self.resolution
                               ],
                               queue=False)

        #
        def change_train_value_by_model_version(base_model,
                                                base_model_revision):
            '''
                Changes to the base model will affect the training parameters,
                and it is best to define the related default values in the YAML.
                Training Iterations:
                Learning Rate:
                Training Prefix Used:
                Training Batch Size:
                Supported Resolutions:
                Supported Fine-tuning Methods:
                Supported Fine-tuning Methods:
                Prefix for Saved Model:
            '''
            ret_data = get_values_by_model_version(self.BASE_CFG_VALUE,
                                                   base_model,
                                                   base_model_revision)
            return ret_data.get('EPOCHS', 10), \
                ret_data.get('LEARNING_RATE', 0.0001), \
                ret_data.get('SAVE_INTERVAL', 10), \
                ret_data.get('TRAIN_BATCH_SIZE', 4), \
                ret_data.get('TRAIN_PREFIX', ''), \
                gr.Dropdown(value=ret_data.get('tuner_default', ''), choices=ret_data.get('tuner_choices', []),
                            interactive=True), \
                gr.Dropdown(value=ret_data.get('resolution_default', 1024),
                            choices=ret_data.get('resolution_choices', []), interactive=True)

        #
        self.base_model_revision.change(
            fn=change_train_value_by_model_version,
            inputs=[self.base_model, self.base_model_revision],
            outputs=[
                self.train_epoch, self.learning_rate, self.save_interval,
                self.train_batch_size, self.prompt_prefix, self.tuner_name,
                self.resolution
            ],
            queue=False)

        #
        #
        def change_train_value_by_model_version_tuner(base_model,
                                                      base_model_revision,
                                                      tuner_name):
            '''
                Changes to the base model will affect the training parameters,
                and it is best to define the related default values in the YAML.
                Training Iterations:
                Learning Rate:
                Training Prefix Used:
                Training Batch Size:
                Supported Resolutions:
                Supported Fine-tuning Methods:
                Supported Fine-tuning Methods:
                Prefix for Saved Model:
            '''
            ret_data = get_values_by_model_version_tuner(
                self.BASE_CFG_VALUE, base_model, base_model_revision,
                tuner_name)
            return ret_data.get('EPOCHS', 10), \
                ret_data.get('LEARNING_RATE', 0.0001), \
                ret_data.get('SAVE_INTERVAL', 10), \
                ret_data.get('TRAIN_BATCH_SIZE', 4), \
                ret_data.get('TRAIN_PREFIX', ''), \
                gr.Dropdown(value=ret_data.get('resolution_default', 1024),
                            choices=ret_data.get('resolution_choices', []), interactive=True)

        #
        self.tuner_name.change(fn=change_train_value_by_model_version_tuner,
                               inputs=[
                                   self.base_model, self.base_model_revision,
                                   self.tuner_name
                               ],
                               outputs=[
                                   self.train_epoch, self.learning_rate,
                                   self.save_interval, self.train_batch_size,
                                   self.prompt_prefix, self.resolution
                               ],
                               queue=False)

        #
        def change_train_value_by_model_version_tuner_resolution(
                base_model, base_model_revision, tuner_name, resolution):
            '''
                Changes to the base model will affect the training parameters,
                and it is best to define the related default values in the YAML.
                Training Iterations:
                Learning Rate:
                Training Prefix Used:
                Training Batch Size:
                Supported Resolutions:
                Supported Fine-tuning Methods:
                Supported Fine-tuning Methods:
                Prefix for Saved Model:
            '''
            ret_data = get_values_by_model_version_tuner_resolution(
                self.BASE_CFG_VALUE, base_model, base_model_revision,
                tuner_name, resolution)
            print('change_train_value_by_model_version_tuner_resolution',
                  ret_data)
            # work_name = get_work_name(base_model, base_model_revision,
            #                           tuner_name, resolution)
            return ret_data.get('EPOCHS', 10), \
                ret_data.get('LEARNING_RATE', 0.0001), \
                ret_data.get('SAVE_INTERVAL', 10), \
                ret_data.get('TRAIN_BATCH_SIZE', 4), \
                ret_data.get('TRAIN_PREFIX', '')

        #
        self.resolution.change(
            fn=change_train_value_by_model_version_tuner_resolution,
            inputs=[
                self.base_model, self.base_model_revision, self.tuner_name,
                self.resolution
            ],
            outputs=[
                self.train_epoch, self.learning_rate, self.save_interval,
                self.train_batch_size, self.prompt_prefix
            ],
            queue=False)

        def run_train(work_name, data_type, ms_data_space, ms_data_name,
                      ms_data_subname, base_model, base_model_revision,
                      tuner_name, resolution, train_epoch, learning_rate,
                      save_interval, train_batch_size, prompt_prefix,
                      replace_keywords, push_to_hub):
            # Check Cuda
            if not torch.cuda.is_available() and not self.is_debug:
                raise gr.Error(self.component_names.training_err1)

            if work_name == 'custom' or work_name is None or work_name == '':
                raise gr.Error(self.component_names.training_err4)
            work_dir = os.path.join(self.work_dir_pre, work_name)
            if not os.path.exists(work_dir):
                os.makedirs(work_dir)
            else:
                raise gr.Error(self.component_names.training_err4)

            if push_to_hub:
                model_id = work_name
                model_id = model_id.replace('@', '-')
                hub_model_id = f'scepter/{model_id}'
            else:
                hub_model_id = ''

            # Check Cuda Memory
            if torch.cuda.is_available() and not self.is_debug:
                device = torch.device('cuda:0')
                required_memory_bytes = 40 * (1024**3)
                try:
                    tensor = torch.empty(  # noqa
                        (required_memory_bytes // 4, ), device=device
                    )  # create 18GB tensor to check the memory if enough
                    del tensor
                except RuntimeError:
                    raise gr.Error(self.component_names.training_err2)

            # Check Instance Valid
            if ms_data_name is None:
                raise gr.Error(self.component_names.training_err3)

            st_time = time.time()

            def prepare_data(data_cfg):
                data_cfg['BATCH_SIZE'] = int(train_batch_size)
                data_cfg['PROMPT_PREFIX'] = prompt_prefix
                data_cfg['REPLACE_KEYWORDS'] = replace_keywords
                if data_type in self.component_names.data_type_choices:
                    if ms_data_name.startswith(
                            'http') or ms_data_name.endswith('zip'):
                        work_data_dir = os.path.join(work_dir, 'data')
                        if not os.path.exists(work_data_dir):
                            os.makedirs(work_data_dir)
                        ms_data_http_zip_path = ms_data_name
                        ms_data_local_name = ms_data_name.split(
                            '/')[-1].replace('.zip', '')
                        ms_data_local_zip_path = os.path.join(
                            work_data_dir,
                            ms_data_name.split('/')[-1])
                        ms_data_local_file_path = os.path.join(
                            work_data_dir, ms_data_local_name)
                        if not os.path.exists(ms_data_local_file_path):
                            if ms_data_http_zip_path.startswith('http'):
                                os.system(
                                    f"wget '{ms_data_http_zip_path}' -O '{ms_data_local_zip_path}'"
                                )
                            elif os.path.exists(ms_data_http_zip_path):
                                ms_data_local_zip_path = ms_data_http_zip_path
                            else:
                                raise gr.Error(
                                    self.component_names.training_err6)
                            print(
                                f"unzip -o '{ms_data_local_zip_path}' -d '{work_data_dir}'"
                            )
                            os.system(
                                f"unzip -o '{ms_data_local_zip_path}' -d '{work_data_dir}'"
                            )
                        data_cfg['MS_DATASET_NAME'] = ms_data_local_file_path
                        data_cfg['MS_DATASET_NAMESPACE'] = ''
                        data_cfg['MS_DATASET_SUBNAME'] = ''
                        data_cfg['MS_REMAP_PATH'] = ms_data_local_file_path
                        data_cfg['MS_REMAP_KEYS'] = None
                    elif (os.path.exists(ms_data_name) and os.path.exists(
                            os.path.join(ms_data_name, 'train.csv'))
                          and os.path.exists(
                              os.path.join(ms_data_name, 'images'))):
                        data_cfg['MS_DATASET_NAME'] = ms_data_name
                        data_cfg['MS_DATASET_NAMESPACE'] = ''
                        data_cfg['MS_DATASET_SUBNAME'] = ''
                        data_cfg['MS_REMAP_PATH'] = ms_data_name
                        data_cfg['MS_REMAP_KEYS'] = None
                    else:
                        data_cfg['MS_DATASET_NAME'] = ms_data_name
                        data_cfg['MS_DATASET_NAMESPACE'] = ms_data_space
                        data_cfg['MS_DATASET_SUBNAME'] = ms_data_subname
                        if ms_data_name == 'style_custom_dataset':
                            data_cfg['MS_REMAP_KEYS'] = {
                                'Image:FILE': 'Target:FILE'
                            }
                        elif ms_data_name == 'lora-stable-diffusion-finetune':
                            data_cfg['MS_REMAP_KEYS'] = {'Text': 'Prompt'}
                        else:
                            data_cfg['MS_REMAP_KEYS'] = None
                    data_cfg['OUTPUT_SIZE'] = int(resolution)
                return data_cfg

            def prepare_train_config():
                cfg_file = os.path.join(work_dir, 'train.yaml')
                current_model_info = self.BASE_CFG_VALUE[base_model][
                    base_model_revision]
                modify_para = current_model_info['modify_para']
                cfg = current_model_info['config_value']
                if isinstance(modify_para, dict) and tuner_name in modify_para:
                    modify_c = modify_para[tuner_name]
                    if isinstance(modify_c, dict) and 'TRAIN' in modify_c:
                        train_modify_c = modify_c['TRAIN']
                        if isinstance(train_modify_c, dict):
                            for key, val in train_modify_c.items():
                                cache_value = [cfg]
                                c_k_list = key.split('.')
                                for idx, c_k in enumerate(c_k_list):
                                    if c_k.strip() == '':
                                        continue
                                    cache_value.append(
                                        copy.deepcopy(cache_value[idx][c_k]))
                                current_val = copy.deepcopy(val)
                                for c_k, v in zip(c_k_list[::-1],
                                                  cache_value[:-1][::-1]):
                                    v[c_k] = current_val
                                    current_val = v
                                cfg = current_val

                # update config
                cfg['SOLVER']['WORK_DIR'] = work_dir
                cfg['SOLVER']['OPTIMIZER']['LEARNING_RATE'] = float(
                    learning_rate * 640 / int(train_batch_size))
                cfg['SOLVER']['MAX_EPOCHS'] = int(train_epoch)
                cfg['SOLVER']['TRAIN_DATA']['BATCH_SIZE'] = int(
                    train_batch_size)
                cfg['SOLVER']['TUNER'] = current_model_info[
                    'tuner_para'][tuner_name] if isinstance(
                        current_model_info['tuner_para'],
                        dict) and tuner_name in current_model_info[
                            'tuner_para'] else None
                cfg['SOLVER']['TRAIN_DATA'] = prepare_data(
                    cfg['SOLVER']['TRAIN_DATA'])
                for hook in cfg['SOLVER']['TRAIN_HOOKS']:
                    if hook['NAME'] == 'CheckpointHook':
                        hook['INTERVAL'] = save_interval
                        hook['PUSH_TO_HUB'] = push_to_hub
                        hook['HUB_MODEL_ID'] = hub_model_id

                with open(cfg_file, 'w') as f_out:
                    yaml.dump(cfg,
                              f_out,
                              encoding='utf-8',
                              allow_unicode=True,
                              default_flow_style=False)
                return cfg_file

            cfg = prepare_train_config()

            def train_fn(cfg_file):
                torch.cuda.empty_cache()
                cmd = f'PYTHONPATH=. python {self.run_script} ' \
                      f'--cfg={cfg_file}  2> {self.work_dir_pre}/std_out.txt'
                print(cmd)
                if not self.is_debug:
                    res = os.system(cmd)
                else:
                    res = 0
                if res != 0:
                    error_info = '\n'.join(
                        open(f'{self.work_dir_pre}/std_out.txt',
                             'r').read().split('\n')[-20:])
                    raise gr.Error(
                        f'{self.component_names.training_err5} ({error_info}) '
                    )

            train_fn(cfg)

            if work_name not in inference_ui.model_list:
                inference_ui.model_list.append(work_name)
            message = f'''
                        Training completed! \n
                        Save in [ {work_name} ] \n
                        Take time [ {time.time() - st_time:.4f}s ] \n
                        Mem: [ {print_memory_status(self.is_debug)} ]
                        '''
            print(message)
            return message, gr.Dropdown.update(choices=inference_ui.model_list,
                                               value=work_name)

        self.training_button.click(
            run_train,
            inputs=[
                self.work_name, self.data_type, self.ms_data_space,
                self.ms_data_name, self.ms_data_subname, self.base_model,
                self.base_model_revision, self.tuner_name, self.resolution,
                self.train_epoch, self.learning_rate, self.save_interval,
                self.train_batch_size, self.prompt_prefix,
                self.replace_keywords, self.push_to_hub
            ],
            outputs=[self.training_message, inference_ui.output_model_name],
            queue=True)
