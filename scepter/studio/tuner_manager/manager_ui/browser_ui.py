# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil
from collections import OrderedDict

import gradio as gr
import scepter
from huggingface_hub import HfApi, snapshot_download
from scepter.modules.utils.config import Config
from scepter.modules.utils.file_system import FS
from scepter.modules.utils.module_transform import \
    convert_tuner_civitai_to_scepter
from scepter.studio.tuner_manager.manager_ui.component_names import \
    TunerManagerNames
from scepter.studio.tuner_manager.utils.dict import (delete_2level_dict,
                                                     update_2level_dict)
from scepter.studio.tuner_manager.utils.path import (
    is_valid_filename, is_valid_huggingface_filename,
    is_valid_modelscope_filename)
from scepter.studio.tuner_manager.utils.yaml import save_yaml
from scepter.studio.utils.uibase import UIBase


def wget_file(file_url, save_file):
    if 'oss' in file_url:
        file_url = file_url.split('?')[0]
    local_path, _ = FS.map_to_local(save_file)
    res = os.popen(f"wget -c '{file_url}' -O '{local_path}'")
    res.readlines()
    FS.put_object_from_local_file(local_path, save_file)
    return save_file, res


class BrowserUI(UIBase):
    def __init__(self, cfg, language='en'):
        self.work_dir = cfg.WORK_DIR
        self.yaml = os.path.join(self.work_dir, cfg.TUNER_LIST_YAML)
        if not FS.exists(self.yaml):
            self.saved_tuners = []
            with FS.put_to(self.yaml) as local_path:
                save_yaml({'TUNERS': self.saved_tuners}, local_path)
        else:
            with FS.get_from(self.yaml) as local_path:
                self.saved_tuners = Config.get_plain_cfg(
                    Config(cfg_file=local_path).TUNERS)
        self.saved_tuners_to_category()
        self.component_names = TunerManagerNames(language)
        self.language = language
        self.export_folder = os.path.join(self.work_dir, cfg.EXPORT_DIR)
        self.readme_file = cfg.README_EN if self.language == 'en' else cfg.README_ZH
        self.readme_file = os.path.join(os.path.dirname(scepter.dirname),
                                        self.readme_file)
        self.base_model_tuner_methods = cfg.BASE_MODEL_VERSION
        self.base_model_tuner_methods_map = {}
        for base_model_item in self.base_model_tuner_methods:
            self.base_model_tuner_methods_map[
                base_model_item.BASE_MODEL] = base_model_item.TUNER_TYPE

    def saved_tuners_to_category(self):
        self.saved_tuners_category = OrderedDict()
        for tuner in self.saved_tuners:
            first_level = f"{tuner['BASE_MODEL']}-{tuner['TUNER_TYPE']}"
            second_level = f"{tuner['NAME']}"
            login_user_name = tuner.get('USER_NAME', 'admin')
            if login_user_name not in self.saved_tuners_category:
                self.saved_tuners_category[login_user_name] = OrderedDict()
            local_tuner_work_dir, _ = FS.map_to_local(tuner['MODEL_PATH'])
            if not os.path.exists(local_tuner_work_dir):
                FS.get_dir_to_local_dir(tuner['MODEL_PATH'])
            update_2level_dict(self.saved_tuners_category[login_user_name],
                               {first_level: {
                                   second_level: tuner
                               }})

    def category_to_saved_tuners(self):
        self.saved_tuners = []
        for login_user_name, user_model in self.saved_tuners_category.items():
            for k, v in user_model.items():
                for kk, vv in v.items():
                    self.saved_tuners.append(vv)

    def get_choices_and_values(self, login_user_name='admin'):
        diffusion_models_choice = list(
            self.saved_tuners_category.get(login_user_name,
                                           OrderedDict()).keys())
        diffusion_model = diffusion_models_choice[0] if len(
            diffusion_models_choice) > 0 else None
        tuner_models_choice = []
        tuner_model = None
        if diffusion_model:
            tuner_models_choice = list(
                self.saved_tuners_category.get(login_user_name,
                                               OrderedDict()).get(
                                                   diffusion_model, {}).keys())
            tuner_model = tuner_models_choice[0] if len(
                tuner_models_choice) > 0 else None
        return diffusion_models_choice, diffusion_model, tuner_models_choice, tuner_model

    def create_ui(self, *args, **kwargs):
        diffusion_models_choice, diffusion_model, tuner_models_choice, tuner_model = self.get_choices_and_values(
        )
        with gr.Column():
            with gr.Group():
                gr.Markdown(self.component_names.browser_block_name)
                with gr.Row(variant='panel', equal_height=True):
                    with gr.Column(scale=4, min_width=0):
                        self.diffusion_models = gr.Dropdown(
                            label=self.component_names.base_models,
                            choices=diffusion_models_choice,
                            value=None,
                            multiselect=False,
                            interactive=True)
                    with gr.Column(scale=4, min_width=0):
                        self.tuner_models = gr.Dropdown(
                            label=self.component_names.tuner_name,
                            choices=tuner_models_choice,
                            value=None,
                            multiselect=False,
                            interactive=True)
                    with gr.Column(scale=1, min_width=0):
                        self.save_button = gr.Button(
                            value=self.component_names.save_symbol,
                            elem_classes='type_row',
                            elem_id='save_button',
                            visible=True)
                        self.delete_button = gr.Button(
                            value=self.component_names.delete_symbol,
                            elem_classes='type_row',
                            elem_id='delete_button',
                            visible=False)
                        self.model_export = gr.Button(
                            value=self.component_names.model_export,
                            elem_classes='type_row',
                            elem_id='save_button',
                            visible=True)
                    with gr.Column(scale=1, min_width=0):
                        self.refresh_button = gr.Button(
                            value=self.component_names.refresh_symbol,
                            elem_classes='type_row',
                            elem_id='refresh_button',
                            visible=True)
                        self.model_import = gr.Button(
                            value=self.component_names.model_import,
                            elem_classes='type_row',
                            elem_id='save_button',
                            visible=True)

                with gr.Group(visible=False) as self.export_setting:
                    gr.Markdown(self.component_names.export_desc)
                    with gr.Column(variant='panel'):
                        with gr.Row(equal_height=True):
                            with gr.Column(scale=9, min_width=0):
                                self.export_to = gr.Dropdown(
                                    choices=['modelscope', 'huggingface'],
                                    value='modelscope',
                                    label=None,
                                    show_label=False)
                            with gr.Column(scale=1, min_width=0):
                                self.export_close = gr.Button(
                                    value=self.component_names.close,
                                    elem_classes='type_row',
                                    elem_id='save_button')

                        with gr.Row(
                                equal_height=True) as self.ms_export_setting:
                            with gr.Column(scale=3, min_width=0):
                                self.ms_sdk = gr.Text(
                                    label=self.component_names.ms_sdk,
                                    show_label=False,
                                    container=False,
                                    placeholder='ModelScope SDK Token',
                                    value='')
                            with gr.Column(scale=3, min_width=0):
                                self.ms_export_username = gr.Text(
                                    label=self.component_names.ms_username,
                                    show_label=False,
                                    container=False,
                                    placeholder='ModelScope UserName',
                                    value='')
                            with gr.Column(scale=3, min_width=0):
                                self.ms_model_private = gr.Checkbox(
                                    label=self.component_names.
                                    ms_model_private,
                                    value=False)
                            with gr.Column(scale=1, min_width=0):
                                self.ms_export_submit = gr.Button(
                                    value=self.component_names.submit,
                                    elem_classes='type_row',
                                    elem_id='save_button')

                        with gr.Row(equal_height=True,
                                    visible=False) as self.hf_export_setting:
                            with gr.Column(scale=3, min_width=0):
                                self.hf_sdk = gr.Text(
                                    label=self.component_names.hf_sdk,
                                    show_label=False,
                                    container=False,
                                    placeholder='HuggingFace SDK Token',
                                    value='')
                            with gr.Column(scale=3, min_width=0):
                                self.hf_export_username = gr.Text(
                                    label=self.component_names.hf_username,
                                    show_label=False,
                                    container=False,
                                    placeholder='HuggingFace UserName',
                                    value='')
                            with gr.Column(scale=3, min_width=0):
                                self.hf_model_private = gr.Checkbox(
                                    label=self.component_names.
                                    hf_model_private,
                                    value=False)
                            with gr.Column(scale=1, min_width=0):
                                self.hf_export_submit = gr.Button(
                                    value=self.component_names.submit,
                                    elem_classes='type_row',
                                    elem_id='save_button')

                with gr.Group(visible=False) as self.import_setting:
                    gr.Markdown(self.component_names.import_desc)
                    with gr.Column(variant='panel'):
                        with gr.Row(equal_height=True):
                            with gr.Column(scale=9, min_width=0):
                                self.import_from = gr.Dropdown(
                                    choices=[
                                        'modelscope', 'huggingface', 'local'
                                    ],
                                    value='modelscope',
                                    label=None,
                                    show_label=False)
                            with gr.Column(scale=1, min_width=0):
                                self.import_close = gr.Button(
                                    value=self.component_names.close,
                                    elem_classes='type_row',
                                    elem_id='save_button')

                        with gr.Row(
                                equal_height=True) as self.ms_import_setting:
                            with gr.Column(scale=4, min_width=0):
                                self.ms_modelid = gr.Text(
                                    label=self.component_names.ms_modelid,
                                    show_label=False,
                                    container=False,
                                    placeholder='ModelScope Model Path',
                                    value='')
                            with gr.Column(scale=4, min_width=0):
                                self.ms_import_username = gr.Text(
                                    label=self.component_names.ms_username,
                                    show_label=False,
                                    container=False,
                                    placeholder='ModelScope UserName',
                                    value='')
                            with gr.Column(scale=1, min_width=0):
                                self.ms_import_submit = gr.Button(
                                    value=self.component_names.submit,
                                    elem_classes='type_row',
                                    elem_id='save_button')

                        with gr.Row(equal_height=True,
                                    visible=False) as self.hf_import_setting:
                            with gr.Column(scale=3, min_width=0):
                                self.hf_modelid = gr.Text(
                                    label=self.component_names.hf_modelid,
                                    show_label=False,
                                    container=False,
                                    placeholder='HuggingFace Model Path',
                                    value='')
                            with gr.Column(scale=3, min_width=0):
                                self.hf_import_username = gr.Text(
                                    label=self.component_names.hf_username,
                                    show_label=False,
                                    container=False,
                                    placeholder='HuggingFace UserName',
                                    value='')
                            with gr.Column(scale=3, min_width=0):
                                self.hf_sdk2 = gr.Text(
                                    label=self.component_names.hf_sdk,
                                    show_label=False,
                                    container=False,
                                    placeholder='HuggingFace SDK Token',
                                    value='')
                            with gr.Column(scale=1, min_width=0):
                                self.hf_import_submit = gr.Button(
                                    value=self.component_names.submit,
                                    elem_classes='type_row',
                                    elem_id='save_button')

                        with gr.Row(visible=False, equal_height=True
                                    ) as self.local_import_setting:
                            with gr.Column(scale=1, min_width=0):
                                self.file_path = gr.File(
                                    label=self.component_names.zip_file,
                                    min_width=0,
                                    file_types=['.zip', '.safetensors'],
                                    elem_classes='upload_zone')
                            with gr.Column(scale=1, min_width=0):
                                self.file_url = gr.Text(
                                    label=self.component_names.file_url,
                                    min_width=0,
                                    show_label=True,
                                    elem_classes='upload_zone')
                            with gr.Column(scale=1, min_width=0):
                                self.upload_base_models = gr.Dropdown(
                                    label=self.component_names.ubase_model,
                                    choices=[
                                        base_model_version.BASE_MODEL
                                        for base_model_version in
                                        self.base_model_tuner_methods
                                    ],
                                    value=self.base_model_tuner_methods[0].
                                    BASE_MODEL,
                                    multiselect=False,
                                    interactive=True)
                            with gr.Column(scale=1, min_width=0):
                                self.upload_tuner_type = gr.Dropdown(
                                    label=self.component_names.utuner_type,
                                    choices=self.base_model_tuner_methods[0].
                                    TUNER_TYPE,
                                    value=self.base_model_tuner_methods[0].
                                    TUNER_TYPE[0],
                                    multiselect=False,
                                    interactive=True)
                            with gr.Column(scale=1, min_width=0):
                                self.upload_tuner_name = gr.Text(
                                    label=self.component_names.utuner_name,
                                    show_label=False,
                                    container=False,
                                    placeholder='Upload Tuner Name',
                                    value='')
                                self.local_upload_bt = gr.Button(
                                    value=self.component_names.submit,
                                    elem_classes='type_row',
                                    elem_id='upload_button')

    def check_new_name(self, tuner_name):
        if tuner_name.strip() == '':
            return False, f"'{tuner_name}' is whitespace! Only support 'a-z', 'A-Z', '0-9', '@', '.', '-' and '_'."
        if not is_valid_filename(tuner_name):
            return False, (
                f"'{tuner_name}' is not a valid tuner name! "
                f"Only support 'a-z', 'A-Z', '0-9', '@', '.', '-' and '_'.")
        for tuner in self.saved_tuners:
            if tuner_name == tuner['NAME']:
                return False, f"Tuner name '{tuner_name}' has been taken!"
        return True, 'legal'

    def save_tuner_file_params(self, src_path, sub_dir, tuner_name, tuner_desc,
                               tuner_example, tuner_prompt_example):
        tar_path = os.path.join(self.work_dir, sub_dir)
        if not FS.exists(tar_path):
            FS.make_dir(tar_path)
        tar_path = os.path.join(tar_path, tuner_name)
        if FS.exists(tar_path):
            raise gr.Error(self.component_names.same_name)
        # local_model_dir, _ = FS.map_to_local(tar_path)
        local_model_dir = src_path
        FS.put_dir_from_local_dir(src_path, tar_path)
        # save image
        tuner_example_path = None
        if tuner_example is not None:
            from PIL import Image
            tuner_example_path = os.path.join(tar_path, 'image.jpg')
            local_example_path = os.path.join(local_model_dir, 'image.jpg')
            if not FS.exists(tuner_example_path):
                tuner_example = Image.fromarray(tuner_example)
                tuner_example.save(local_example_path)
                FS.put_object_from_local_file(local_example_path,
                                              tuner_example_path)

        # save param
        enable_share = True
        split_path = src_path.split('/')
        if len(split_path) > 1 and split_path[-2] == 'checkpoints':
            src_dir = '/'.join(split_path[:-2])
            ckpt_name = split_path[-1]
            meta_read = f'{src_dir}/meta_{ckpt_name}.yaml'
            meta_local = f'{local_model_dir}/params.yaml'
            meta_save = f'{tar_path}/params.yaml'
        else:
            meta_read = f'{src_path}/params.yaml'
            meta_local = f'{local_model_dir}/params.yaml'
            meta_save = f'{tar_path}/params.yaml'

        if os.path.exists(meta_read):
            meta = Config(cfg_file=meta_read)
            enable_share = Config.get_plain_cfg(meta.get('IS_SHARE', True))
            params = Config.get_plain_cfg(meta.get('PARAMS', {}))
            params['work_dir'] = ''
            params['work_name'] = ''
            params['USER_NAME'] = ''
            save_yaml(
                {
                    'PARAMS':
                    params,
                    'DESCRIPTION':
                    tuner_desc if len(tuner_desc) > 0 else tuner_name
                }, meta_local)
            FS.put_object_from_local_file(meta_local, meta_save)

            # rewrite readme
            with open(self.readme_file, 'r') as f:
                rc = f.read()

            rc = rc.replace(r'{MODEL_NAME}', tuner_name)
            rc = rc.replace(r'{MODEL_DESCRIPTION}',
                            tuner_desc if len(tuner_desc) > 0 else tuner_name)
            rc = rc.replace(r'{EVAL_PROMPT}', tuner_prompt_example)
            rc = rc.replace(r'{IMAGE_PATH}', './image.jpg')
            rc = rc.replace(r'{BASE_MODEL}',
                            meta['PARAMS'].get('base_model_revision', ''))
            rc = rc.replace(r'{TUNER_TYPE}',
                            meta['PARAMS'].get('tuner_name', ''))
            rc = rc.replace(r'{TRAIN_BATCH_SIZE}',
                            str(meta['PARAMS'].get('train_batch_size', '')))
            rc = rc.replace(r'{TRAIN_EPOCH}',
                            str(meta['PARAMS'].get('train_epoch', '')))
            rc = rc.replace(r'{LEARNING_RATE}',
                            str(meta['PARAMS'].get('learning_rate', '')))
            rc = rc.replace(r'{HEIGHT}',
                            str(meta['PARAMS'].get('resolution_height', '')))
            rc = rc.replace(r'{WIDTH}',
                            str(meta['PARAMS'].get('resolution_width', '')))
            rc = rc.replace(r'{DATA_TYPE}',
                            meta['PARAMS'].get('data_type', ''))
            rc = rc.replace(r'{MS_DATA_SPACE}',
                            meta['PARAMS'].get('ms_data_space', ''))
            rc = rc.replace(
                r'{MS_DATA_NAME}',
                meta['PARAMS']['ms_data_name'] if 'ms_data_name'
                in meta['PARAMS'] else meta['PARAMS'].get('ori_data_name', ''))
            rc = rc.replace(r'{MS_DATA_SUBNAME}',
                            meta['PARAMS'].get('ms_data_subname', ''))
            local_readme_file = os.path.join(local_model_dir, 'README.md')
            with open(local_readme_file, 'w') as f:
                f.write(rc)
            FS.put_object_from_local_file(local_readme_file,
                                          os.path.join(tar_path, 'README.md'))
        # repull to a official dir
        # FS.get_dir_to_local_dir(tar_path)
        return tar_path, tuner_example_path, enable_share

    def save_tuner(self, manager, tuner_name, new_name, tuner_desc,
                   tuner_example, tuner_prompt_example, base_model, tuner_type,
                   login_user_name):
        is_legal, msg = self.check_new_name(new_name)
        if not is_legal:
            raise gr.Error('Save failed because ' + msg)

        sub_dir = f'{base_model}-{tuner_type}'
        if hasattr(manager, 'self_train'):
            self_train_work_dir = manager.self_train.trainer_ui.work_dir_pre
        else:
            self_train_work_dir = self.work_dir
        if FS.exists(os.path.join(self_train_work_dir, '@'.join(tuner_name.split('@')[:-1]))) \
                and len(tuner_name.split('@')[:-1]) > 0:
            steps = tuner_name.split('@')[-1]
            source_dir = os.path.join(self_train_work_dir,
                                      '@'.join(tuner_name.split('@')[:-1]),
                                      'checkpoints', steps)
        else:
            source_dir = self.saved_tuners_category.get(
                login_user_name,
                OrderedDict()).get(sub_dir, {}).get(tuner_name,
                                                    {}).get('MODEL_PATH', '')
            if source_dir == '':
                raise gr.Error(self.component_names.model_err4 + tuner_name)

        model_dir, tuner_example, enable_share = self.save_tuner_file_params(
            source_dir, sub_dir, new_name, tuner_desc, tuner_example,
            tuner_prompt_example)
        # config info update
        new_tuner = {
            'NAME': new_name,
            'NAME_ZH': new_name,
            'SOURCE': 'self_train',
            'DESCRIPTION': tuner_desc,
            'BASE_MODEL': base_model,
            'USER_NAME': login_user_name,
            'MODEL_PATH': model_dir,
            'TUNER_TYPE': tuner_type,
            'PROMPT_EXAMPLE': tuner_prompt_example,
            'ENABLE_SHARE': enable_share,
        }
        if tuner_example is not None:
            new_tuner.update({'IMAGE_PATH': tuner_example.split('/')[-1]})
        pipeline_level_modules = manager.inference.model_manage_ui.pipe_manager.pipeline_level_modules
        if new_tuner['BASE_MODEL'] not in pipeline_level_modules:
            raise gr.Error(self.component_names.model_err3 +
                           new_tuner['BASE_MODEL'])
        pipeline_ins = pipeline_level_modules[new_tuner['BASE_MODEL']]
        now_diffusion_model = f"{new_tuner['BASE_MODEL']}_{pipeline_ins.diffusion_model['name']}"

        custom_tuner_choices = self.add_tuner(new_tuner,
                                              manager,
                                              now_diffusion_model,
                                              login_user_name=login_user_name)

        gr.Info('Successfully save tuner model!')

        return (gr.update(choices=list(
            self.saved_tuners_category.get(login_user_name, {}).keys()),
                          value=sub_dir),
                gr.update(choices=list(
                    self.saved_tuners_category.get(login_user_name,
                                                   {}).get(sub_dir,
                                                           {}).keys()),
                          value=new_name), gr.Text(value=new_name),
                gr.Dropdown(choices=custom_tuner_choices))

    def add_tuner(self, new_tuner, manager, now_diffusion_model,
                  login_user_name):
        self.saved_tuners.append(new_tuner)
        self.saved_tuners_to_category()
        with FS.put_to(self.yaml) as local_path:
            save_yaml({'TUNERS': self.saved_tuners}, local_path)

        # register to pipeline
        new_tuner = Config(cfg_dict=new_tuner, load=False)
        manager.inference.model_manage_ui.pipe_manager.register_tuner(
            new_tuner,
            name=new_tuner.NAME_ZH
            if self.language == 'zh' else new_tuner.NAME,
            is_customized=True)

        # update choices
        pipe_manager = manager.inference.model_manage_ui.pipe_manager
        now_pipeline = pipe_manager.model_level_info[now_diffusion_model][
            'pipeline'][0]
        default_choices = pipe_manager.module_level_choices
        custom_tunner_choices = []
        if 'customized_tuners' in default_choices and now_pipeline in default_choices[
                'customized_tuners']:
            custom_tunner_choices = default_choices['customized_tuners'][
                now_pipeline]['choices']

        # update tuner ui name_level_tuners
        name_level_tuners = manager.inference.tuner_ui.name_level_tuners
        if new_tuner.BASE_MODEL not in name_level_tuners:
            name_level_tuners[new_tuner.BASE_MODEL] = {}
        if self.language == 'zh':
            name_level_tuners[new_tuner.BASE_MODEL][
                new_tuner.NAME_ZH] = new_tuner
        else:
            name_level_tuners[new_tuner.BASE_MODEL][new_tuner.NAME] = new_tuner

        return custom_tunner_choices

    def delete_tuner(self, first_level, second_level, manager,
                     now_diffusion_model, login_user_name):
        self.saved_tuners_category[
            login_user_name], del_tuner = delete_2level_dict(
                self.saved_tuners_category.get(login_user_name, OrderedDict()),
                first_level, second_level)
        self.category_to_saved_tuners()
        save_yaml({'TUNERS': self.saved_tuners}, self.yaml)

        # update choices
        pipe_manager = manager.inference.model_manage_ui.pipe_manager
        now_pipeline = pipe_manager.model_level_info[now_diffusion_model][
            'pipeline'][0]
        default_choices = pipe_manager.module_level_choices
        custom_tuner_choices = []
        if 'customized_tuners' in default_choices and now_pipeline in default_choices[
                'customized_tuners']:
            custom_tuner_choices = default_choices['customized_tuners'][
                now_pipeline]['choices']

        # update tuner ui name_level_tuners
        del_tuner = Config(cfg_dict=del_tuner, load=False)
        name_level_tuners = manager.inference.tuner_ui.name_level_tuners
        if del_tuner.BASE_MODEL in name_level_tuners:
            if self.language == 'zh':
                del name_level_tuners[del_tuner.BASE_MODEL][del_tuner.NAME_ZH]
            else:
                del name_level_tuners[del_tuner.BASE_MODEL][del_tuner.NAME]

        return custom_tuner_choices

    def update_tuner_info(self, base_model, tuner_name, update_items,
                          login_user_name):
        if login_user_name not in self.saved_tuners_category:
            self.saved_tuners_category[login_user_name] = OrderedDict()
            self.saved_tuners_category[login_user_name][
                base_model] = OrderedDict()
        if tuner_name not in self.saved_tuners_category[login_user_name][
                base_model]:
            raise gr.Error(f'{tuner_name} not found in {base_model}')
        self.saved_tuners_category[base_model][tuner_name].update(update_items)
        self.category_to_saved_tuners()
        with FS.put_to(self.yaml) as local_path:
            save_yaml({'TUNERS': self.saved_tuners}, local_path)

    def set_callbacks(self, manager, info_ui):
        def refresh_browser(login_user_name):
            diffusion_models_choice, diffusion_model, tuner_models_choice, tuner_model = self.get_choices_and_values(
                login_user_name=login_user_name)
            return (gr.Dropdown(choices=diffusion_models_choice,
                                value=diffusion_model),
                    gr.Dropdown(choices=tuner_models_choice,
                                value=tuner_model))

        self.refresh_button.click(
            refresh_browser,
            inputs=[manager.user_name],
            outputs=[self.diffusion_models, self.tuner_models])

        def diffusion_model_change(diffusion_model, login_user_name):
            choices = list(
                self.saved_tuners_category.get(login_user_name,
                                               OrderedDict()).get(
                                                   diffusion_model, {}).keys())
            return gr.Dropdown(choices=choices,
                               value=choices[-1] if len(choices) > 0 else None)

        self.diffusion_models.change(
            diffusion_model_change,
            inputs=[self.diffusion_models, manager.user_name],
            outputs=[self.tuner_models],
            queue=True)

        def tuner_model_change(tuner_model, diffusion_model, login_user_name):
            if tuner_model is None:
                # fix refresh bug
                return (gr.Text(), gr.Text(), gr.Text(), gr.Text(), gr.Text(),
                        gr.Image(), gr.Text(), gr.Text(), gr.Text())

            tuner_info = self.saved_tuners_category.get(
                login_user_name, OrderedDict())[diffusion_model][tuner_model]
            local_model_dir, _ = FS.map_to_local(tuner_info['MODEL_PATH'])
            image_path = tuner_info.get('IMAGE_PATH', None)
            if image_path is not None:
                local_image_path = os.path.join(local_model_dir, image_path)
                if not FS.exists(local_image_path):
                    local_image_path = None
            else:
                local_image_path = None
            return (gr.Text(value=tuner_info.get('NAME', '')),
                    gr.Text(value=tuner_info.get('NAME', ''),
                            interactive=True),
                    gr.Text(value=tuner_info.get('TUNER_TYPE', '')),
                    gr.Text(value=tuner_info.get('BASE_MODEL', '')),
                    gr.Text(value=tuner_info.get('DESCRIPTION', ''),
                            interactive=True),
                    gr.Image(value=local_image_path),
                    gr.Text(value=tuner_info.get('PROMPT_EXAMPLE', ''),
                            interactive=True),
                    gr.Text(value=tuner_info.get('MODELSCOPE_URL', ''),
                            interactive=False),
                    gr.Text(value=tuner_info.get('HUGGINGFACE_URL', ''),
                            interactive=False))

        self.tuner_models.change(tuner_model_change,
                                 inputs=[
                                     self.tuner_models, self.diffusion_models,
                                     manager.user_name
                                 ],
                                 outputs=[
                                     info_ui.tuner_name,
                                     info_ui.new_name,
                                     info_ui.tuner_type,
                                     info_ui.base_model,
                                     info_ui.tuner_desc,
                                     info_ui.tuner_example,
                                     info_ui.tuner_prompt_example,
                                     info_ui.ms_url,
                                     info_ui.hf_url,
                                 ],
                                 queue=False)

        def save_tuner_func(tuner_name, new_name, tuner_desc, tuner_example,
                            tuner_prompt_example, base_model, tuner_type,
                            login_user_name):
            return self.save_tuner(manager, tuner_name, new_name, tuner_desc,
                                   tuner_example, tuner_prompt_example,
                                   base_model, tuner_type, login_user_name)

        self.save_button.click(
            save_tuner_func,
            inputs=[
                info_ui.tuner_name, info_ui.new_name, info_ui.tuner_desc,
                info_ui.tuner_example, info_ui.tuner_prompt_example,
                info_ui.base_model, info_ui.tuner_type, manager.user_name
            ],
            outputs=[
                self.diffusion_models, self.tuner_models, info_ui.tuner_name,
                manager.inference.tuner_ui.custom_tuner_model
            ],
            queue=True)

        def delete_tuner(tuner_name, tuner_type, base_model,
                         now_diffusion_model, login_user_name):
            first_level = f'{base_model}-{tuner_type}'
            second_level = f'{tuner_name}'
            custom_tuner_choices = self.delete_tuner(first_level, second_level,
                                                     manager,
                                                     now_diffusion_model,
                                                     login_user_name)
            return (gr.Dropdown(choices=list(
                self.saved_tuners_category.get(login_user_name,
                                               OrderedDict()).keys()),
                                value=None),
                    gr.Dropdown(choices=custom_tuner_choices))

        self.delete_button.click(
            delete_tuner,
            inputs=[
                info_ui.tuner_name, info_ui.tuner_type, info_ui.base_model,
                manager.inference.model_manage_ui.diffusion_model, manager.user_name
            ],
            outputs=[
                self.diffusion_models,
                manager.inference.tuner_ui.custom_tuner_model
            ],
            queue=True)

        def change_visible():
            return gr.update(visible=True), gr.update(visible=False)

        self.model_export.click(
            fn=change_visible,
            inputs=[],
            outputs=[self.export_setting, self.import_setting],
            queue=False)

        def change_visible_load_base_model_tuner():
            base_versions = list(self.base_model_tuner_methods_map.keys())
            default_version = base_versions[0]
            default_tuner_type_choices = self.base_model_tuner_methods_map[
                default_version]
            default_tuner_type = default_tuner_type_choices[0]

            return gr.update(visible=True), gr.update(visible=False), \
                gr.Dropdown(
                    label=self.component_names.ubase_model,
                    choices=base_versions,
                    value=default_version,
                    multiselect=False,
                    interactive=True), \
                gr.Dropdown(
                    label=self.component_names.utuner_type,
                    choices=default_tuner_type_choices,
                    value=default_tuner_type,
                    multiselect=False,
                    interactive=True
                )

        self.model_import.click(fn=change_visible_load_base_model_tuner,
                                inputs=[],
                                outputs=[
                                    self.import_setting, self.export_setting,
                                    self.upload_base_models,
                                    self.upload_tuner_type
                                ],
                                queue=False)

        def change_invisible():
            return gr.update(visible=False)

        self.export_close.click(fn=change_invisible,
                                inputs=[],
                                outputs=[self.export_setting],
                                queue=False)
        self.import_close.click(fn=change_invisible,
                                inputs=[],
                                outputs=[self.import_setting],
                                queue=False)

        def change_export_source(export_to):
            if export_to == 'modelscope':
                ms_visible = True
                hf_visible = False
            elif export_to == 'huggingface':
                hf_visible = True
                ms_visible = False

            return gr.update(visible=ms_visible), gr.update(visible=hf_visible)

        self.export_to.change(
            fn=change_export_source,
            inputs=[self.export_to],
            outputs=[self.ms_export_setting, self.hf_export_setting],
            queue=False)

        def change_import_source(import_from):
            if import_from == 'modelscope':
                ms_visible = True
                hf_visible = False
                local_visible = False
            elif import_from == 'huggingface':
                ms_visible = False
                hf_visible = True
                local_visible = False
            elif import_from == 'local':
                ms_visible = False
                hf_visible = False
                local_visible = True
            return gr.update(visible=ms_visible), gr.update(
                visible=hf_visible), gr.update(visible=local_visible)

        self.import_from.change(fn=change_import_source,
                                inputs=[self.import_from],
                                outputs=[
                                    self.ms_import_setting,
                                    self.hf_import_setting,
                                    self.local_import_setting
                                ],
                                queue=False)

        def push_to_modelscope(ms_sdk, username, private, base_model_name,
                               tuner_model_name, login_user_name):
            from swift import push_to_hub
            gr.Info('Start uploading tuner model to ModelScope!')
            if (isinstance(base_model_name, list)
                    and len(base_model_name) == 0) or (
                        isinstance(tuner_model_name, list)
                        and len(tuner_model_name) == 0
                    ) or tuner_model_name is None or (
                        base_model_name not in self.saved_tuners_category.get(
                            login_user_name, {}) and tuner_model_name
                        not in self.saved_tuners_category[login_user_name]
                        [base_model_name]):
                raise gr.Error(
                    'Please save model first or select a valid base model name.'
                )
            tuner = self.saved_tuners_category[login_user_name][
                base_model_name][tuner_model_name]

            enable_share = tuner.get('ENABLE_SHARE', True)
            if enable_share:
                if not is_valid_modelscope_filename(tuner_model_name):
                    raise gr.Error(
                        f"'{tuner_model_name}' is not a valid tuner name for modelscope! "
                        f"Only support 'a-z', 'A-Z', '0-9', '-' and '_'. Please rename it."
                    )
                repo_name = f'{username}/{tuner_model_name}'
                ckpt_path = tuner['MODEL_PATH']
                ms_url = f'https://www.modelscope.cn/models/{repo_name}'
                local_ckpt_path, _ = FS.map_to_local(ckpt_path)
                local_readme = os.path.join(local_ckpt_path, 'README.md')
                if FS.exists(local_readme):
                    with open(local_readme, 'r') as f:
                        rc = f.read()
                        rc = rc.replace(r'{MODEL_URL}', ms_url)
                        rc = rc.replace(r'{login_user_name}', username)
                    with open(local_readme, 'w') as f:
                        f.write(rc)
                local_configuration = os.path.join(local_ckpt_path,
                                                   'configuration.json')
                if not FS.exists(local_configuration):
                    with open(local_configuration, 'w') as f:
                        f.write('{}')

                push_status = push_to_hub(repo_name,
                                          local_ckpt_path,
                                          token=ms_sdk,
                                          private=private)
                if FS.exists(local_readme):
                    FS.put_object_from_local_file(
                        local_readme, os.path.join(ckpt_path, 'README.md'))
                if push_status:
                    update_items = {'MODELSCOPE_URL': ms_url}
                    self.update_tuner_info(base_model_name, tuner_model_name,
                                           update_items)
                    gr.Info(
                        'The tuner model has been uploaded to ModelScope Successfully!'
                    )
                    return update_items['MODELSCOPE_URL'], gr.update(
                        visible=False)
                else:
                    raise gr.Error(
                        'Error: The model failed to be uploaded to ModelScope!'
                    )
            else:
                raise gr.Error(
                    'Error: The model is not allowed to be shared to ModelScope!'
                )

        self.ms_export_submit.click(
            fn=push_to_modelscope,
            inputs=[
                self.ms_sdk, self.ms_export_username, self.ms_model_private,
                self.diffusion_models, self.tuner_models, manager.user_name
            ],
            outputs=[info_ui.ms_url, self.export_setting],
            queue=True)

        def pull_from_modelscope(modelid, username, login_user_name):
            gr.Info('Start pulling tuner model from ModelScope to Local!')
            src_path = f'ms://{username}/{modelid}'
            local_work_dir, _ = FS.map_to_local(src_path)
            FS.get_dir_to_local_dir(src_path, local_work_dir)

            meta_file = f'{local_work_dir}/{username}/{modelid}/params.yaml'
            if not os.path.exists(meta_file):
                raise gr.Error(
                    'The tuner model failed to be downloaded from ModelScope!')

            gr.Info(
                'The tuner model has been downloaded from ModelScope Successfully!'
            )

            #
            meta = Config(cfg_file=meta_file)
            base_model = meta['PARAMS']['base_model_revision']
            tuner_type = meta['PARAMS']['tuner_name']
            tuner_prompt_example = meta['PARAMS']['eval_prompts']
            if isinstance(tuner_prompt_example, list):
                tuner_prompt_example = tuner_prompt_example[0]
            tuner_category = f'{base_model}-{tuner_type}'
            new_name = f'modelscope@{username}@{modelid}'
            tar_path = os.path.join(
                self.work_dir,
                f'{tuner_category}/modelscope_{username}_{modelid}')
            FS.put_dir_from_local_dir(f'{local_work_dir}/{username}/{modelid}',
                                      tar_path)
            if os.path.exists(local_work_dir):
                shutil.rmtree(local_work_dir)

            new_tuner = {
                'NAME': new_name,
                'NAME_ZH': new_name,
                'USER_NAME': login_user_name,
                'SOURCE': 'modelscope',
                'DESCRIPTION': meta.get('DESCRIPTION', ''),
                'BASE_MODEL': base_model,
                'MODEL_PATH': tar_path,
                'IMAGE_PATH': 'image.jpg',
                'TUNER_TYPE': tuner_type,
                'PROMPT_EXAMPLE': tuner_prompt_example
            }
            pipeline_level_modules = manager.inference.model_manage_ui.pipe_manager.pipeline_level_modules
            if new_tuner['BASE_MODEL'] not in pipeline_level_modules:
                raise gr.Error(self.component_names.model_err3 +
                               new_tuner['BASE_MODEL'])
            pipeline_ins = pipeline_level_modules[new_tuner['BASE_MODEL']]
            now_diffusion_model = f"{new_tuner['BASE_MODEL']}_{pipeline_ins.diffusion_model['name']}"

            custom_tuner_choices = self.add_tuner(
                new_tuner,
                manager,
                now_diffusion_model,
                login_user_name=login_user_name)

            update_items = {
                'MODELSCOPE_URL':
                f'https://www.modelscope.cn/models/{username}/{modelid}'
            }
            self.update_tuner_info(tuner_category,
                                   new_name,
                                   update_items=update_items)

            return (gr.update(choices=list(
                self.saved_tuners_category.get(login_user_name,
                                               OrderedDict()).keys()),
                              value=tuner_category),
                    gr.update(choices=list(
                        self.saved_tuners_category.get(login_user_name,
                                                       OrderedDict()).get(
                                                           tuner_category,
                                                           {}).keys()),
                              value=new_name), gr.Text(value=new_name),
                    gr.Dropdown(choices=custom_tuner_choices),
                    gr.update(visible=False))

        self.ms_import_submit.click(
            fn=pull_from_modelscope,
            inputs=[
                self.ms_modelid,
                self.ms_import_username,
                manager.user_name
            ],
            outputs=[
                self.diffusion_models, self.tuner_models, info_ui.tuner_name,
                manager.inference.tuner_ui.custom_tuner_model,
                self.import_setting
            ],
            queue=True)

        def push_to_huggingface(sdk, username, private, base_model_name,
                                tuner_model_name, login_user_name):
            gr.Info('Start uploading tuner model to HuggingFace!')
            if (
                    isinstance(base_model_name, list)
                    and len(base_model_name) == 0
            ) or (isinstance(tuner_model_name, list) and len(tuner_model_name)
                  == 0) or tuner_model_name is None or (
                      base_model_name not in self.saved_tuners_category.get(
                          login_user_name, OrderedDict())
                      and tuner_model_name not in self.
                      saved_tuners_category[login_user_name][base_model_name]):
                raise gr.Error(
                    'Please save model first or select a valid base model name.'
                )
            tuner = self.saved_tuners_category[login_user_name][
                base_model_name][tuner_model_name]

            enable_share = tuner.get('ENABLE_SHARE', True)
            if enable_share:
                if not is_valid_huggingface_filename(tuner_model_name):
                    raise gr.Error(
                        f"'{tuner_model_name}' is not a valid tuner name for huggingface! "
                        f"Only support 'a-z', 'A-Z', '0-9', '-' and '_'. Please rename it."
                    )
                repo_name = f'{username}/{tuner_model_name}'
                ckpt_path = tuner['MODEL_PATH']
                hf_url = f'https://huggingface.co/{repo_name}'

                local_readme = os.path.join(ckpt_path, 'README.md')
                if FS.exists(local_readme):
                    with open(local_readme, 'r') as f:
                        rc = f.read()
                        rc = rc.replace(r'{MODEL_URL}', hf_url)
                        rc = rc.replace(r'{login_user_name}', username)
                    with open(local_readme, 'w') as f:
                        f.write(rc)
                local_configuration = os.path.join(ckpt_path,
                                                   'configuration.json')
                if not FS.exists(local_configuration):
                    with open(local_configuration, 'w') as f:
                        f.write('{}')

                api = HfApi(token=sdk)
                api.create_repo(repo_id=repo_name,
                                private=private,
                                exist_ok=True)
                commit_info = api.upload_folder(folder_path=ckpt_path,
                                                repo_id=repo_name,
                                                repo_type='model',
                                                commit_message='')
                if commit_info:
                    update_items = {'HUGGINGFACE_URL': hf_url}
                    self.update_tuner_info(base_model_name, tuner_model_name,
                                           update_items)
                    gr.Info(
                        'The tuner model has been uploaded to HuggingFace Successfully!'
                    )
                    return update_items['HUGGINGFACE_URL'], gr.update(
                        visible=False)
                else:
                    raise gr.Error(
                        'Error: The model failed to be uploaded to HuggingFace!'
                    )
            else:
                raise gr.Error(
                    'Error: The model is not allowed to be shared to HuggingFace!'
                )

        self.hf_export_submit.click(
            fn=push_to_huggingface,
            inputs=[
                self.hf_sdk, self.hf_export_username, self.hf_model_private,
                self.diffusion_models, self.tuner_models, manager.user_name
            ],
            outputs=[info_ui.hf_url, self.export_setting],
            queue=True)

        def pull_from_huggingface(modelid, username, hf_sdk, login_user_name):
            gr.Info('Start pulling tuner model from HuggingFace to Local!')
            src_path = f'{username}/{modelid}'

            local_tar_path = f'cache/temp_dir/{src_path}'
            snapshot_download(repo_id=src_path,
                              repo_type='model',
                              local_dir=local_tar_path,
                              local_dir_use_symlinks=False,
                              token=hf_sdk if len(hf_sdk) > 0 else None)

            meta_file = f'{local_tar_path}/params.yaml'
            if not os.path.exists(meta_file):
                raise gr.Error(
                    'The tuner model failed to be downloaded from HuggingFace!'
                )

            gr.Info(
                'The tuner model has been downloaded from HuggingFace Successfully!'
            )
            meta = Config(cfg_file=meta_file)
            base_model = meta['PARAMS']['base_model_revision']
            tuner_type = meta['PARAMS']['tuner_name']
            tuner_prompt_example = meta['PARAMS']['eval_prompts']
            if isinstance(tuner_prompt_example, list):
                tuner_prompt_example = tuner_prompt_example[0]
            tuner_category = f'{base_model}-{tuner_type}'
            new_name = f'huggingface@{username}@{modelid}'
            tar_path = os.path.join(
                self.work_dir,
                f'{tuner_category}/huggingface_{username}_{modelid}')
            FS.put_dir_from_local_dir(local_tar_path, tar_path)
            if os.path.exists(local_tar_path):
                shutil.rmtree(local_tar_path)

            new_tuner = {
                'NAME': new_name,
                'NAME_ZH': new_name,
                'SOURCE': 'huggingface',
                'DESCRIPTION': meta.get('DESCRIPTION', ''),
                'BASE_MODEL': base_model,
                'MODEL_PATH': tar_path,
                'IMAGE_PATH': 'image.jpg',
                'TUNER_TYPE': tuner_type,
                'PROMPT_EXAMPLE': tuner_prompt_example
            }
            pipeline_level_modules = manager.inference.model_manage_ui.pipe_manager.pipeline_level_modules
            if new_tuner['BASE_MODEL'] not in pipeline_level_modules:
                raise gr.Error(self.component_names.model_err3 +
                               new_tuner['BASE_MODEL'])
            pipeline_ins = pipeline_level_modules[new_tuner['BASE_MODEL']]
            now_diffusion_model = f"{new_tuner['BASE_MODEL']}_{pipeline_ins.diffusion_model['name']}"

            custom_tuner_choices = self.add_tuner(new_tuner, manager,
                                                  now_diffusion_model,
                                                  login_user_name)

            update_items = {
                'HUGGINGFACE_URL':
                f'https://huggingface.co/{username}/{modelid}'
            }
            self.update_tuner_info(tuner_category,
                                   new_name,
                                   update_items=update_items,
                                   login_user_name=login_user_name)

            return (gr.update(choices=list(
                self.saved_tuners_category.get(login_user_name,
                                               OrderedDict()).keys()),
                              value=tuner_category),
                    gr.update(choices=list(
                        self.saved_tuners_category.get(login_user_name,
                                                       OrderedDict()).get(
                                                           tuner_category,
                                                           {}).keys()),
                              value=new_name), gr.Text(value=new_name),
                    gr.Dropdown(choices=custom_tuner_choices),
                    gr.update(visible=False))

        self.hf_import_submit.click(
            fn=pull_from_huggingface,
            inputs=[
                self.hf_modelid, self.hf_import_username, self.hf_sdk2,
                manager.user_name
            ],
            outputs=[
                self.diffusion_models, self.tuner_models, info_ui.tuner_name,
                manager.inference.tuner_ui.custom_tuner_model,
                self.import_setting
            ],
            queue=True)

        def change_tuner_type_by_model_version(base_model_revision):
            if base_model_revision in self.base_model_tuner_methods_map:
                base_model_tuner = self.base_model_tuner_methods_map[
                    base_model_revision]
                return gr.Dropdown(value=base_model_tuner[0],
                                   choices=base_model_tuner,
                                   interactive=True)
            return gr.Dropdown(value='', choices=[], interactive=True)

        self.upload_base_models.change(fn=change_tuner_type_by_model_version,
                                       inputs=[self.upload_base_models],
                                       outputs=[self.upload_tuner_type],
                                       queue=False)

        def analyse_lora(model_path, model_dir):
            from safetensors import safe_open
            import json
            import torch
            civitai_lora = {}
            with safe_open(model_path, framework='pt', device='cpu') as f:
                for k in f.keys():
                    civitai_lora[k] = f.get_tensor(k)
            lora_config, swift_lora, unload_params = convert_tuner_civitai_to_scepter(
                civitai_lora)
            local_model_dir, _ = FS.map_to_local(model_dir)
            os.makedirs(local_model_dir, exist_ok=True)
            config_path = os.path.join(model_dir, '0_SwiftLoRA',
                                       'adapter_config.json')
            module_path = os.path.join(model_dir, '0_SwiftLoRA',
                                       'adapter_model.bin')
            with FS.put_to(config_path) as local_config:
                with open(local_config, 'w') as fw:
                    fw.write(json.dumps(lora_config))
            with FS.put_to(module_path) as local_module:
                torch.save(swift_lora, local_module)
            FS.get_dir_to_local_dir(model_dir, local_model_dir)
            return model_dir, local_model_dir

        def upload_zip(file_path, file_url, tuner_name, base_model, tuner_type,
                       login_user_name):
            sub_dir = f'{base_model}-{tuner_type}'
            sub_work_dir = os.path.join(self.work_dir, sub_dir)
            if not FS.exists(sub_work_dir):
                FS.make_dir(sub_work_dir)
            model_dir = os.path.join(sub_work_dir, tuner_name)
            if FS.exists(model_dir):
                raise gr.Error(self.component_names.same_name)
            if file_url == '':
                if file_path is None:
                    raise gr.Error(self.component_names.files_null)
                model_input_path = file_path.name
            else:
                if '.zip' in file_url:
                    save_file = os.path.join(self.work_dir, sub_dir,
                                             f'{tuner_name}.zip')
                elif '.safetensors' in file_url:
                    save_file = os.path.join(self.work_dir, sub_dir,
                                             f'{tuner_name}.safetensors')
                else:
                    raise gr.Error(self.component_names.url_invalid)
                save_file, _ = wget_file(file_url, save_file=save_file)
                model_input_path, _ = FS.map_to_local(save_file)
            if not FS.exists(model_input_path):
                raise gr.Error(self.component_names.upload_file_error2)

            if model_input_path.endswith('.safetensors'):
                # analyse lora from civitai.com
                model_dir, local_model_dir = analyse_lora(
                    model_input_path, model_dir)
            else:
                save_file = os.path.join(self.work_dir, sub_dir,
                                         f'{tuner_name}.zip')
                FS.put_object_from_local_file(model_input_path, save_file)
                local_model_dir, _ = FS.map_to_local(model_dir)
                os.makedirs(local_model_dir, exist_ok=True)
                with FS.get_from(save_file) as local_path:
                    res = os.popen(
                        f"unzip -o '{local_path}' -d '{local_model_dir}/tmp'")
                    res = res.readlines()
                local_model_subdir = ''
                for root_path, dirs, files in os.walk(f'{local_model_dir}/tmp',
                                                      topdown=False):
                    if 'adapter_model.bin' in files:
                        local_model_subdir = '/'.join(
                            root_path.split('/')[:-1])
                        break
                if local_model_subdir == '':
                    raise gr.Error(self.component_names.upload_file_error)
                res = os.popen(
                    f"cp -r '{local_model_subdir}'/* '{local_model_dir}' "
                    f"&& rm -rf '{local_model_dir}'/tmp")
                res = res.readlines()

                FS.put_dir_from_local_dir(local_model_dir, model_dir)
                if not FS.exists(model_dir):
                    raise gr.Error(f'unzip {save_file} failed, {str(res)}')

            # find meta.yaml
            if FS.exists(os.path.join(model_dir, 'meta.yaml')):
                local_meta_file = os.path.join(local_model_dir, 'meta.yaml')
                if not FS.exists(local_meta_file):
                    FS.get_from(os.path.join(model_dir, 'meta.yaml'),
                                local_meta_file)
                meta = Config(cfg_file=local_meta_file)
                tuner_desc = meta.get('DESCRIPTION', '')
                tuner_example = meta.get('IMAGE_PATH', None)
                if tuner_example is not None:
                    tuner_example = os.path.join(
                        model_dir, os.path.basename(tuner_example))
                    if FS.exists(tuner_example):
                        tuner_example = tuner_example.split('/')[-1]
                    elif FS.exists(os.path.join(model_dir, 'image.jpg')):
                        tuner_example = 'image.jpg'
                    else:
                        tuner_example = None
                tuner_prompt_example = meta.get('PROMPT_EXAMPLE', '')
            else:
                tuner_desc = ''
                tuner_example = None
                tuner_prompt_example = ''

            if not FS.exists(model_dir):
                raise gr.Error(
                    f'{self.component_names.illegal_data_err1}{str(res)}')
            # config info update
            new_tuner = {
                'NAME': tuner_name,
                'NAME_ZH': tuner_name,
                'SOURCE': 'self_train',
                'USER_NAME': login_user_name,
                'DESCRIPTION': tuner_desc,
                'BASE_MODEL': base_model,
                'MODEL_PATH': model_dir,
                'TUNER_TYPE': tuner_type,
                'PROMPT_EXAMPLE': tuner_prompt_example
            }
            if tuner_example is not None:
                new_tuner.update({'IMAGE_PATH': tuner_example})
                local_tuner_example = os.path.join(local_model_dir,
                                                   tuner_example)
            else:
                local_tuner_example = None
            pipeline_level_modules = manager.inference.model_manage_ui.pipe_manager.pipeline_level_modules
            if new_tuner['BASE_MODEL'] not in pipeline_level_modules:
                raise gr.Error(self.component_names.model_err3 +
                               new_tuner['BASE_MODEL'])
            pipeline_ins = pipeline_level_modules[new_tuner['BASE_MODEL']]
            now_diffusion_model = f"{new_tuner['BASE_MODEL']}_{pipeline_ins.diffusion_model['name']}"

            custom_tuner_choices = self.add_tuner(
                new_tuner,
                manager,
                now_diffusion_model,
                login_user_name=login_user_name)
            gr.Info(self.component_names.upload_success)

            return (gr.Dropdown(choices=list(
                self.saved_tuners_category.get(login_user_name,
                                               OrderedDict()).keys()),
                                value=sub_dir),
                    gr.Dropdown(choices=list(
                        self.saved_tuners_category.get(login_user_name,
                                                       OrderedDict()).get(
                                                           sub_dir,
                                                           {}).keys()),
                                value=tuner_name), gr.Text(value=tuner_name),
                    gr.Text(value=tuner_type), gr.Text(value=base_model),
                    gr.Text(value=tuner_desc),
                    gr.Text(value=tuner_prompt_example),
                    gr.Image(value=local_tuner_example),
                    gr.Dropdown(choices=custom_tuner_choices),
                    gr.update(visible=False))

        self.local_upload_bt.click(
            upload_zip,
            inputs=[
                self.file_path, self.file_url, self.upload_tuner_name,
                self.upload_base_models, self.upload_tuner_type, manager.user_name
            ],
            outputs=[
                self.diffusion_models, self.tuner_models, info_ui.tuner_name,
                info_ui.tuner_type, info_ui.base_model, info_ui.tuner_desc,
                info_ui.tuner_prompt_example, info_ui.tuner_example,
                manager.inference.tuner_ui.custom_tuner_model,
                self.import_setting
            ],
            queue=False)
