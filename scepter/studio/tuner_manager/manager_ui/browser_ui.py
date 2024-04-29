# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from collections import OrderedDict

import gradio as gr
from swift import push_to_hub

import scepter
from scepter.modules.utils.config import Config
from scepter.modules.utils.file_system import FS
from scepter.studio.tuner_manager.manager_ui.component_names import \
    TunerManagerNames
from scepter.studio.tuner_manager.utils.dict import (delete_2level_dict,
                                                     update_2level_dict)
from scepter.studio.tuner_manager.utils.path import is_valid_filename
from scepter.studio.tuner_manager.utils.yaml import save_yaml
from scepter.studio.utils.uibase import UIBase


class BrowserUI(UIBase):
    def __init__(self, cfg, language='en'):
        self.work_dir = cfg.WORK_DIR
        self.train_dir = cfg.SELF_TRAIN_DIR
        self.base_model_tuner_methods = cfg.BASE_MODEL_VERSION
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
        self.readme_file = os.path.join(os.path.dirname(scepter.dirname), self.readme_file)

    def saved_tuners_to_category(self):
        self.saved_tuners_category = OrderedDict()
        for tuner in self.saved_tuners:
            first_level = f"{tuner['BASE_MODEL']}-{tuner['TUNER_TYPE']}"
            second_level = f"{tuner['NAME']}"
            update_2level_dict(self.saved_tuners_category,
                               {first_level: {
                                   second_level: tuner
                               }})

    def category_to_saved_tuners(self):
        self.saved_tuners = []
        for k, v in self.saved_tuners_category.items():
            for kk, vv in v.items():
                self.saved_tuners.append(vv)

    def get_choices_and_values(self):
        diffusion_models_choice = list(self.saved_tuners_category.keys())
        diffusion_model = diffusion_models_choice[0] if len(
            diffusion_models_choice) > 0 else None
        tuner_models_choice = []
        tuner_model = None
        if diffusion_model:
            tuner_models_choice = list(
                self.saved_tuners_category.get(diffusion_model, {}).keys())
            tuner_model = tuner_models_choice[0] if len(
                tuner_models_choice) > 0 else None
        return diffusion_models_choice, diffusion_model, tuner_models_choice, tuner_model

    def create_ui(self, *args, **kwargs):
        diffusion_models_choice, diffusion_model, tuner_models_choice, tuner_model = self.get_choices_and_values(
        )
        with gr.Column():
            with gr.Box():
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
                            label='Save',
                            value=self.component_names.save_symbol,
                            elem_classes='type_row',
                            elem_id='save_button',
                            visible=True)
                        self.delete_button = gr.Button(
                            label='Delete',
                            value=self.component_names.delete_symbol,
                            elem_classes='type_row',
                            elem_id='delete_button',
                            visible=False)
                        self.model_upload = gr.Button(
                            label='ModelScope Upload',
                            value=self.component_names.upload,
                            elem_classes='type_row',
                            elem_id='save_button',
                            visible=True)
                    with gr.Column(scale=1, min_width=0):
                        self.refresh_button = gr.Button(
                            label='Refresh',
                            value=self.component_names.refresh_symbol,
                            elem_classes='type_row',
                            elem_id='refresh_button',
                            visible=True)
                        self.model_download = gr.Button(
                            label='ModelScope Download',
                            value=self.component_names.download,
                            elem_classes='type_row',
                            elem_id='save_button',
                            visible=True)

                with gr.Box(visible=False) as self.upload_setting:
                    gr.Markdown(self.component_names.export_desc)
                    with gr.Column(variant='panel'):
                        with gr.Row(equal_height=True):
                            with gr.Column(scale=9, min_width=0):
                                self.import_src = gr.Dropdown(
                                    choices=['modelscope'],
                                    value='modelscope',
                                    label=None,
                                    show_label=False)
                            with gr.Column(scale=1, min_width=0):
                                self.ms_upload_close = gr.Button(
                                    label='Close MS Upload',
                                    value=self.component_names.close,
                                    elem_classes='type_row',
                                    elem_id='save_button')
                        with gr.Row(equal_height=True):
                            with gr.Column(scale=3, min_width=0):
                                self.ms_sdk = gr.Text(
                                    label=self.component_names.ms_sdk,
                                    show_label=False,
                                    container=False,
                                    placeholder='ModelScope SDK Token',
                                    value='')
                            with gr.Column(scale=3, min_width=0):
                                self.ms_upload_username = gr.Text(
                                    label=self.component_names.ms_username,
                                    show_label=False,
                                    container=False,
                                    placeholder='ModelScope UserName',
                                    value='')
                            with gr.Column(scale=3, min_width=0):
                                self.model_private = gr.Checkbox(
                                    label=self.component_names.model_private,
                                    value=False)
                            with gr.Column(scale=1, min_width=0):
                                self.ms_upload_submit = gr.Button(
                                    label='Submit MS',
                                    value=self.component_names.ms_submit,
                                    elem_classes='type_row',
                                    elem_id='save_button')

                with gr.Box(visible=False) as self.download_setting:
                    gr.Markdown(self.component_names.import_desc)
                    with gr.Column(variant='panel'):
                        with gr.Row(equal_height=True):
                            with gr.Column(scale=9, min_width=0):
                                self.import_src = gr.Dropdown(
                                    choices=['modelscope', 'local'],
                                    value='modelscope',
                                    label=None,
                                    show_label=False)
                            with gr.Column(scale=1, min_width=0):
                                self.ms_download_close = gr.Button(
                                    label='Close MS Download',
                                    value=self.component_names.close,
                                    elem_classes='type_row',
                                    elem_id='save_button')
                        with gr.Row(
                                equal_height=True) as self.ms_import_setting:
                            with gr.Column(scale=4.5, min_width=0):
                                self.ms_modelid = gr.Text(
                                    label=self.component_names.ms_modelid,
                                    show_label=False,
                                    container=False,
                                    placeholder='ModelScope Model Path',
                                    value='')
                            with gr.Column(scale=4.5, min_width=0):
                                self.ms_download_username = gr.Text(
                                    label=self.component_names.ms_username,
                                    show_label=False,
                                    container=False,
                                    placeholder='ModelScope UserName',
                                    value='')
                            with gr.Column(scale=1, min_width=0):
                                self.ms_download_submit = gr.Button(
                                    label='Submit MS',
                                    value=self.component_names.ms_submit,
                                    elem_classes='type_row',
                                    elem_id='save_button')
                        with gr.Row(visible=False, equal_height=True
                                    ) as self.local_import_setting:
                            with gr.Column(scale=2, min_width=0):
                                self.file_path = gr.File(
                                    label=self.component_names.zip_file,
                                    min_width=0,
                                    file_types=['.zip'],
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
                                    label='Submit Local Model',
                                    value=self.component_names.ms_submit,
                                    elem_classes='type_row',
                                    elem_id='upload_button')

    def check_new_name(self, tuner_name):
        if tuner_name.strip() == '':
            return False, f"'{tuner_name}' is whitespace! Only support 'a-z', 'A-Z', '0-9' and '_'."
        if not is_valid_filename(tuner_name):
            return False, f"'{tuner_name}' is not a valid tuner name! Only support 'a-z', 'A-Z', '0-9' and '_'."
        for tuner in self.saved_tuners:
            if tuner_name == tuner['NAME']:
                return False, f"Tuner name '{tuner_name}' has been taken!"
        return True, 'legal'

    def save_tuner(self, src_path, sub_dir, tuner_name, tuner_desc,
                   tuner_example, tuner_prompt_example):
        tar_path = os.path.join(self.work_dir, sub_dir)
        if not FS.exists(tar_path):
            FS.make_dir(tar_path)
        tar_path = os.path.join(tar_path, tuner_name)

        FS.put_dir_from_local_dir(src_path, tar_path)

        # save image
        tuner_example_path = None
        if tuner_example is not None:
            from PIL import Image
            tuner_example_path = os.path.join(tar_path, 'image.jpg')
            if not os.path.exists(tuner_example_path):
                tuner_example = Image.fromarray(tuner_example)
                with FS.put_to(tuner_example_path) as local_path:
                    tuner_example.save(local_path)

        # save param
        enable_share = True
        split_path = src_path.split('/')
        if split_path[-2] == 'checkpoints':
            src_dir = '/'.join(split_path[:-2])
            ckpt_name = split_path[-1]
            meta_read = f'{src_dir}/meta_{ckpt_name}.yaml'
            meta_save = f'{tar_path}/params.yaml'
        else:
            meta_read = f'{src_path}/params.yaml'
            meta_save = f'{tar_path}/params.yaml'

        if os.path.exists(meta_read):
            meta = Config(cfg_file=meta_read)
            with FS.put_to(meta_save) as local_path:
                enable_share = Config.get_plain_cfg(meta.get('IS_SHARE', True))
                params = Config.get_plain_cfg(meta.get('PARAMS', {}))
                params['work_dir'] = ''
                params['work_name'] = ''
                save_yaml({'PARAMS': params}, local_path)

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
            rc = rc.replace(r'{MS_DATA_NAME}',
                            meta['PARAMS'].get('ms_data_name', ''))
            rc = rc.replace(r'{MS_DATA_SUBNAME}',
                            meta['PARAMS'].get('ms_data_subname', ''))

            with FS.put_to(os.path.join(tar_path, 'README.md')) as local_path:
                with open(local_path, 'w') as f:
                    f.write(rc)

        return tar_path, tuner_example_path, enable_share

    def add_tuner(self, new_tuner, manager, now_diffusion_model):
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
                     now_diffusion_model):
        self.saved_tuners_category, del_tuner = delete_2level_dict(
            self.saved_tuners_category, first_level, second_level)
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

    def update_tuner_info(self, base_model, tuner_name, update_items):
        self.saved_tuners_category[base_model][tuner_name].update(update_items)
        self.category_to_saved_tuners()
        with FS.put_to(self.yaml) as local_path:
            save_yaml({'TUNERS': self.saved_tuners}, local_path)

    def set_callbacks(self, manager, info_ui):
        def refresh_browser():
            diffusion_models_choice, diffusion_model, tuner_models_choice, tuner_model = self.get_choices_and_values(
            )
            return (gr.Dropdown(choices=diffusion_models_choice,
                                value=diffusion_model),
                    gr.Dropdown(choices=tuner_models_choice,
                                value=tuner_model))

        self.refresh_button.click(
            refresh_browser,
            inputs=[],
            outputs=[self.diffusion_models, self.tuner_models])

        def diffusion_model_change(diffusion_model):
            choices = list(
                self.saved_tuners_category.get(diffusion_model, {}).keys())
            return gr.Dropdown(choices=choices,
                               value=choices[-1] if len(choices) > 0 else None)

        self.diffusion_models.change(diffusion_model_change,
                                     inputs=[self.diffusion_models],
                                     outputs=[self.tuner_models],
                                     queue=True)

        def tuner_model_change(tuner_model, diffusion_model):
            if tuner_model is None:
                # fix refresh bug
                return (gr.Text(), gr.Text(), gr.Text(), gr.Text(), gr.Text(),
                        gr.Image(), gr.Text(), gr.Text())

            tuner_info = self.saved_tuners_category[diffusion_model][
                tuner_model]
            image_path = tuner_info.get('IMAGE_PATH', None)
            if image_path is not None:
                image_path = FS.get_from(image_path)
            return (gr.Text(value=tuner_info.get('NAME', '')),
                    gr.Text(value=tuner_info.get('NAME', ''),
                            interactive=True),
                    gr.Text(value=tuner_info.get('TUNER_TYPE', '')),
                    gr.Text(value=tuner_info.get('BASE_MODEL', '')),
                    gr.Text(value=tuner_info.get('DESCRIPTION', ''),
                            interactive=True), gr.Image(value=image_path),
                    gr.Text(value=tuner_info.get('PROMPT_EXAMPLE', ''),
                            interactive=True),
                    gr.Text(value=tuner_info.get('MODELSCOPE_URL', ''),
                            interactive=False))

        self.tuner_models.change(
            tuner_model_change,
            inputs=[self.tuner_models, self.diffusion_models],
            outputs=[
                info_ui.tuner_name, info_ui.new_name, info_ui.tuner_type,
                info_ui.base_model, info_ui.tuner_desc, info_ui.tuner_example,
                info_ui.tuner_prompt_example, info_ui.ms_url
            ],
            queue=False)

        def save_tuner(tuner_name, new_name, tuner_desc, tuner_example,
                       tuner_prompt_example, base_model, tuner_type):
            is_legal, msg = self.check_new_name(new_name)
            if not is_legal:
                gr.Info('Save failed because ' + msg)
                return (gr.Dropdown(), gr.Dropdown(), gr.Text(), gr.Dropdown())

            sub_dir = f'{base_model}-{tuner_type}'
            if os.path.exists(os.path.join(self.train_dir, '@'.join(tuner_name.split('@')[:-1]))) \
                    and len(tuner_name.split('@')[:-1]) > 0:
                steps = tuner_name.split('@')[-1]
                model_dir = os.path.join(self.train_dir,
                                         '@'.join(tuner_name.split('@')[:-1]),
                                         'checkpoints', steps)
            else:
                model_dir = self.saved_tuners_category.get(sub_dir, {}).get(
                    tuner_name, {}).get('MODEL_PATH', '')
                if model_dir == '':
                    gr.Error(self.component_names.model_err4 + tuner_name)

            model_dir, tuner_example, enable_share = self.save_tuner(
                model_dir, sub_dir, new_name, tuner_desc, tuner_example,
                tuner_prompt_example)
            # config info update
            new_tuner = {
                'NAME': new_name,
                'NAME_ZH': new_name,
                'SOURCE': 'self_train',
                'DESCRIPTION': tuner_desc,
                'BASE_MODEL': base_model,
                'MODEL_PATH': model_dir,
                'IMAGE_PATH': tuner_example,
                'TUNER_TYPE': tuner_type,
                'PROMPT_EXAMPLE': tuner_prompt_example,
                'ENABLE_SHARE': enable_share,
            }
            pipeline_level_modules = manager.inference.model_manage_ui.pipe_manager.pipeline_level_modules
            if new_tuner['BASE_MODEL'] not in pipeline_level_modules:
                gr.Error(self.component_names.model_err3 +
                         new_tuner['BASE_MODEL'])
            pipeline_ins = pipeline_level_modules[new_tuner['BASE_MODEL']]
            now_diffusion_model = f"{new_tuner['BASE_MODEL']}_{pipeline_ins.diffusion_model['name']}"

            custom_tuner_choices = self.add_tuner(new_tuner, manager,
                                                  now_diffusion_model)

            return (gr.update(choices=list(self.saved_tuners_category.keys()),
                              value=sub_dir),
                    gr.update(choices=list(
                        self.saved_tuners_category.get(sub_dir, {}).keys()),
                              value=new_name), gr.Text(value=new_name),
                    gr.Dropdown(choices=custom_tuner_choices))

        self.save_button.click(
            save_tuner,
            inputs=[
                info_ui.tuner_name, info_ui.new_name, info_ui.tuner_desc,
                info_ui.tuner_example, info_ui.tuner_prompt_example,
                info_ui.base_model, info_ui.tuner_type
            ],
            outputs=[
                self.diffusion_models, self.tuner_models, info_ui.tuner_name,
                manager.inference.tuner_ui.custom_tuner_model
            ],
            queue=True)

        def delete_tuner(tuner_name, tuner_type, base_model,
                         now_diffusion_model):
            first_level = f'{base_model}-{tuner_type}'
            second_level = f'{tuner_name}'
            custom_tuner_choices = self.delete_tuner(first_level, second_level,
                                                     manager,
                                                     now_diffusion_model)
            return (gr.Dropdown(
                choices=list(self.saved_tuners_category.keys()),
                value=None), gr.Dropdown(choices=custom_tuner_choices))

        self.delete_button.click(
            delete_tuner,
            inputs=[
                info_ui.tuner_name, info_ui.tuner_type, info_ui.base_model,
                manager.inference.model_manage_ui.diffusion_model
            ],
            outputs=[
                self.diffusion_models,
                manager.inference.tuner_ui.custom_tuner_model
            ],
            queue=True)

        def change_visible():
            return gr.update(visible=True), gr.update(visible=False)

        self.model_upload.click(
            fn=change_visible,
            inputs=[],
            outputs=[self.upload_setting, self.download_setting],
            queue=False)
        self.model_download.click(
            fn=change_visible,
            inputs=[],
            outputs=[self.download_setting, self.upload_setting],
            queue=False)

        def change_invisible():
            return gr.update(visible=False)

        self.ms_upload_close.click(fn=change_invisible,
                                   inputs=[],
                                   outputs=[self.upload_setting],
                                   queue=False)
        self.ms_download_close.click(fn=change_invisible,
                                     inputs=[],
                                     outputs=[self.download_setting],
                                     queue=False)

        def change_import_source(import_src):
            if import_src == 'modelscope':
                ms_visible = True
                local_visible = False
            elif import_src == 'local':
                ms_visible = False
                local_visible = True
            return gr.update(visible=ms_visible), gr.update(
                visible=local_visible)

        self.import_src.change(
            fn=change_import_source,
            inputs=[self.import_src],
            outputs=[self.ms_import_setting, self.local_import_setting],
            queue=False)

        def push_to_modelscope(ms_sdk, username, private, base_model_name,
                               tuner_model_name):
            tuner = self.saved_tuners_category[base_model_name][
                tuner_model_name]

            enable_share = tuner.get('ENABLE_SHARE', True)
            if enable_share:
                repo_name = f'{username}/{tuner_model_name}'
                ckpt_path = tuner['MODEL_PATH']
                ms_url = f'https://www.modelscope.cn/models/{repo_name}'

                with open(os.path.join(ckpt_path, 'README.md'), 'r') as f:
                    rc = f.read()
                    rc = rc.replace(r'{MODEL_URL}', ms_url)
                    rc = rc.replace(r'{USER_NAME}', username)
                with open(os.path.join(ckpt_path, 'README.md'), 'w') as f:
                    f.write(rc)

                push_status = push_to_hub(repo_name,
                                          ckpt_path,
                                          token=ms_sdk,
                                          private=private)
                if push_status:
                    update_items = {'MODELSCOPE_URL': ms_url}
                    self.update_tuner_info(base_model_name, tuner_model_name,
                                           update_items)
                    gr.Info(
                        'The tuner model has been uploaded to ModelScope Successfully!'
                    )
                    return update_items['MODELSCOPE_URL']
                else:
                    gr.Info(
                        'Error: The model failed to be uploaded to ModelScope!'
                    )
                    return ''
            else:
                gr.Info(
                    'Error: The model is not allowed to be shared to ModelScope!'
                )
                return ''

        self.ms_upload_submit.click(fn=push_to_modelscope,
                                    inputs=[
                                        self.ms_sdk, self.ms_upload_username,
                                        self.model_private,
                                        self.diffusion_models,
                                        self.tuner_models
                                    ],
                                    outputs=[info_ui.ms_url],
                                    queue=True)

        def pull_from_modelscope(modelid, username):
            tar_path = os.path.join(self.work_dir, 'modelscope')
            src_path = f'ms://{username}/{modelid}'
            FS.get_dir_to_local_dir(src_path, tar_path)
            gr.Info(
                'The tuner model has been downloaded from ModelScope Successfully!'
            )

            tar_path = f'{tar_path}/{username}/{modelid}'
            meta_file = f'{tar_path}/params.yaml'
            meta = Config(cfg_file=meta_file)
            base_model = meta['PARAMS']['base_model_revision']
            tuner_type = meta['PARAMS']['tuner_name']
            tuner_prompt_example = meta['PARAMS']['eval_prompts'][0]
            tuner_category = f'{base_model}-{tuner_type}'
            new_name = f'modelscope@{username}@{modelid}'
            new_tuner = {
                'NAME': new_name,
                'NAME_ZH': new_name,
                'SOURCE': 'modelscope',
                'DESCRIPTION': '',
                'BASE_MODEL': base_model,
                'MODEL_PATH': tar_path,
                'IMAGE_PATH': f'{tar_path}/image.jpg',
                'TUNER_TYPE': tuner_type,
                'PROMPT_EXAMPLE': tuner_prompt_example
            }
            pipeline_level_modules = manager.inference.model_manage_ui.pipe_manager.pipeline_level_modules
            if new_tuner['BASE_MODEL'] not in pipeline_level_modules:
                gr.Error(self.component_names.model_err3 +
                         new_tuner['BASE_MODEL'])
            pipeline_ins = pipeline_level_modules[new_tuner['BASE_MODEL']]
            now_diffusion_model = f"{new_tuner['BASE_MODEL']}_{pipeline_ins.diffusion_model['name']}"

            custom_tuner_choices = self.add_tuner(new_tuner, manager,
                                                  now_diffusion_model)

            update_items = {
                'MODELSCOPE_URL':
                f'https://www.modelscope.cn/models/{username}/{modelid}'
            }
            self.update_tuner_info(tuner_category,
                                   new_name,
                                   update_items=update_items)

            return (gr.update(choices=list(self.saved_tuners_category.keys()),
                              value=tuner_category),
                    gr.update(choices=list(
                        self.saved_tuners_category.get(tuner_category,
                                                       {}).keys()),
                              value=new_name), gr.Text(value=new_name),
                    gr.Dropdown(choices=custom_tuner_choices))

        self.ms_download_submit.click(
            fn=pull_from_modelscope,
            inputs=[
                self.ms_modelid,
                self.ms_download_username,
            ],
            outputs=[
                self.diffusion_models, self.tuner_models, info_ui.tuner_name,
                manager.inference.tuner_ui.custom_tuner_model
            ],
            queue=True)

        def change_tuner_type_by_model_version(base_model_revision):
            for base_model_tuner in self.base_model_tuner_methods:
                if base_model_tuner.BASE_MODEL == base_model_revision:
                    return gr.Dropdown(value=base_model_tuner.TUNER_TYPE[0],
                                       choices=base_model_tuner.TUNER_TYPE,
                                       interactive=True)
            return gr.Dropdown(value='', choices=[], interactive=True)

        self.upload_base_models.change(fn=change_tuner_type_by_model_version,
                                       inputs=[self.upload_base_models],
                                       outputs=[self.upload_tuner_type],
                                       queue=False)

        def upload_zip(file_path, tuner_name, base_model, tuner_type):
            sub_dir = f'{base_model}-{tuner_type}'
            save_file = os.path.join(self.work_dir, sub_dir,
                                     f'{tuner_name}.zip')
            model_dir = os.path.join(self.work_dir, sub_dir, tuner_name)
            with FS.put_to(save_file) as local_zip:
                res = os.popen(f"cp '{file_path.name}' '{local_zip}'")
                res = res.readlines()
            with FS.get_from(save_file) as local_path:
                res = os.popen(f"unzip -o '{local_path}' -d '{model_dir}'")
                res = res.readlines()
            if not os.path.exists(model_dir):
                raise gr.Error(f'解压{save_file}失败{str(res)}')
            # find meta.yaml
            if os.path.exists(os.path.join(model_dir, 'meta.yaml')):
                meta = Config(cfg_file=os.path.join(model_dir, 'meta.yaml'))
                tuner_desc = meta.get('DESCRIPTION', '')
                tuner_example = meta.get('IMAGE_PATH', None)
                if tuner_example is not None:
                    tuner_example = os.path.join(
                        model_dir, os.path.basename(tuner_example))
                    if os.path.exists(tuner_example):
                        from PIL import Image
                        tuner_example_path = os.path.join(
                            model_dir, 'image.jpg')
                        if not os.path.exists(tuner_example_path):
                            tuner_example = Image.open(tuner_example)
                            with FS.put_to(tuner_example_path) as local_path:
                                tuner_example.save(local_path)
                        tuner_example = tuner_example_path
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
                'DESCRIPTION': tuner_desc,
                'BASE_MODEL': base_model,
                'MODEL_PATH': model_dir,
                'IMAGE_PATH': tuner_example,
                'TUNER_TYPE': tuner_type,
                'PROMPT_EXAMPLE': tuner_prompt_example
            }
            pipeline_level_modules = manager.inference.model_manage_ui.pipe_manager.pipeline_level_modules
            if new_tuner['BASE_MODEL'] not in pipeline_level_modules:
                gr.Error(self.component_names.model_err3 +
                         new_tuner['BASE_MODEL'])
            pipeline_ins = pipeline_level_modules[new_tuner['BASE_MODEL']]
            now_diffusion_model = f"{new_tuner['BASE_MODEL']}_{pipeline_ins.diffusion_model['name']}"

            custom_tuner_choices = self.add_tuner(new_tuner, manager,
                                                  now_diffusion_model)

            return (gr.Dropdown(choices=list(
                self.saved_tuners_category.keys()),
                                value=sub_dir),
                    gr.Dropdown(choices=list(
                        self.saved_tuners_category.get(sub_dir, {}).keys()),
                                value=tuner_name), gr.Text(value=tuner_name),
                    gr.Text(value=tuner_type), gr.Text(value=base_model),
                    gr.Text(value=tuner_desc),
                    gr.Text(value=tuner_prompt_example),
                    gr.Image(value=tuner_example),
                    gr.Dropdown(choices=custom_tuner_choices))

        self.local_upload_bt.click(
            upload_zip,
            inputs=[
                self.file_path, self.upload_tuner_name,
                self.upload_base_models, self.upload_tuner_type
            ],
            outputs=[
                self.diffusion_models, self.tuner_models, info_ui.tuner_name,
                info_ui.tuner_type, info_ui.base_model, info_ui.tuner_desc,
                info_ui.tuner_prompt_example, info_ui.tuner_example,
                manager.inference.tuner_ui.custom_tuner_model
            ],
            queue=False)
