# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from __future__ import annotations

import datetime
import os.path
from collections import OrderedDict

import gradio as gr

from scepter.modules.utils.file_system import FS
from scepter.studio.preprocess.caption_editor_ui.component_names import \
    CreateDatasetUIName
from scepter.studio.preprocess.utils.data_card import Text2ImageDataCard
from scepter.studio.utils.uibase import UIBase

refresh_symbol = '\U0001f504'  # 🔄


def wget_file(file_url, save_file):
    if 'oss' in file_url:
        file_url = file_url.split('?')[0]
    local_path, _ = FS.map_to_local(save_file)
    res = os.popen(f"wget -c '{file_url}' -O '{local_path}'")
    res.readlines()
    FS.put_object_from_local_file(local_path, save_file)
    return save_file, res


class CreateDatasetUI(UIBase):
    def __init__(self, cfg, is_debug=False, language='en'):
        self.work_dir = cfg.WORK_DIR
        self.language = language
        self.dataset_type_dict = OrderedDict(
            {'scepter_txt2img': Text2ImageDataCard})
        self.components_name = CreateDatasetUIName(language)
        self.default_dataset_type = list(self.dataset_type_dict.keys())[0]
        self.default_dataset_cls = Text2ImageDataCard

        self.cache_file = {}
        '''
            {
                "dataset_type": {
                    "dataset_name" : {}
                }
            }
        '''
        self.dataset_dict = {}
        '''
            {
                "user_name": {
                    "dataset_type" : []
                }
            }
        '''
        self.init_example()
        self.user_level_dataset_list = self.load_history()

    def init_example(self):
        file_url = self.components_name.default_dataset_zip
        file_name, surfix = os.path.splitext(file_url)
        save_file = os.path.join(
            self.work_dir,
            f'{self.components_name.default_dataset_name}{surfix}')
        save_file, res = wget_file(file_url, save_file)
        if not FS.exists(save_file):
            print('Init example eroor, skip!')
            self.example_dataset_ins = None
            return

        dataset_folder = os.path.join(
            self.work_dir, '_'.join([
                self.default_dataset_type,
                self.components_name.default_dataset_name
            ]))
        self.example_dataset_ins = self.default_dataset_cls(
            dataset_folder,
            dataset_name=self.components_name.default_dataset_name,
            src_file=save_file,
            surfix=surfix,
            user_name='example',
            language=self.language)
        # add to list by load_history.

    def user_filter(self, dataset_ins, login_user_name):
        if not dataset_ins.is_valid:
            return False
        if (dataset_ins.dataset_name.startswith(login_user_name)
                or dataset_ins.user_name == login_user_name
                or login_user_name == 'admin'
                or dataset_ins.user_name == 'example'):
            return True
        return False

    def load_history(self, login_user_name='admin'):
        dataset_list = {login_user_name: {}}
        self.dir_list = FS.walk_dir(self.work_dir, recurse=False)
        # From v0.0.5 we use the classname of DataCard as prefix of folder to indicate the type of data.
        for one_dir in self.dir_list:
            if not FS.isdir(one_dir):
                continue
            # the meta.json is the unique sign for dataset
            meta_file = os.path.join(one_dir, 'meta.json')
            if not FS.exists(meta_file):
                continue
            dataset_type = self.default_dataset_type
            dataset_cls = self.default_dataset_cls
            for key, value in self.dataset_type_dict.items():
                if one_dir.split('/')[-1].startswith(key):
                    dataset_type = key
                    dataset_cls = value
                    break
            dataset_ins = dataset_cls(dataset_folder=one_dir)
            if dataset_type not in self.dataset_dict:
                self.dataset_dict[dataset_type] = {}
            if dataset_type not in dataset_list[login_user_name]:
                dataset_list[login_user_name][dataset_type] = []
            if self.user_filter(dataset_ins, login_user_name):
                dataset_list[login_user_name][dataset_type].append(
                    dataset_ins.dataset_name)
            self.dataset_dict[dataset_type][
                dataset_ins.dataset_name] = dataset_ins
        return dataset_list

    def create_ui(self):
        with gr.Box():
            gr.Markdown(self.components_name.user_direction)
        with gr.Box():
            with gr.Row():
                self.sys_log = gr.Markdown(
                    self.components_name.system_log.format(''))
            with gr.Row(variant='panel', ):
                with gr.Column(scale=2, min_width=0):
                    self.dataset_type = gr.Dropdown(
                        label=self.components_name.dataset_type,
                        choices=[
                            self.components_name.dataset_type_name[data_type]
                            for data_type in self.dataset_type_dict.keys()
                        ],
                        interactive=True,
                        value=self.components_name.dataset_type_name[
                            self.default_dataset_type])
                with gr.Column(scale=3, min_width=0):
                    self.dataset_name = gr.Dropdown(
                        label=self.components_name.dataset_name,
                        choices=self.user_level_dataset_list.get(
                            'admin', {}).get(self.default_dataset_type, []),
                        value=None if self.example_dataset_ins is None else
                        self.example_dataset_ins.dataset_name,
                        interactive=True)
                with gr.Column(scale=3, min_width=0):
                    self.user_dataset_name = gr.Text(
                        label=self.components_name.user_data_name,
                        value='' if self.example_dataset_ins is None else
                        self.example_dataset_ins.dataset_name,
                        interactive=True)
                    self.user_data_name_state = gr.State(
                        value=None if self.example_dataset_ins is None else
                        self.example_dataset_ins.dataset_name)
                    self.create_mode = gr.State(value=0)
                with gr.Column(scale=1, min_width=0):
                    with gr.Column(scale=1, min_width=0):
                        self.refresh_dataset_name = gr.Button(
                            value=self.components_name.refresh_list_button)
                    with gr.Column(scale=1, min_width=0):
                        self.modify_data_button = gr.Button(
                            value=self.components_name.modify_data_button)
                with gr.Column(scale=1, min_width=0):
                    with gr.Column(scale=1, min_width=0):
                        self.btn_create_datasets = gr.Button(
                            value=self.components_name.btn_create_datasets)
                    with gr.Column(scale=1, min_width=0):
                        self.btn_delete_datasets = gr.Button(
                            value=self.components_name.delete_dataset_button)
                self.panel_state = gr.Checkbox(label='panel_state',
                                               value=False,
                                               visible=False)
            with gr.Row(variant='panel', ):
                with gr.Column(scale=4, visible=False,
                               min_width=0) as dataname_panel:
                    self.new_dataset_name = gr.Text(
                        label=self.components_name.new_data_name,
                        value='',
                        interactive=True)
                    with gr.Row(visible=False, ) as file_panel:
                        with gr.Column(scale=1, min_width=0):
                            self.use_link = gr.Checkbox(
                                label=self.components_name.use_link,
                                value=False,
                                visible=False)
                        with gr.Column(scale=2, min_width=0):
                            self.file_path = gr.File(
                                label=self.components_name.zip_file,
                                min_width=0,
                                file_types=['.zip', '.txt', '.csv'],
                                visible=False)
                            self.file_path_url = gr.Text(
                                label=self.components_name.zip_file_url,
                                value='',
                                placeholder=self.components_name.
                                default_dataset_zip,
                                visible=False)
                with gr.Column(scale=1, visible=False,
                               min_width=0) as btn_panel:
                    self.random_data_button = gr.Button(
                        value=self.components_name.get_data_name_button)
                    self.confirm_data_button = gr.Button(
                        value=self.components_name.confirm_data_button)
                    self.cancel_data_button = gr.Button(
                        value=self.components_name.cancel_create_button)
                    self.btn_create_datasets_from_file = gr.Checkbox(
                        label=self.components_name.
                        btn_create_datasets_from_file,
                        value=False)
        self.btn_panel = btn_panel
        self.file_panel = file_panel
        self.dataname_panel = dataname_panel

    def get_trans_dataset_type(self, dataset_type):
        reverse_data_type = {
            v: k
            for k, v in self.components_name.dataset_type_name.items()
        }
        trans_dataset_type = reverse_data_type[dataset_type]
        return trans_dataset_type

    def set_callbacks(self, gallery_dataset, export_dataset, manager):
        def get_random_dataset_name():
            data_name = 'name-version-{0:%Y%m%d_%H_%M_%S}'.format(
                datetime.datetime.now())
            return data_name

        def clear_file():
            return gr.Text(visible=False)

        def show_dataset_panel():
            return gr.Checkbox(
                value=True), self.components_name.system_log.format('')

        # Click Create
        self.btn_create_datasets.click(show_dataset_panel, [],
                                       [self.panel_state, self.sys_log],
                                       queue=False)

        def unshow_dataset_panel():
            return gr.Checkbox(
                value=False), self.components_name.system_log.format('')

        self.cancel_data_button.click(unshow_dataset_panel, [],
                                      [self.panel_state, self.sys_log],
                                      queue=False)

        def delete_dataset(dataset_name, dataset_type, login_user_name):
            dataset_type = self.get_trans_dataset_type(dataset_type)
            if dataset_name in self.dataset_dict[dataset_type]:
                dataset_ins = self.dataset_dict[dataset_type][dataset_name]
                if dataset_ins.user_name == 'example':
                    sys_log = self.components_name.system_log.format(
                        self.components_name.delete_data_err1)
                    return (dataset_name, gr.Dropdown(value=dataset_name),
                            sys_log)
                dataset_ins.deactive_dataset()
            dataset_list = self.user_level_dataset_list.get(
                login_user_name, {}).get(dataset_type, [])
            if dataset_name in dataset_list:
                dataset_list.remove(dataset_name)
            if len(dataset_list) > 0:
                now_dataset = dataset_list[-1]
            else:
                now_dataset = ''
            return (now_dataset,
                    gr.Dropdown(choices=dataset_list, value=now_dataset),
                    self.components_name.system_log.format(''))

        self.btn_delete_datasets.click(
            delete_dataset,
            [self.dataset_name, self.dataset_type, manager.user_name],
            [self.user_dataset_name, self.dataset_name, self.sys_log],
            queue=False)

        def show_file_panel(show_file_panel):
            if show_file_panel:
                return (gr.Row(visible=True), gr.File(value=None,
                                                      visible=True),
                        gr.Text(value='', visible=False), 2,
                        gr.Checkbox(value=False, visible=True),
                        self.components_name.system_log.format(''))
            else:
                return (gr.Row(visible=False),
                        gr.File(value=None,
                                visible=False), gr.Text(value='',
                                                        visible=False), 1,
                        gr.Checkbox(value=False, visible=False),
                        self.components_name.system_log.format(''))

        self.btn_create_datasets_from_file.change(
            show_file_panel, [self.btn_create_datasets_from_file], [
                self.file_panel, self.file_path, self.file_path_url,
                self.create_mode, self.use_link, self.sys_log
            ],
            queue=False)

        def use_link_change(use_link):
            if use_link:
                create_mode = 3
                return (gr.File(value=None, visible=False),
                        gr.Text(value='', visible=True), create_mode,
                        self.components_name.system_log.format(''))
            else:
                create_mode = 2
                return (gr.File(value=None, visible=True),
                        gr.Text(value='', visible=False), create_mode,
                        self.components_name.system_log.format(''))

        self.use_link.change(use_link_change, [self.use_link], [
            self.file_path, self.file_path_url, self.create_mode, self.sys_log
        ])
        # Click Refresh
        self.random_data_button.click(get_random_dataset_name, [],
                                      [self.new_dataset_name],
                                      queue=False)

        self.file_path.clear(clear_file,
                             outputs=[self.file_path_url],
                             queue=False)

        def confirm_create_dataset(user_dataset_name, create_mode, file_url,
                                   file_path, login_user_name, dataset_type):
            if user_dataset_name.strip(
            ) == '' or ' ' in user_dataset_name or '/' in user_dataset_name:
                sys_log = self.components_name.system_log.format(
                    self.components_name.illegal_data_name_err1)
                return (gr.Checkbox(), gr.Dropdown(), gr.Text(), sys_log)

            if len(user_dataset_name.split('-')) < 3:
                sys_log = self.components_name.system_log.format(
                    self.components_name.illegal_data_name_err2)
                return (gr.Checkbox(), gr.Dropdown(), gr.Text(), sys_log)
            if '.' in user_dataset_name:
                sys_log = self.components_name.system_log.format(
                    self.components_name.illegal_data_name_err3)
                return (gr.Checkbox(), gr.Dropdown(), gr.Text(), sys_log)
            if not file_url.strip() == '' and file_path is not None:
                sys_log = self.components_name.system_log.format(
                    self.components_name.illegal_data_name_err4)
                return (gr.Checkbox(), gr.Dropdown(), gr.Text(), sys_log)
            dataset_type = self.get_trans_dataset_type(dataset_type)
            if create_mode == 3 and not file_url.strip() == '':
                file_name, surfix = os.path.splitext(file_url)
                save_file = os.path.join(self.work_dir,
                                         f'{user_dataset_name}{surfix}')
                save_file, res = wget_file(file_url, save_file)
                if not FS.exists(save_file):
                    sys_log = self.components_name.system_log.format(
                        f'{self.components_name.illegal_data_err1} {str(res)}')
                    return (gr.Checkbox(), gr.Dropdown(), gr.Text(), sys_log)
            elif create_mode == 2 and file_path is not None and file_path.name:
                self.cache_file[user_dataset_name] = {
                    'file_name': file_path.name,
                    'surfix': os.path.splitext(file_path.name)[-1]
                }
                cache_file = self.cache_file.pop(user_dataset_name)
                surfix = cache_file['surfix']
                ori_file = cache_file['file_name']
                save_file = os.path.join(self.work_dir,
                                         f'{user_dataset_name}{surfix}')
                with FS.put_to(save_file) as local_path:
                    res = os.popen(f"cp '{ori_file}' '{local_path}'")
                    res = res.readlines()
                if not FS.exists(save_file):
                    sys_log = self.components_name.system_log.format(
                        f'{self.components_name.illegal_data_err1}{str(res)}')
                    return (gr.Checkbox(), gr.Dropdown(), gr.Text(), sys_log)
            else:
                surfix = None
                save_file = None
            # untar file or create blank dataset
            dataset_folder = os.path.join(
                self.work_dir, '_'.join([dataset_type, user_dataset_name]))
            dataset_cls = self.dataset_type_dict[dataset_type]
            dataset_ins = dataset_cls(dataset_folder,
                                      dataset_name=user_dataset_name,
                                      src_file=save_file,
                                      surfix=surfix,
                                      user_name=login_user_name,
                                      language=self.language)
            if login_user_name not in self.user_level_dataset_list:
                self.user_level_dataset_list[login_user_name] = {}
            if dataset_type not in self.user_level_dataset_list[
                    login_user_name]:
                self.user_level_dataset_list[login_user_name][
                    dataset_type] = []
            if dataset_ins.dataset_name not in self.user_level_dataset_list[
                    login_user_name][dataset_type]:
                self.user_level_dataset_list[login_user_name][
                    dataset_type].append(dataset_ins.dataset_name)
            if dataset_type not in self.dataset_dict:
                self.dataset_dict[dataset_type] = {}
            self.dataset_dict[dataset_type][
                dataset_ins.dataset_name] = dataset_ins
            return (gr.Checkbox(value=False, visible=False),
                    gr.Dropdown(value=dataset_ins.dataset_name,
                                choices=self.user_level_dataset_list.get(
                                    login_user_name, {}).get(dataset_type,
                                                             [])),
                    gr.Text(value=dataset_ins.dataset_name),
                    self.components_name.system_log.format(''))

        # Click Confirm
        self.confirm_data_button.click(confirm_create_dataset, [
            self.new_dataset_name, self.create_mode, self.file_path_url,
            self.file_path, manager.user_name, self.dataset_type
        ], [
            self.panel_state, self.dataset_name, self.user_dataset_name,
            self.sys_log
        ],
                                       queue=False)

        def show_edit_panel(panel_state, data_name):
            if panel_state:
                return (gr.Row(visible=False), gr.Column(visible=True),
                        gr.Column(visible=True),
                        gr.Text(value=get_random_dataset_name()), 1,
                        gr.Checkbox(value=False), data_name,
                        self.components_name.system_log.format(''))
            else:
                return (gr.Row(visible=False), gr.Column(visible=False),
                        gr.Column(visible=False), gr.Text(), 0,
                        gr.Checkbox(value=False), data_name,
                        self.components_name.system_log.format(''))

        self.panel_state.change(
            show_edit_panel, [self.panel_state, self.dataset_name], [
                self.file_panel, self.btn_panel, self.dataname_panel,
                self.new_dataset_name, self.create_mode,
                self.btn_create_datasets_from_file, self.user_data_name_state,
                self.sys_log
            ],
            queue=False)

        def modify_data_name(user_dataset_name, prev_data_name,
                             login_user_name, dataset_type):

            if user_dataset_name.strip(
            ) == '' or ' ' in user_dataset_name or '/' in user_dataset_name:
                sys_log = self.components_name.system_log.format(
                    self.components_name.illegal_data_name_err1)
                return prev_data_name, prev_data_name, gr.Dropdown(), sys_log
            if len(user_dataset_name.split('-')) < 3:
                sys_log = self.components_name.system_log.format(
                    self.components_name.illegal_data_name_err2)
                return prev_data_name, prev_data_name, gr.Dropdown(), sys_log
            if '.' in user_dataset_name:
                sys_log = self.components_name.system_log.format(
                    self.components_name.illegal_data_name_err3)
                return prev_data_name, prev_data_name, gr.Dropdown(), sys_log
            print(
                f'Current file name {prev_data_name}, new file name {user_dataset_name}.'
            )
            dataset_type = self.get_trans_dataset_type(dataset_type)
            if user_dataset_name != prev_data_name:
                if prev_data_name in self.dataset_dict[dataset_type]:
                    ori_dataset_ins = self.dataset_dict[dataset_type][
                        prev_data_name]
                    if ori_dataset_ins.user_name == 'example':
                        sys_log = self.components_name.system_log.format(
                            self.components_name.modify_data_err1)
                        return prev_data_name, prev_data_name, gr.Dropdown(
                        ), sys_log
                    ori_dataset_ins.modify_data_name(user_dataset_name)
                    user_level_dataset_list = self.load_history(
                        login_user_name)
                    self.user_level_dataset_list.update(
                        user_level_dataset_list)
                    dataset_list = self.user_level_dataset_list.get(
                        login_user_name, {}).get(dataset_type, [])
                    if prev_data_name in dataset_list:
                        dataset_list.remove(prev_data_name)
                        dataset_list.append(user_dataset_name)
                    self.user_level_dataset_list[
                        login_user_name] = dataset_list
                    self.dataset_dict[dataset_type].pop(prev_data_name)
                    self.dataset_dict[dataset_type][
                        user_dataset_name] = ori_dataset_ins
                else:
                    sys_log = self.components_name.system_log.format(
                        self.components_name.modify_data_name_err1)
                    return prev_data_name, prev_data_name, gr.Dropdown(
                    ), sys_log
                return user_dataset_name, user_dataset_name, gr.Dropdown(
                    choices=dataset_list, value=user_dataset_name
                ), self.components_name.system_log.format('')
            else:
                return user_dataset_name, user_dataset_name, gr.Dropdown(
                ), self.components_name.system_log.format('')

        self.modify_data_button.click(modify_data_name,
                                      inputs=[
                                          self.user_dataset_name,
                                          self.user_data_name_state,
                                          manager.user_name, self.dataset_type
                                      ],
                                      outputs=[
                                          self.user_dataset_name,
                                          self.user_data_name_state,
                                          self.dataset_name, self.sys_log
                                      ],
                                      queue=False)

        def dataset_change(dataset_name, dataset_type):
            trans_dataset_type = self.get_trans_dataset_type(dataset_type)
            if dataset_name is None or dataset_name == '':
                sys_log = (self.components_name.system_log.format(
                    self.components_name.illegal_data_name_err5 +
                    f'{dataset_name}'))
                return (gr.Row(), gr.Text(), dataset_name, gr.Text(), sys_log)
            if trans_dataset_type not in self.dataset_dict:
                self.dataset_dict[trans_dataset_type] = {}
            if dataset_name not in self.dataset_dict[trans_dataset_type]:
                sys_log = (self.components_name.system_log.format(
                    self.components_name.refresh_data_list_info1))
                return (gr.Row(), gr.Text(), dataset_name, gr.Text(), sys_log)
            return (gr.Row(visible=False),
                    gr.Text(value=dataset_name,
                            interactive=True), dataset_name,
                    gr.Text(value=dataset_name),
                    self.components_name.system_log.format(''))

        self.dataset_name.change(dataset_change,
                                 inputs=[self.dataset_name, self.dataset_type],
                                 outputs=[
                                     self.file_panel, self.user_dataset_name,
                                     self.user_data_name_state,
                                     gallery_dataset.gallery_state,
                                     self.sys_log
                                 ],
                                 queue=False)

        def dataset_type_change(dataset_type, login_user_name):
            trans_dataset_type = self.get_trans_dataset_type(dataset_type)
            user_level_dataset_list = self.load_history(
                login_user_name=login_user_name)
            self.user_level_dataset_list.update(user_level_dataset_list)
            dataset_list = user_level_dataset_list[login_user_name].get(
                trans_dataset_type, [])
            return gr.Dropdown(
                value=dataset_list[-1] if len(dataset_list) > 0 else '',
                choices=dataset_list), self.components_name.system_log.format(
                    '')

        manager.user_name.change(dataset_type_change,
                                 inputs=[self.dataset_type, manager.user_name],
                                 outputs=[self.dataset_name, self.sys_log],
                                 queue=False)

        self.dataset_type.change(dataset_type_change,
                                 inputs=[self.dataset_type, manager.user_name],
                                 outputs=[self.dataset_name, self.sys_log],
                                 queue=False)

        self.refresh_dataset_name.click(
            dataset_type_change,
            inputs=[self.dataset_type, manager.user_name],
            outputs=[self.dataset_name, self.sys_log],
            queue=False)
