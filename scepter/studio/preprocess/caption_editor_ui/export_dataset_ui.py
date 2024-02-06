# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from __future__ import annotations

import os
import urllib.parse as parse

import gradio as gr

from scepter.modules.utils.file_system import FS
from scepter.studio.preprocess.caption_editor_ui.component_names import \
    ExportDatasetUIName
from scepter.studio.utils.uibase import UIBase


class ExportDatasetUI(UIBase):
    def __init__(self, cfg, is_debug=False, language='en'):
        self.dataset_name = ''
        self.work_dir = cfg.WORK_DIR
        self.export_folder = os.path.join(self.work_dir, cfg.EXPORT_DIR)
        self.component_names = ExportDatasetUIName(language)

    def create_ui(self):
        with gr.Row(variant='panel', visible=False,
                    equal_height=True) as export_panel:
            self.data_state = gr.State(value=False)
            with gr.Column(scale=1, min_width=0):
                self.export_to_zip = gr.Button(
                    value=self.component_names.btn_export_zip)
                self.export_url = gr.File(
                    label=self.component_names.export_file,
                    visible=False,
                    value=None,
                    interactive=False,
                    show_label=True)
            with gr.Column(scale=1, min_width=0):
                self.go_to_train = gr.Button(
                    value=self.component_names.go_to_train, size='lg')
        self.export_panel = export_panel

    def set_callbacks(self, create_dataset, manager):
        def export_zip(dataset_name):
            meta = create_dataset.meta_dict[dataset_name]
            work_dir = meta['work_dir']
            local_work_dir = meta['local_work_dir']
            train_csv = os.path.join(work_dir, 'train.csv')
            if len(meta['file_list']) < 1:
                raise gr.Error(self.component_names.export_err1)
            train_csv = create_dataset.write_csv(meta['file_list'], train_csv,
                                                 work_dir)
            _ = FS.get_from(train_csv, os.path.join(local_work_dir,
                                                    'train.csv'))
            save_file_list = work_dir + '_file.csv'
            save_file_list = create_dataset.write_file_list(
                meta['file_list'], save_file_list)
            _ = FS.get_from(save_file_list,
                            os.path.join(local_work_dir, 'file.csv'))
            zip_path = os.path.join(self.export_folder, f'{dataset_name}.zip')
            with FS.put_to(zip_path) as local_zip:
                res = os.popen(
                    f"cd '{local_work_dir}' && mkdir -p '{dataset_name}' "
                    f"&& cp -rf images '{dataset_name}/images' "
                    f"&& cp -rf train.csv '{dataset_name}/train.csv' "
                    f"&& zip -r '{os.path.abspath(local_zip)}' '{dataset_name}'/* "
                    f"&& rm -rf '{dataset_name}'")
                print(res.readlines())

            if not FS.exists(zip_path):
                raise gr.Error(self.component_names.export_zip_err1)
            create_dataset.save_meta(meta, work_dir)
            local_zip = FS.get_from(zip_path)
            return gr.File(value=local_zip, visible=True)

        self.export_to_zip.click(export_zip,
                                 inputs=[create_dataset.user_data_name],
                                 outputs=[self.export_url],
                                 queue=False)

        def export_csv(dataset_name):
            meta = create_dataset.meta_dict[dataset_name]
            work_dir = meta['work_dir']
            local_work_dir = meta['local_work_dir']
            train_csv = os.path.join(work_dir, 'train.csv')
            if len(meta['file_list']) < 1:
                raise gr.Error(self.component_names.export_err1)
            train_csv = create_dataset.write_csv(meta['file_list'], train_csv,
                                                 work_dir)
            _ = FS.get_from(train_csv, os.path.join(local_work_dir,
                                                    'train.csv'))
            save_file_list = os.path.join(work_dir, 'file.csv')
            save_file_list = create_dataset.write_file_list(
                meta['file_list'], save_file_list)
            local_file_csv = FS.get_from(
                save_file_list, os.path.join(local_work_dir, 'file.csv'))
            create_dataset.save_meta(meta, work_dir)
            is_flag = FS.put_object_from_local_file(
                local_file_csv,
                os.path.join(self.export_folder, dataset_name + '_file.csv'))
            if not is_flag:
                raise gr.Error(self.component_names.upload_err1)
            list_url = FS.get_url(os.path.join(self.export_folder,
                                               dataset_name + '_file.csv'),
                                  set_public=True)
            list_url = parse.unquote(list_url)
            if 'wulanchabu' in list_url:
                list_url = list_url.replace(
                    '.cn-wulanchabu.oss-internal.aliyun-inc.',
                    '.oss-cn-wulanchabu.aliyuncs.')
            else:
                list_url = list_url.replace('.oss-internal.aliyun-inc.',
                                            '.oss.aliyuncs.')
            if not list_url.split('/')[-1] == dataset_name + '_file.csv':
                list_url = os.path.join(os.path.dirname(list_url),
                                        dataset_name + '_file.csv')
            return gr.Text(value=list_url)

        # self.export_to_list.click(export_csv,
        #                           inputs=[create_dataset.user_data_name],
        #                           outputs=[self.export_url])

        def go_to_train(dataset_name):
            meta = create_dataset.meta_dict[dataset_name]
            work_dir = meta['work_dir']
            local_work_dir = meta['local_work_dir']
            train_csv = os.path.join(work_dir, 'train.csv')
            if len(meta['file_list']) < 1:
                raise gr.Error(self.component_names.export_err1)
            train_csv = create_dataset.write_csv(meta['file_list'], train_csv,
                                                 work_dir)
            _ = FS.get_from(train_csv, os.path.join(local_work_dir,
                                                    'train.csv'))
            save_file_list = work_dir + '_file.csv'
            _ = create_dataset.write_file_list(meta['file_list'],
                                               save_file_list)
            return (gr.Tabs(selected='self_train'),
                    gr.Textbox(value=os.path.abspath(local_work_dir)))

        self.go_to_train.click(
            go_to_train,
            inputs=[create_dataset.user_data_name],
            outputs=[manager.tabs, manager.self_train.trainer_ui.ms_data_name],
            queue=False)
