# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from __future__ import annotations

import os

import gradio as gr

from scepter.studio.preprocess.caption_editor_ui.component_names import \
    ExportDatasetUIName
from scepter.studio.utils.uibase import UIBase


class ExportDatasetUI(UIBase):
    def __init__(self, cfg, is_debug=False, language='en', gallery_ins=None):
        self.dataset_name = ''
        self.work_dir = cfg.WORK_DIR
        self.export_folder = os.path.join(self.work_dir, cfg.EXPORT_DIR)
        self.component_names = ExportDatasetUIName(language)
        if gallery_ins is not None:
            self.default_dataset = gallery_ins.default_dataset
        else:
            self.default_dataset = None

    def create_ui(self):
        with gr.Row(variant='panel',
                    visible=self.default_dataset is not None,
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
        def export_zip(dataset_type, dataset_name):
            dataset_type = create_dataset.get_trans_dataset_type(dataset_type)
            dataset_ins = create_dataset.dataset_dict[dataset_type][
                dataset_name]
            local_zip = dataset_ins.export_zip(self.export_folder)
            return gr.File(value=local_zip, visible=True)

        self.export_to_zip.click(
            export_zip,
            inputs=[create_dataset.dataset_type, create_dataset.dataset_name],
            outputs=[self.export_url],
            queue=False)

        def go_to_train(dataset_type, dataset_name):
            dataset_type = create_dataset.get_trans_dataset_type(dataset_type)
            dataset_ins = create_dataset.dataset_dict[dataset_type][
                dataset_name]
            dataset_ins.update_dataset()
            return (gr.Tabs(selected='self_train'),
                    gr.Textbox(
                        value=os.path.abspath(dataset_ins.local_work_dir)),
                    dataset_name)

        self.go_to_train.click(
            go_to_train,
            inputs=[create_dataset.dataset_type, create_dataset.dataset_name],
            outputs=[
                manager.tabs, manager.self_train.trainer_ui.ms_data_name,
                manager.self_train.trainer_ui.ori_data_name
            ],
            queue=False)
