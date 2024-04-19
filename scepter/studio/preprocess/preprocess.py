# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path

import gradio as gr

from scepter.modules.utils.config import Config
from scepter.modules.utils.file_system import FS
from scepter.studio.preprocess.caption_editor_ui.create_dataset_ui import \
    CreateDatasetUI
from scepter.studio.preprocess.caption_editor_ui.dataset_gallery_ui import \
    DatasetGalleryUI
from scepter.studio.preprocess.caption_editor_ui.export_dataset_ui import \
    ExportDatasetUI
from scepter.studio.utils.env import init_env


class PreprocessUI():
    def __init__(self,
                 cfg_general_file,
                 is_debug=False,
                 language='en',
                 root_work_dir='./'):
        cfg_general = Config(cfg_file=cfg_general_file)

        cfg_general.WORK_DIR = os.path.join(root_work_dir,
                                            cfg_general.WORK_DIR)
        if not FS.exists(cfg_general.WORK_DIR):
            FS.make_dir(cfg_general.WORK_DIR)

        cfg_general = init_env(cfg_general)
        self.create_dataset = CreateDatasetUI.get_instance(cfg_general,
                                                           is_debug=is_debug,
                                                           language=language)
        self.dataset_gallery = DatasetGalleryUI.get_instance(
            cfg_general,
            is_debug=is_debug,
            language=language,
            create_ins=self.create_dataset)
        self.export_dataset = ExportDatasetUI.get_instance(
            cfg_general,
            is_debug=is_debug,
            language=language,
            gallery_ins=self.dataset_gallery)

    def create_ui(self):
        self.create_dataset.create_ui()
        self.export_dataset.create_ui()
        self.dataset_gallery.create_ui()

    def set_callbacks(self, manager):
        self.create_dataset.set_callbacks(self.dataset_gallery,
                                          self.export_dataset, manager)
        self.dataset_gallery.set_callbacks(self.create_dataset, manager)
        self.export_dataset.set_callbacks(self.create_dataset, manager)


if __name__ == '__main__':
    pre_ui = PreprocessUI('scepter/methods/studio/preprocess/preprocess.yaml',
                          root_work_dir='./cache')
    with gr.Blocks() as demo:
        gr.Markdown('<h2><center>SCEPTER Preprocess</center><h2>')
        with gr.Tabs(elem_id='tabs') as tabs:
            with gr.TabItem('editor', id=1, elem_id=f'tab_{1}'):
                pre_ui.create_ui()
        pre_ui.set_callbacks(None)
    demo.queue(status_update_rate=1).launch(show_error=True,
                                            debug=True,
                                            enable_queue=True)
