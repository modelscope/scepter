# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import os

import gradio as gr

import scepter
from scepter.modules.utils.config import Config
from scepter.modules.utils.file_system import FS
from scepter.studio.self_train.self_train_ui.model_ui import ModelUI
from scepter.studio.self_train.self_train_ui.trainer_ui import TrainerUI
from scepter.studio.self_train.utils.config_parser import get_all_config
from scepter.studio.utils.env import init_env


class SelfTrainUI():
    def __init__(self,
                 cfg_general_file,
                 is_debug=False,
                 language='en',
                 root_work_dir='./'):
        cfg_general = Config(cfg_file=cfg_general_file)

        BASE_CFG_VALUE = get_all_config(os.path.dirname(cfg_general_file),
                                        global_meta=cfg_general)

        cfg_general.WORK_DIR = os.path.join(root_work_dir,
                                            cfg_general.WORK_DIR)
        if not FS.exists(cfg_general.WORK_DIR):
            FS.make_dir(cfg_general.WORK_DIR)
        cfg_general = init_env(cfg_general)

        self.trainer_ui = TrainerUI(cfg_general,
                                    BASE_CFG_VALUE,
                                    is_debug=is_debug,
                                    language=language)
        self.model_ui = ModelUI(cfg_general,
                                BASE_CFG_VALUE,
                                is_debug=is_debug,
                                language=language)

    def create_ui(self):
        with gr.Row():
            self.trainer_ui.create_ui()
        with gr.Row():
            self.model_ui.create_ui()

    def set_callbacks(self, manager):
        self.trainer_ui.set_callbacks(self.model_ui, manager)
        self.model_ui.set_callbacks(self.trainer_ui, manager)


if __name__ == '__main__':
    st_ins = SelfTrainUI(os.path.join(
        scepter.dirname, 'scepter/methods/studio/self_train/self_train.yaml'),
                         is_debug=True,
                         language='zh',
                         root_work_dir='./cache')
    with gr.Blocks() as demo:
        gr.Markdown('<h2><center>SCEPTER SELF TRAIN</center><h2>')
        with gr.Tabs(elem_id='tabs') as tabs:
            with gr.TabItem('editor', id=1, elem_id=f'tab_{1}'):
                st_ins.create_ui()
        st_ins.set_callbacks()
    demo.queue(status_update_rate=1).launch(show_error=True,
                                            debug=True,
                                            enable_queue=True)
