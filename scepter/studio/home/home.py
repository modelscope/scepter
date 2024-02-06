# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import gradio as gr

from scepter.modules.utils.config import Config
from scepter.studio.home.home_ui.desc_ui import DescUI
from scepter.studio.home.home_ui.guide_ui import GuideUI


class HomeUI():
    def __init__(self,
                 cfg_general_file,
                 is_debug=False,
                 language='en',
                 root_work_dir='./'):
        cfg_general = Config(cfg_file=cfg_general_file)
        desc_info = cfg_general.DESC_INFO
        self.desc_ui = DescUI(desc_info=desc_info,
                              is_debug=is_debug,
                              language=language)
        guide_info = cfg_general.GUIDE_INFO
        self.guide_ui = GuideUI(guide_info=guide_info,
                                is_debug=is_debug,
                                language=language)

    def create_ui(self):
        self.desc_ui.create_ui()
        self.guide_ui.create_ui()

    def set_callbacks(self, manager):
        self.desc_ui.set_callbacks(self.desc_ui, manager)
        self.guide_ui.set_callbacks(self.guide_ui, manager)


if __name__ == '__main__':
    st_ins = HomeUI(None,
                    is_debug=True,
                    language='zh',
                    root_work_dir='./cache')
    with gr.Blocks() as demo:
        gr.Markdown('<h2><center>SCEPTER Home</center><h2>')
        with gr.Tabs(elem_id='tabs') as tabs:
            with gr.TabItem('editor', id=1, elem_id=f'tab_{1}'):
                st_ins.create_ui()
        st_ins.set_callbacks()
    demo.queue(status_update_rate=1).launch(show_error=True,
                                            debug=True,
                                            enable_queue=True)
