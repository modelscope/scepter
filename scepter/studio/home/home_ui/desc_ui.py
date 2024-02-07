# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import gradio as gr

from scepter.studio.home.home_ui.component_names import DescUIName
from scepter.studio.utils.uibase import UIBase


class DescUI(UIBase):
    def __init__(self, desc_info, is_debug=False, language='en'):
        self.desc_info = desc_info
        self.language = language
        self.is_debug = is_debug
        self.component_names = DescUIName(language)

    def create_ui(self, *args, **kwargs):
        if self.language == 'en':
            gr.HTML(self.desc_info.EN_INFO)
        elif self.language == 'zh':
            gr.HTML(self.desc_info.ZH_INFO)

    def set_callbacks(self, desc_ui, manager=None):
        pass
