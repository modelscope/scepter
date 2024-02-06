# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import gradio as gr

from scepter.studio.home.home_ui.component_names import GuideUIName
from scepter.studio.utils.uibase import UIBase


class GuideUI(UIBase):
    def __init__(self, guide_info, is_debug=False, language='en'):
        self.guide_info = guide_info
        self.language = language
        self.is_debug = is_debug
        self.component_names = GuideUIName(language)

    def create_ui(self, *args, **kwargs):
        if self.language == 'en':
            gr.HTML(self.guide_info.EN_INFO)
        elif self.language == 'zh':
            gr.HTML(self.guide_info.ZH_INFO)

    def set_callbacks(self, guide_ui, manager=None):
        pass
