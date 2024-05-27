# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from collections import OrderedDict
from glob import glob

import gradio as gr
import scepter
from scepter.modules.utils.config import Config
from scepter.modules.utils.file_system import FS
from scepter.studio.inference.inference_manager.infer_runer import \
    PipelineManager
from scepter.studio.inference.inference_ui.component_names import \
    InferenceUIName
from scepter.studio.inference.inference_ui.control_ui import ControlUI
from scepter.studio.inference.inference_ui.diffusion_ui import DiffusionUI
from scepter.studio.inference.inference_ui.gallery_ui import GalleryUI
from scepter.studio.inference.inference_ui.largen_ui import LargenUI
from scepter.studio.inference.inference_ui.mantra_ui import MantraUI
from scepter.studio.inference.inference_ui.model_manage_ui import ModelManageUI
from scepter.studio.inference.inference_ui.refiner_ui import RefinerUI
from scepter.studio.inference.inference_ui.stylebooth_ui import StyleboothUI
from scepter.studio.inference.inference_ui.tuner_ui import TunerUI
from scepter.studio.utils.env import init_env

UI_MAP = [('diffusion', DiffusionUI), ('mantra', MantraUI), ('tuner', TunerUI),
          ('control', ControlUI), ('refiner', RefinerUI), ('largen', LargenUI),
          ('stylebooth', StyleboothUI)]


class InferenceUI():
    def __init__(self,
                 cfg_general_file,
                 is_debug=False,
                 language='en',
                 root_work_dir='./'):
        config_dir = os.path.dirname(cfg_general_file)
        cfg_general = Config(cfg_file=cfg_general_file)
        cfg_general.WORK_DIR = os.path.join(root_work_dir,
                                            cfg_general.WORK_DIR)
        if not FS.exists(cfg_general.WORK_DIR):
            FS.make_dir(cfg_general.WORK_DIR)
        cfg_general = init_env(cfg_general)
        # official mantra
        mantra_book = Config(
            cfg_file=os.path.join(os.path.dirname(scepter.dirname),
                                  cfg_general.EXTENSION_PARAS.MANTRA_BOOK))
        cfg_general.MANTRAS = mantra_book.MANTRAS
        # official tuners
        official_tuners = Config(
            cfg_file=os.path.join(os.path.dirname(scepter.dirname),
                                  cfg_general.EXTENSION_PARAS.OFFICIAL_TUNERS))
        cfg_general.TUNERS = official_tuners.TUNERS
        official_controllers = Config(cfg_file=os.path.join(
            os.path.dirname(scepter.dirname),
            cfg_general.EXTENSION_PARAS.OFFICIAL_CONTROLLERS))
        cfg_general.CONTROLLERS = official_controllers.CONTROLLERS

        # customized tuners
        tuner_manager = Config(
            cfg_file=os.path.join(os.path.dirname(scepter.dirname),
                                  cfg_general.EXTENSION_PARAS.TUNER_MANAGER))
        tuner_manager = os.path.join(root_work_dir, tuner_manager.WORK_DIR,
                                     tuner_manager.TUNER_LIST_YAML)
        if FS.exists(tuner_manager):
            with FS.get_from(tuner_manager) as local_path:
                custom_tuners = Config(cfg_file=local_path)
            cfg_general.CUSTOM_TUNERS = custom_tuners.get('TUNERS', [])
        else:
            cfg_general.CUSTOM_TUNERS = []

        pipe_manager = PipelineManager()
        config_list = glob(os.path.join(config_dir, '*/*_pro.yaml'),
                           recursive=True)
        for config_file in config_list:
            pipe_manager.register_pipeline(Config(cfg_file=config_file))

        for one_tuner in cfg_general.TUNERS:
            pipe_manager.register_tuner(
                one_tuner,
                name=one_tuner.NAME_ZH if language == 'zh' else one_tuner.NAME)

        for one_controller in cfg_general.CONTROLLERS:
            pipe_manager.register_controllers(one_controller)

        for one_tuner in cfg_general.CUSTOM_TUNERS:
            pipe_manager.register_tuner(
                one_tuner,
                name=one_tuner.NAME_ZH if language == 'zh' else one_tuner.NAME,
                is_customized=True)

        self.model_manage_ui = ModelManageUI(cfg_general,
                                             pipe_manager,
                                             is_debug=is_debug,
                                             language=language)
        self.gallery_ui = GalleryUI(cfg_general,
                                    pipe_manager,
                                    is_debug=is_debug,
                                    language=language)
        self.component_names = InferenceUIName(language=language)
        self.tab_ui = OrderedDict()
        self.tab_ui_kwargs = OrderedDict()
        for name, UI in UI_MAP:
            ui = UI(cfg_general,
                    pipe_manager,
                    is_debug=is_debug,
                    language=language)
            self.tab_ui[name] = ui
            self.tab_ui_kwargs[f'{name}_ui'] = ui
            self.__setattr__(f'{name}_ui', ui)

        self.check_box_controlled_tabs = [
            'mantra', 'tuner', 'control', 'largen', 'stylebooth'
        ]
        self.pipe_manager = pipe_manager
        assert len(self.component_names.check_box_for_setting) == len(
            self.check_box_controlled_tabs)

    def create_ui(self):
        # create model
        self.model_manage_ui.create_ui(gallery_ui=self.gallery_ui)
        self.gallery_ui.create_ui()
        self.infer_info = gr.State(value=None)

        # create tabs
        def create_tab(name, ui):
            label = getattr(self.component_names, f'{name}_paras')
            if name in ['refiner']:
                ui.create_ui(gallery_ui=self.gallery_ui)
            else:
                with gr.TabItem(label=label, id=f'{name}_ui'):
                    ui.create_ui(gallery_ui=self.gallery_ui)

        with gr.Row(variant='panel', equal_height=True):
            with gr.Accordion(label=self.component_names.advance_block_name,
                              open=True):
                self.check_box_for_setting = gr.CheckboxGroup(
                    choices=self.component_names.check_box_for_setting,
                    show_label=False)
                with gr.Tabs() as self.setting_tab:
                    for name, ui in self.tab_ui.items():
                        create_tab(name, ui)

    def set_callbacks(self, manager):
        self.model_manage_ui.set_callbacks(**self.tab_ui_kwargs)
        self.gallery_ui.set_callbacks(self,
                                      self.model_manage_ui,
                                      **self.tab_ui_kwargs,
                                      manager=manager)
        for name, ui in self.tab_ui_kwargs.items():
            ui.set_callbacks(self.model_manage_ui,
                             **self.tab_ui_kwargs,
                             gallery_ui=self.gallery_ui,
                             manager=manager)

        def change_setting_tab(check_box, default_diffusion_model, *args):
            selected_tab = 'diffusion_ui'
            name2key_map = dict(
                zip(self.component_names.check_box_for_setting,
                    self.check_box_controlled_tabs))
            key2name_map = dict(
                zip(self.check_box_controlled_tabs,
                    self.component_names.check_box_for_setting))
            ui_tabs_state = dict.fromkeys(self.check_box_controlled_tabs,
                                          False)
            gr_state = dict(zip(self.check_box_controlled_tabs, args))
            for name in check_box:
                key = name2key_map[name]
                ui_tabs_state[key] = True
                if ui_tabs_state[key] != gr_state[key]:
                    selected_tab = key + '_ui'
            if len(check_box) >= 1 and selected_tab == 'diffusion_ui':
                selected_tab = name2key_map[check_box[0]] + '_ui'

            new_check_box_value = check_box
            for name in check_box:
                key = name2key_map[name]
                if ui_tabs_state[key] != gr_state[key]:
                    if key == 'largen':
                        allowed_keys = [key]
                    elif key == 'stylebooth':
                        allowed_keys = [key, 'tuner']
                    elif key == 'tuner':
                        allowed_keys = [
                            key for key in self.check_box_controlled_tabs
                            if key not in ('largen')
                        ]
                    else:
                        allowed_keys = [
                            key for key in self.check_box_controlled_tabs
                            if key not in ('largen', 'stylebooth')
                        ]
                    allowed_names = [key2name_map[key] for key in allowed_keys]
                    new_check_box_value = [
                        n for n in check_box if n in allowed_names
                    ]
                    ui_tabs_state = {
                        key: (value and key in allowed_keys)
                        for key, value in ui_tabs_state.items()
                    }

            ui_tabs_updates = [
                gr.update(visible=ui_tabs_state[key])
                for key in self.check_box_controlled_tabs
            ]

            if ui_tabs_state['largen']:
                diffusion_model = gr.Dropdown(
                    label=self.model_manage_ui.component_names.diffusion_model,
                    choices=self.model_manage_ui.
                    default_choices['diffusion_model']['choices'],
                    value='LARGEN_LargenUNetXL',
                    interactive=False)
            elif ui_tabs_state['stylebooth']:
                diffusion_model = gr.Dropdown(
                    label=self.model_manage_ui.component_names.diffusion_model,
                    choices=self.model_manage_ui.
                    default_choices['diffusion_model']['choices'],
                    value='EDIT_DiffusionUNet',
                    interactive=False)
            elif gr_state['largen'] or gr_state['stylebooth']:
                diffusion_model = gr.Dropdown(
                    label=self.model_manage_ui.component_names.diffusion_model,
                    choices=self.model_manage_ui.
                    default_choices['diffusion_model']['choices'],
                    value=self.model_manage_ui.
                    default_choices['diffusion_model']['default'],
                    interactive=True)
            else:
                diffusion_model = gr.Dropdown(
                    label=self.model_manage_ui.component_names.diffusion_model,
                    choices=self.model_manage_ui.
                    default_choices['diffusion_model']['choices'],
                    value=default_diffusion_model,
                    interactive=True)

            app_choices = self.stylebooth_ui.component_names.apps[
                -1 if ui_tabs_state['stylebooth'] and ui_tabs_state['tuner']
                else 0]
            ui_tabs_state = [
                ui_tabs_state[key] for key in self.check_box_controlled_tabs
            ]
            return (gr.CheckboxGroup(
                choices=self.component_names.check_box_for_setting,
                value=new_check_box_value,
                show_label=False), gr.update(selected=selected_tab),
                    *ui_tabs_state, *ui_tabs_updates, diffusion_model,
                    gr.update(value=app_choices))

        gr_states = [
            self.tab_ui[name].state for name in self.check_box_controlled_tabs
        ]
        gr_tabs = [
            self.tab_ui[name].tab for name in self.check_box_controlled_tabs
        ]
        self.check_box_for_setting.change(
            change_setting_tab,
            inputs=[
                self.check_box_for_setting,
                self.model_manage_ui.diffusion_model, *gr_states
            ],
            outputs=[
                self.check_box_for_setting, self.setting_tab, *gr_states,
                *gr_tabs, self.model_manage_ui.diffusion_model,
                self.stylebooth_ui.selected_app
            ],
            queue=True)


if __name__ == '__main__':
    infer_ins = InferenceUI('scepter/methods/studio/inference/inference.yaml',
                            is_debug=True,
                            language='en',
                            root_work_dir='./cache')
    with gr.Blocks() as demo:
        gr.Markdown('<h2><center>scepter studio</center><h2>')
        with gr.Tabs(elem_id='tabs') as tabs:
            with gr.TabItem('editor', id=1, elem_id=f'tab_{1}'):
                infer_ins.creat_ui()
        infer_ins.set_callbacks()
    demo.queue(status_update_rate=1).launch(show_error=True,
                                            debug=True,
                                            enable_queue=True)
