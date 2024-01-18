# -*- coding: utf-8 -*-
import os
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
from scepter.studio.inference.inference_ui.mantra_ui import MantraUI
from scepter.studio.inference.inference_ui.model_manage_ui import ModelManageUI
from scepter.studio.inference.inference_ui.refiner_ui import RefinerUI
from scepter.studio.inference.inference_ui.tuner_ui import TunerUI
from scepter.studio.utils.env import init_env


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

        self.model_manage_ui = ModelManageUI(cfg_general,
                                             pipe_manager,
                                             is_debug=is_debug,
                                             language=language)
        self.gallery_ui = GalleryUI(cfg_general,
                                    pipe_manager,
                                    is_debug=is_debug,
                                    language=language)
        self.diffusion_ui = DiffusionUI(cfg_general,
                                        pipe_manager,
                                        is_debug=is_debug,
                                        language=language)
        self.mantra_ui = MantraUI(cfg_general,
                                  pipe_manager,
                                  is_debug=is_debug,
                                  language=language)
        self.tuner_ui = TunerUI(cfg_general,
                                pipe_manager,
                                is_debug=is_debug,
                                language=language)
        self.refiner_ui = RefinerUI(cfg_general,
                                    pipe_manager,
                                    is_debug=is_debug,
                                    language=language)
        self.control_ui = ControlUI(cfg_general,
                                    pipe_manager,
                                    is_debug=is_debug,
                                    language=language)
        self.component_names = InferenceUIName(language=language)

    def create_ui(self):
        self.model_manage_ui.create_ui()
        self.gallery_ui.create_ui()
        with gr.Row(variant='panel', equal_height=True):
            self.check_box_for_setting = gr.CheckboxGroup(
                choices=self.component_names.check_box_for_setting,
                show_label=False)
        with gr.Row(variant='panel', equal_height=True):
            with gr.Accordion(label=self.component_names.advance_block_name,
                              open=True):
                with gr.Tabs() as self.setting_tab:
                    with gr.TabItem(label=self.component_names.diffusion_paras,
                                    id='diffusion_ui'):
                        self.diffusion_ui.create_ui()
                    # 0
                    with gr.TabItem(label=self.component_names.mantra_paras,
                                    id='mantra_ui',
                                    visible=True) as self.mantra_tab:
                        self.mantra_ui.create_ui()
                        self.mantra_state = gr.State(value=False)
                    # 1
                    with gr.TabItem(label=self.component_names.tuner_paras,
                                    id='tuner_ui',
                                    visible=True) as self.tuner_tab:
                        self.tuner_ui.create_ui()
                        self.tuner_state = gr.State(value=False)
                    # 2
                    with gr.TabItem(label=self.component_names.contrl_paras,
                                    id='control_ui',
                                    visible=True) as self.control_tab:
                        self.control_ui.create_ui()
                        self.control_state = gr.State(value=False)
                    # 3
                    with gr.TabItem(label=self.component_names.refine_paras,
                                    id='refiner_ui',
                                    visible=True) as self.refine_tab:
                        self.refiner_ui.create_ui()

    def set_callbacks(self, manager):
        self.model_manage_ui.set_callbacks(self.diffusion_ui, self.tuner_ui,
                                           self.control_ui, self.mantra_ui)
        self.gallery_ui.set_callbacks(self, self.model_manage_ui,
                                      self.diffusion_ui, self.mantra_ui,
                                      self.tuner_ui, self.refiner_ui,
                                      self.control_ui)
        self.diffusion_ui.set_callbacks(self.model_manage_ui)
        self.mantra_ui.set_callbacks(self.model_manage_ui)
        self.tuner_ui.set_callbacks(self.model_manage_ui)
        self.control_ui.set_callbacks(self.model_manage_ui, self.diffusion_ui)
        self.refiner_ui.set_callbacks()

        def change_setting_tab(check_box):
            mantra_ui, tuner_ui, control_ui = False, False, False
            for key in check_box:
                if self.component_names.check_box_for_setting.index(key) == 0:
                    mantra_ui = True
                if self.component_names.check_box_for_setting.index(key) == 1:
                    tuner_ui = True
                if self.component_names.check_box_for_setting.index(key) == 2:
                    control_ui = True
            return (mantra_ui, tuner_ui, control_ui)

        self.check_box_for_setting.change(
            change_setting_tab,
            inputs=[self.check_box_for_setting],
            outputs=[self.mantra_state, self.tuner_state, self.control_state])


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
