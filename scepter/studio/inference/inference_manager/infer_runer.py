# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from scepter.modules.inference.diffusion_inference import DiffusionInference
from scepter.modules.inference.largen_inference import LargenInference
from scepter.modules.inference.stylebooth_inference import StyleboothInference
from scepter.modules.utils.logger import get_logger


class PipelineManager():
    def __init__(self, logger=None):
        '''
        Args:
            logger:
        '''
        """
            Only (refine) cond model and (refine) diffusion model are binded strictly.
            Users can choose any vae or refiner according to the given diffusion model.
        """
        if logger is None:
            logger = get_logger(name='scepter')

        self.module_list = [
            'diffusion_model', 'first_stage_model', 'cond_stage_model',
            'refiner_cond_model', 'refiner_diffusion_model'
        ]
        self.pipeline_level_modules = {}
        self.module_level_choices = {}
        self.model_level_info = {}
        self.logger = logger

    def contruct_models_index(self, pipeline_name, pipeline):
        """

        Args:
            pipeline_name:
            pipeline:

        Returns:

        """
        """
            self.pipeline_level_modules is used to index the modules given pipeline
            {
                "SD_XL1.0": {
                   "diffusion_model": "",
                   ....
                }
            }
            self.module_level_choices is used to provide all choices for modules.
            an example for self.module_level_models
            {
                "diffusion_model" : {
                    "choices": [],
                    "default": ""
                }
                ....
            }

            self.model_level_info is used to index the best combination for different modules give model name
            and check the combination is legal or not.
            {
                "xxxxxx": {
                   "pipeline": [],
                   "check_bind_module": [],
                   "model_info": {}
                   ....
                }
            }

        """
        self.pipeline_level_modules[pipeline_name] = pipeline
        for module_name in self.module_list:
            if module_name not in self.module_level_choices:
                self.module_level_choices[module_name] = {
                    'choices': [],
                    'default': ''
                }
            module = getattr(pipeline, module_name)
            if module is None:
                continue
            model_name = f"{pipeline_name}_{module['name']}"
            self.module_level_choices[module_name]['choices'].append(
                model_name)
            if pipeline.is_default or self.module_level_choices[module_name][
                    'default'] == '':
                self.module_level_choices[module_name]['default'] = model_name
            if model_name not in self.model_level_info:
                self.model_level_info[model_name] = {
                    'pipeline': [],
                    'check_bind_module': [],
                    'model_info': {}
                }
            self.model_level_info[model_name]['pipeline'].append(pipeline_name)
            self.model_level_info[model_name]['model_info'] = module

    def construct_new_pipeline(self):
        pass

    def register_pipeline(self, cfg):
        pipeline_name = cfg.NAME
        if 'LARGEN' in pipeline_name:
            PipelineBuilder = LargenInference
        elif pipeline_name.startswith('EDIT'):
            PipelineBuilder = StyleboothInference
        else:
            PipelineBuilder = DiffusionInference
        new_inference = PipelineBuilder(logger=self.logger)
        new_inference.init_from_cfg(cfg)
        self.contruct_models_index(cfg.NAME, new_inference)

    def register_tuner(self, cfg, name=None, is_customized=False):
        '''
        Args:
            cfg: {
                NAME: ""
                NAME_ZH: ""
                BASE_MODEL: ""
                MODEL_PATH: "",
                DESCRIPTION: ""
            }

        Returns:

        '''
        if not is_customized:
            tuners_key = 'tuners'
        else:
            tuners_key = 'customized_tuners'

        if tuners_key not in self.module_level_choices:
            self.module_level_choices[tuners_key] = {}

        if cfg.BASE_MODEL not in self.module_level_choices[tuners_key]:
            self.module_level_choices[tuners_key][cfg.BASE_MODEL] = {
                'choices': [],
                'default': ''
            }
        if name not in self.module_level_choices[tuners_key][
                cfg.BASE_MODEL]['choices']:
            self.module_level_choices[tuners_key][
                cfg.BASE_MODEL]['choices'].append(name)
        self.module_level_choices[tuners_key][cfg.BASE_MODEL]['default'] = name
        if tuners_key not in self.model_level_info:
            self.model_level_info[tuners_key] = {}
        if cfg.BASE_MODEL not in self.model_level_info[tuners_key]:
            self.model_level_info[tuners_key][cfg.BASE_MODEL] = {}
        self.model_level_info[tuners_key][cfg.BASE_MODEL][name] = {
            'pipeline': [],
            'check_bind_module': [],
            'model_info': cfg
        }

    def register_controllers(self, cfg):
        '''
        Args:
            cfg: {
                NAME: ""
                NAME_ZH: ""
                BASE_MODEL: ""
                MODEL_PATH: "",
                DESCRIPTION: ""
            }

        Returns:

        '''
        if 'controllers' not in self.module_level_choices:
            self.module_level_choices['controllers'] = {}

        if cfg.BASE_MODEL not in self.module_level_choices['controllers']:
            self.module_level_choices['controllers'][cfg.BASE_MODEL] = {}
        if cfg.TYPE not in self.module_level_choices['controllers'][
                cfg.BASE_MODEL]:
            self.module_level_choices['controllers'][cfg.BASE_MODEL][
                cfg.TYPE] = {
                    'choices': [],
                    'default': ''
                }
        controller_name = cfg.BASE_MODEL + '_' + cfg.NAME
        self.module_level_choices['controllers'][cfg.BASE_MODEL][
            cfg.TYPE]['choices'].append(controller_name)
        self.module_level_choices['controllers'][cfg.BASE_MODEL][
            cfg.TYPE]['default'] = controller_name
        if 'controllers' not in self.model_level_info:
            self.model_level_info['controllers'] = {}
        if cfg.BASE_MODEL not in self.model_level_info['controllers']:
            self.model_level_info['controllers'][cfg.BASE_MODEL] = {}
        self.model_level_info['controllers'][
            cfg.BASE_MODEL][controller_name] = {
                'pipeline': [],
                'check_bind_module': [],
                'model_info': cfg
            }

    def get_pipeline_given_modules(self, modules):
        diffusion_model = modules['diffusion_model']
        pipepline_name = self.model_level_info[diffusion_model]['pipeline'][0]
        return self.pipeline_level_modules[pipepline_name]
