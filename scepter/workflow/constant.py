# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from dataclasses import dataclass

WORKFLOW_PREFIX = 'custom_nodes/ComfyUI-Scepter/'
WORKFLOW_MODEL_PREFIX = 'models/scepter/'
WORKFLOW_CONFIG_PATH = os.path.join(WORKFLOW_PREFIX, 'config/scepter_workflow.yaml')
MANTRA_CONFIG_PATH = os.path.join(WORKFLOW_PREFIX, 'config/mantra.yaml')
TUNER_CONFIG_PATH = os.path.join(WORKFLOW_PREFIX, 'config/tuner_model.yaml')
CONTROL_CONFIG_PATH = os.path.join(WORKFLOW_PREFIX, 'config/control_model.yaml')
ANNOTATOR_CONFIG_PATH = os.path.join(WORKFLOW_PREFIX, 'config/annotator.yaml')

class WorkflowConfig(object):
    def __init__(self):
        # pass
        from scepter.modules.utils.file_system import FS
        from scepter.modules.utils.config import Config
        self.workflow_config = Config(cfg_file=WORKFLOW_CONFIG_PATH)
        self.mantra_config = Config(cfg_file=MANTRA_CONFIG_PATH)
        self.tuner_config = Config(cfg_file=TUNER_CONFIG_PATH)
        self.control_config = Config(cfg_file=CONTROL_CONFIG_PATH)
        self.annotator_config = Config(cfg_file=ANNOTATOR_CONFIG_PATH)

        if 'FILE_SYSTEMS' in self.workflow_config:
            for fs_info in self.workflow_config['FILE_SYSTEMS']:
                FS.init_fs_client(fs_info)

        self.model_info = {
            item['NAME']: {
                "diffusion_model": item['DIFFUSION_MODEL'],
                "first_stage_model": item['FIRST_STAGE_MODEL'],
                "cond_stage_model": item['COND_STAGE_MODEL'],
                "config_file": os.path.join(WORKFLOW_PREFIX, item['CONFIG']),
                "config": Config(cfg_file=os.path.join(WORKFLOW_PREFIX, item['CONFIG'])),
            } for item in self.workflow_config['BASE_MODELS']
        }

        self.mantra_info = {
            item['NAME']: item for item in self.mantra_config['MANTRAS']
        }

        self.tuner_info = {
            item['NAME']: item for item in self.tuner_config['TUNERS']
        }

        self.control_info = {
            item['NAME']: item for item in self.control_config['CONTROLLERS']
        }

        self.anno_info = {
            item['TYPE']: item for item in self.annotator_config['ANNOTATORS']
        }

global WORKFLOW_CONFIG
WORKFLOW_CONFIG = WorkflowConfig()