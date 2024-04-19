# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path

from scepter.modules.utils.config import Config
from scepter.modules.utils.file_system import FS
from scepter.studio.tuner_manager.manager_ui.browser_ui import BrowserUI
from scepter.studio.tuner_manager.manager_ui.info_ui import InfoUI
from scepter.studio.utils.env import init_env


class TunerManagerUI():
    def __init__(self,
                 cfg_general_file,
                 is_debug=False,
                 language='en',
                 root_work_dir='./'):
        cfg_general = Config(cfg_file=cfg_general_file)
        cfg_general.WORK_DIR = os.path.join(root_work_dir,
                                            cfg_general.WORK_DIR)
        cfg_general.SELF_TRAIN_DIR = os.path.join(root_work_dir,
                                                  cfg_general.SELF_TRAIN_DIR)
        if not FS.exists(cfg_general.WORK_DIR):
            FS.make_dir(cfg_general.WORK_DIR)

        cfg_general = init_env(cfg_general)
        self.info_ui = InfoUI(cfg_general, language=language)
        self.browser_ui = BrowserUI(cfg_general, language=language)

    def create_ui(self):
        self.browser_ui.create_ui()
        self.info_ui.create_ui()

    def set_callbacks(self, manager):
        self.info_ui.set_callbacks(manager)
        self.browser_ui.set_callbacks(manager, self.info_ui)
