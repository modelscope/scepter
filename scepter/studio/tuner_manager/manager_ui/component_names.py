# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
# For dataset manager


class TunerManagerNames():
    def __init__(self, language='en'):
        self.save_symbol = '\U0001F4BE'  # 💾
        self.delete_symbol = '\U0001f5d1'  # 🗑️
        self.refresh_symbol = '\U0001f504'  # 🔄
        if language == 'en':
            self.browser_block_name = 'Tuner Browser'
            self.base_models = 'Base Model-Tuner Type'
            self.tuner_models = 'Tuner Name'
            self.info_block_name = 'Tuner Info'
            self.tuner_name = 'Tuner Name'
            self.rename = 'Rename'
            self.tuner_type = 'Tuner Type'
            self.base_model_name = 'Base Model Name'
            self.tuner_desc = 'Tuner Description'
            self.tuner_example = 'Results Example'
            self.tuner_prompt_example = 'Prompt Example'
            self.save = 'save changes'
            self.delete = 'Delete'
        elif language == 'zh':
            self.browser_block_name = '微调模型查找'
            self.base_models = '基模型-微调类型'
            self.tuner_models = '微调模型名称'
            self.info_block_name = '微调模型详情'
            self.tuner_name = '微调模型名称'
            self.rename = '重命名'
            self.tuner_type = '微调模型类型'
            self.base_model_name = '基模型名称'
            self.tuner_desc = '微调模型描述'
            self.tuner_example = '示例结果'
            self.tuner_prompt_example = '示例提示词'
            self.save = '保存修改'
            self.delete = '删除'
