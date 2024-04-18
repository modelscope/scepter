# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import scepter.studio.preprocess.processors.caption_processors as caption_processors
import scepter.studio.preprocess.processors.image_processors as image_processors

model_dict = {'image': image_processors, 'caption': caption_processors}


class ProcessorsManager():
    def __init__(self, cfg, language='en'):
        self.type_level_processor = {}
        for processor in cfg:
            name = processor.NAME
            type = processor.TYPE
            assert type in model_dict
            assert hasattr(model_dict[type], name)
            processor_ins = getattr(model_dict[type], name)(processor,
                                                            language=language)
            if type not in self.type_level_processor:
                self.type_level_processor[type] = {}
            self.type_level_processor[type][name] = processor_ins

    def dynamic_unload(self, type='all', name='all'):
        print('Unloading {} processor model'.format(name))
        if name == 'all':
            for module_type, module_dict in self.type_level_processor.items():
                for model_name, processor_ins in module_dict.items():
                    if type == 'all' or type == module_type:
                        processor_ins.unload_model()
        else:
            for module_type, module_dict in self.type_level_processor.items():
                for model_name, processor_ins in module_dict.items():
                    if (type == 'all'
                            or type == module_type) and model_name == name:
                        processor_ins.unload_model()

    def get_choices(self, type):
        return list(self.type_level_processor.get(type, {}).keys())

    def get_default(self, type):
        processors_list = self.get_choices(type)
        return processors_list[0] if len(processors_list) > 0 else None

    def get_default_device(self, type):
        processor_ins = self.get_processor(type, self.get_default(type))
        return processor_ins.use_device

    def get_default_memory(self, type):
        processor_ins = self.get_processor(type, self.get_default(type))
        return f'{processor_ins.use_memory}M'

    def get_processor(self, type, name):
        if type not in self.type_level_processor:
            return None
        if name in self.type_level_processor[type]:
            return self.type_level_processor[type].get(name, None)
