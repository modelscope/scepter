# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import argparse
import importlib
import os
import sys

#
from scepter.modules.utils.registry import REGISTRY_LIST
from scepter.modules.utils.ast_utils import load_index
from scepter.modules.utils.import_utils import LazyImportModule


if os.path.exists('__init__.py'):
    package_name = 'scepter_ext'
    spec = importlib.util.spec_from_file_location(package_name, '__init__.py')
    package = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = package
    spec.loader.exec_module(package)
sys.path.insert(0, os.path.abspath(os.curdir))


def get_module_list():
    ast_index = load_index()
    index_keys = ast_index.get('index', {}).keys()
    ret_msg = [item[0] for item in index_keys]
    ret_msg.extend(v.name for v in REGISTRY_LIST)
    ret_msg = sorted(set(ret_msg))
    print(ret_msg)
    return None


def get_module_objects(module_name):
    ret_msg = 'Not support module!'
    ast_index = load_index()
    class_list = []
    func_list = []

    for k in ast_index.get('index').keys():
        if k[0] == module_name:
            class_list.append(k[1])
    for v in REGISTRY_LIST:
        if v.name == module_name:
            class_map = v.class_map
            func_map = v.func_map
            for i in class_map:
                class_list.append(i)
            for i in func_map:
                func_list.append(i)

    if class_list or func_list:
        ret_msg = 'The {} module support the following object: \n'.format(module_name)
        class_list = list(set(class_list))
        class_list.sort()
        func_list.sort()
        index = 0
        for item in class_list:
            ret_msg += '{}: class {}\n'.format(index, item)
            index += 1
        for item in func_list:
            ret_msg += '{}: function {}\n'.format(index, item)
    print(ret_msg)
    return None


def get_module_object_config(module_name, object_name):
    ret_msg = 'Not support module or object!'
    sig = (module_name.upper(), object_name)
    if LazyImportModule.get_module_type(sig):
        LazyImportModule.import_module(sig)
    for v in REGISTRY_LIST:
        if v.name == module_name:
            class_map = v.class_map
            func_map = v.func_map

            for k in class_map:
                if object_name == k:
                    ret_msg = 'The {module_name}:{object_name} object need follow config: \n'
                    ret_msg += v.get_config_template(k)
            for k in func_map:
                if object_name == k:
                    ret_msg = 'The {module_name}:{object_name} object need follow config: \n'
                    ret_msg += v.get_config_template(k)
    print(ret_msg)
    return None


if __name__ == '__main__':
    # initialize the data manager instance

    usage_string = 'usage for the torch dist: \n' \
                   '1. view mudule list \n' \
                   '   examples: \n' \
                   '   '

    parser = argparse.ArgumentParser(usage=usage_string)

    parser.add_argument('-t',
                        '--tool',
                        dest='tool_type',
                        type=str,
                        choices=['lm', 'lo', 'config'],
                        default='data_info',
                        help='choose your operation for torch dist!')

    parser.add_argument('-m',
                        '--module',
                        dest='module',
                        type=str,
                        choices=get_module_list(),
                        default='',
                        help='choose the module from {}!'.format(
                            get_module_list()))

    parser.add_argument('-o',
                        '--object',
                        dest='object',
                        type=str,
                        default='',
                        help='choose the object!')

    args = parser.parse_args()

    if args.tool_type == 'lm':
        get_module_list()
    if args.tool_type == 'lo':
        get_module_objects(args.module)
    if args.tool_type == 'config':
        get_module_object_config(args.module, args.object)
