# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import argparse
import importlib
import os
import sys

#
from scepter.modules.utils.registry import REGISTRY_LIST

if os.path.exists('__init__.py'):
    package_name = 'scepter_ext'
    spec = importlib.util.spec_from_file_location(package_name, '__init__.py')
    package = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = package
    spec.loader.exec_module(package)
sys.path.insert(0, os.path.abspath(os.curdir))


def get_module_list():
    ret_msg = [v.name for v in REGISTRY_LIST]
    print(ret_msg)
    return None


def get_module_objects(module_name):
    ret_msg = 'Not surpport module!'
    for v in REGISTRY_LIST:
        if v.name == module_name:
            class_map = v.class_map
            func_map = v.func_map
            ret_msg = 'The {} module surpport the following object: \n'.format(
                module_name)
            index = 0
            for k in class_map:
                ret_msg += '{}: class {}\n'.format(index, k)
                index += 1
            for k in func_map:
                ret_msg += '{}: function {}\n'.format(index, k)
                index += 1
    print(ret_msg)
    return None


def get_module_object_config(module_name, object_name):
    ret_msg = 'Not surpport module or object!'
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
