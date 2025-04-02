# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from __future__ import annotations

import ast
import logging
import os
import os.path as osp
import time
import traceback
from pathlib import Path
from typing import Union, Any


p = Path(__file__)


SKIP_FUNCTION_SCANNING = True
SCEPTER_PATH = p.resolve().parents[2]
REGISTER_CLASS = 'register_class'
IGNORED_PACKAGES = ['.']
SCAN_SUB_FOLDERS = [
    'modules', 'studio', 'tools', 'workflow'
]
INDEXER_FILE = 'ast_indexer'
DECORATOR_KEY = 'decorators'
EXPRESS_KEY = 'express'
FROM_IMPORT_KEY = 'from_imports'
IMPORT_KEY = 'imports'
FILE_NAME_KEY = 'filepath'
INDEX_KEY = 'index'
REQUIREMENT_KEY = 'requirements'
MODULE_KEY = 'module'
CLASS_NAME = 'class_name'


def get_ast_logger():
    ast_logger = logging.getLogger('scepter.ast')
    ast_logger.setLevel(logging.INFO)
    return ast_logger


logger = get_ast_logger()


class AstScanning(object):
    def __init__(self) -> None:
        self.result_import = dict()
        self.result_from_import = dict()
        self.result_decorator = []
        self.express = []

    def _is_sub_node(self, node: object) -> bool:
        return isinstance(node,
                          ast.AST) and not isinstance(node, ast.expr_context)

    def _is_leaf(self, node: ast.AST) -> bool:
        for field in node._fields:
            attr = getattr(node, field)
            if self._is_sub_node(attr):
                return False
            elif isinstance(attr, (list, tuple)):
                for val in attr:
                    if self._is_sub_node(val):
                        return False
        else:
            return True

    def _skip_function(self, node: Union[ast.AST, 'str']) -> bool:
        if SKIP_FUNCTION_SCANNING:
            if type(node).__name__ == 'FunctionDef' or node == 'FunctionDef':
                return True
        return False

    def _fields(self, n: ast.AST, show_offsets: bool = True) -> tuple:
        if show_offsets:
            return n._attributes + n._fields
        else:
            return n._fields

    def _leaf(self, node: ast.AST, show_offsets: bool = True) -> str:
        output = dict()
        if isinstance(node, ast.AST):
            local_dict = dict()
            for field in self._fields(node, show_offsets=show_offsets):
                field_output = self._leaf(
                    getattr(node, field), show_offsets=show_offsets)
                local_dict[field] = field_output
            output[type(node).__name__] = local_dict
            return output
        else:
            return node

    def _refresh(self):
        self.result_import = dict()
        self.result_from_import = dict()
        self.result_decorator = []
        self.result_express = []

    def scan_ast(self, node: Union[ast.AST, None, str]):
        self._setup_global()
        self.scan_import(node, indent='  ', show_offsets=False)

    def scan_import(
        self,
        node: Union[ast.AST, None, str],
        show_offsets: bool = True,
        parent_node_name: str = '',
    ) -> None | str | dict[Any, Any]:
        if node is None:
            return node
        elif self._is_leaf(node):
            return self._leaf(node, show_offsets=show_offsets)
        else:
            def _scan_import(el: Union[ast.AST, None, str],
                             parent_node_name: str = '') -> str:
                return self.scan_import(
                    el,
                    show_offsets=show_offsets,
                    parent_node_name=parent_node_name)

            outputs = dict()
            # add relative path expression
            if type(node).__name__ == 'ImportFrom':
                level = getattr(node, 'level')
                if level >= 1:
                    path_level = ''.join(['.'] * level)
                    setattr(node, 'level', 0)
                    module_name = getattr(node, 'module')
                    if module_name is None:
                        setattr(node, 'module', path_level)
                    else:
                        setattr(node, 'module', path_level + module_name)

            for field in self._fields(node, show_offsets=show_offsets):
                attr = getattr(node, field)
                if not attr:
                    outputs[field] = []
                elif self._skip_function(parent_node_name):
                    continue
                elif (isinstance(attr, list) and len(attr) == 1
                      and isinstance(attr[0], ast.AST)
                      and self._is_leaf(attr[0])):
                    local_out = _scan_import(attr[0])
                    outputs[field] = local_out
                elif isinstance(attr, list):
                    el_dict = dict()
                    for el in attr:
                        local_out = _scan_import(el, type(el).__name__)
                        name = type(el).__name__
                        if (name == 'Import' or name == 'ImportFrom'
                                or parent_node_name == 'ImportFrom'
                                or parent_node_name == 'Import'):
                            if name not in el_dict:
                                el_dict[name] = []
                            el_dict[name].append(local_out)
                    outputs[field] = el_dict
                elif isinstance(attr, ast.AST):
                    output = _scan_import(attr)
                    outputs[field] = output
                else:
                    outputs[field] = attr

                if (type(node).__name__ == 'Import'
                        or type(node).__name__ == 'ImportFrom'):
                    if type(node).__name__ == 'ImportFrom':
                        if field == 'module':
                            self.result_from_import[outputs[field]] = dict()
                        if field == 'names':
                            if isinstance(outputs[field]['alias'], list):
                                item_name = []
                                for item in outputs[field]['alias']:
                                    local_name = item['alias']['name']
                                    item_name.append(local_name)
                                self.result_from_import[
                                    outputs['module']] = item_name
                            else:
                                local_name = outputs[field]['alias']['name']
                                self.result_from_import[outputs['module']] = [
                                    local_name
                                ]

                    if type(node).__name__ == 'Import':
                        final_dict = outputs[field]['alias']
                        if isinstance(final_dict, list):
                            for item in final_dict:
                                self.result_import[item['alias']
                                                   ['name']] = item['alias']
                        else:
                            self.result_import[outputs[field]['alias']
                                               ['name']] = final_dict

                if 'decorator_list' == field and attr != []:
                    for item in attr:
                        setattr(item, CLASS_NAME, node.name)
                    self.result_decorator.extend(attr)

                if attr != [] and type(
                        attr
                ).__name__ == 'Call' and parent_node_name == 'Expr':
                    self.result_express.append(attr)
            return {IMPORT_KEY: self.result_import,
                    FROM_IMPORT_KEY: self.result_from_import,
                    DECORATOR_KEY: self.result_decorator,
                    EXPRESS_KEY: self.result_express}

    def _parse_decorator(self, node: ast.AST) -> tuple:
        def _get_attribute_item(node: ast.AST) -> tuple:
            value, id, attr = None, None, None
            if type(node).__name__ == 'Attribute':
                value = getattr(node, 'value')
                id = getattr(value, 'id', None)
                attr = getattr(node, 'attr')
            if type(node).__name__ == 'Name':
                id = getattr(node, 'id')
            return id, attr

        def _get_args_name(nodes: list) -> list:
            result = []
            for node in nodes:
                if type(node).__name__ == 'Str':
                    result.append((node.s, None))
                elif type(node).__name__ == 'Constant':
                    result.append((node.value, None))
                else:
                    result.append(_get_attribute_item(node))
            return result

        def _get_keyword_name(nodes: ast.AST) -> list:
            result = []
            for node in nodes:
                if type(node).__name__ == 'keyword':
                    attribute_node = getattr(node, 'value')
                    if type(attribute_node).__name__ == 'Str':
                        result.append((getattr(node,
                                               'arg'), attribute_node.s, None))
                    elif type(attribute_node).__name__ == 'Constant':
                        result.append(
                            (getattr(node, 'arg'), attribute_node.value, None))
                    else:
                        result.append((getattr(node, 'arg'), )
                                      + _get_attribute_item(attribute_node))
            return result

        functions = _get_attribute_item(node.func)
        args_list = _get_args_name(node.args)
        keyword_list = _get_keyword_name(node.keywords)
        return functions, args_list, keyword_list

    def _registry_indexer(self, parsed_input: tuple, class_name: str) -> tuple:
        """format registry information to a tuple indexer

        Return:
            tuple: (MODELS, ClassName, RegisterName)
        """
        functions, args_list, keyword_list = parsed_input

        if REGISTER_CLASS != functions[1]:
            return None
        output = [functions[0]]
        return (output[0], class_name, args_list)

    def parse_decorators(self, nodes: list) -> list:
        """parse the AST nodes of decorators object to registry indexer

        Args:
            nodes (list): list of AST decorator nodes

        Returns:
            list: list of registry indexer
        """
        results = []
        for node in nodes:
            if type(node).__name__ != 'Call':
                continue
            class_name = getattr(node, CLASS_NAME, None)
            func = getattr(node, 'func')
            if getattr(func, 'attr', None) != REGISTER_CLASS:
                continue

            parse_output = self._parse_decorator(node)
            index = self._registry_indexer(parse_output, class_name)
            if None is not index:
                results.append(index)
        return results

    def generate_ast(self, file):
        self._refresh()
        with open(file, 'r', encoding='utf8') as code:
            data = code.readlines()
        data = ''.join(data)
        node = ast.parse(data)
        output = self.scan_import(node, show_offsets=False)
        output[DECORATOR_KEY] = self.parse_decorators(output[DECORATOR_KEY])
        output[EXPRESS_KEY] = self.parse_decorators(output[EXPRESS_KEY])
        output[DECORATOR_KEY].extend(output[EXPRESS_KEY])
        return output


class FilesAstScanning(object):
    def __init__(self) -> None:
        self.astScaner = AstScanning()
        self.file_dirs = []
        self.requirement_dirs = []

    def _parse_import_path(self,
                           import_package: str,
                           current_path: str = None) -> str:
        """
        Args:
            import_package (str): relative import or abs import
            current_path (str): path/to/current/file
        """
        if import_package.startswith(IGNORED_PACKAGES[0]):
            return SCEPTER_PATH + '/' + '/'.join(
                import_package.split('.')[1:]) + '.py'
        elif import_package.startswith(IGNORED_PACKAGES[1]):
            current_path_list = current_path.split('/')
            import_package_list = import_package.split('.')
            level = 0
            for index, item in enumerate(import_package_list):
                if item != '':
                    level = index
                    break

            abs_path_list = current_path_list[0:-level]
            abs_path_list.extend(import_package_list[index:])
            return '/' + '/'.join(abs_path_list) + '.py'
        else:
            return current_path

    def parse_import(self, scan_result: dict) -> list:
        """parse import and from import dicts to a third party package list

        Args:
            scan_result (dict): including the import and from import result

        Returns:
            list: a list of package ignored 'scepter' and relative path import
        """
        output = []
        output.extend(list(scan_result[IMPORT_KEY].keys()))
        output.extend(list(scan_result[FROM_IMPORT_KEY].keys()))

        # get the package name
        for index, item in enumerate(output):
            if '' == item.split('.')[0]:
                output[index] = '.'
            else:
                output[index] = item.split('.')[0]

        ignored = set()
        for item in output:
            for ignored_package in IGNORED_PACKAGES:
                if item.startswith(ignored_package):
                    ignored.add(item)
        return list(set(output) - set(ignored))

    def traversal_files(self, path, check_sub_dir=None, include_init=False):
        self.file_dirs = []
        if check_sub_dir is None or len(check_sub_dir) == 0:
            self._traversal_files(path, include_init=include_init)
        else:
            for item in check_sub_dir:
                sub_dir = os.path.join(path, item)
                if os.path.isdir(sub_dir):
                    self._traversal_files(sub_dir, include_init=include_init)

    def _traversal_files(self, path, include_init=False):
        dir_list = os.scandir(path)
        for item in dir_list:
            if item.name == '__init__.py' and not include_init:
                continue
            elif (item.name.startswith('__')
                  and item.name != '__init__.py') or item.name.endswith(
                      '.json') or item.name.endswith('.md'):
                continue
            if item.is_dir():
                self._traversal_files(item.path, include_init=include_init)
            elif item.is_file() and item.name.endswith('.py'):
                self.file_dirs.append(item.path)
            elif item.is_file() and 'requirement' in item.name:
                self.requirement_dirs.append(item.path)

    def _get_single_file_scan_result(self, file):
        try:
            output = self.astScaner.generate_ast(file)
        except Exception as e:
            detail = traceback.extract_tb(e.__traceback__)
            raise Exception(
                f'During ast indexing the file {file}, a related error excepted '
                f'in the file {detail[-1].filename} at line: '
                f'{detail[-1].lineno}: "{detail[-1].line}" with error msg: '
                f'"{type(e).__name__}: {e}", please double check the origin file {file} '
                f'to see whether the file is correctly edited.')

        import_list = self.parse_import(output)
        return output[DECORATOR_KEY], import_list

    def _inverted_index(self, forward_index):
        inverted_index = dict()
        for index in forward_index:
            for item in forward_index[index][DECORATOR_KEY]:
                inverted_index[item[:2]] = {
                    FILE_NAME_KEY: index,
                    IMPORT_KEY: forward_index[index][IMPORT_KEY],
                    MODULE_KEY: forward_index[index][MODULE_KEY],
                }
                if item[-1]:
                    for register_name in item[-1]:
                        inverted_index[(item[0], register_name[0])] = {
                            FILE_NAME_KEY: index,
                            IMPORT_KEY: forward_index[index][IMPORT_KEY],
                            MODULE_KEY: forward_index[index][MODULE_KEY],
                        }
        return inverted_index

    def _module_import(self, forward_index):
        module_import = dict()
        for index, value_dict in forward_index.items():
            module_import[value_dict[MODULE_KEY]] = value_dict[IMPORT_KEY]
        return module_import

    def get_files_scan_results(self,
                               target_file_list=None,
                               target_dir=SCEPTER_PATH,
                               target_folders=SCAN_SUB_FOLDERS):
        """the entry method of the ast scan method

        Args:
            target_file_list can override the dir and folders combine
            target_dir (str, optional): the absolute path of the target directory to be scanned. Defaults to None.
            target_folder (list, optional): the list of
            sub-folders to be scanned in the target folder.
            Defaults to SCAN_SUB_FOLDERS.

        Returns:
            dict: indexer of registry
        """
        start = time.time()
        if target_file_list is not None:
            self.file_dirs = target_file_list
        else:
            self.traversal_files(target_dir, target_folders)
        logger.info(
            f'AST-Scanning the path "{target_dir}" with the following sub folders {target_folders}'
        )

        result = dict()
        for file in self.file_dirs:
            filepath = file[file.rfind('scepter'):]
            module_name = filepath.replace(osp.sep, '.').replace('.py', '')
            decorator_list, import_list = self._get_single_file_scan_result(
                file)
            result[file] = {
                DECORATOR_KEY: decorator_list,
                IMPORT_KEY: import_list,
                MODULE_KEY: module_name
            }

        inverted_index_with_results = self._inverted_index(result)
        module_import = self._module_import(result)
        index = {
            INDEX_KEY: inverted_index_with_results,
            REQUIREMENT_KEY: module_import
        }
        logger.info(
            f'Scanning done! A number of {len(inverted_index_with_results)} '
            f'components indexed or updated! Time consumed {time.time()-start}s'
        )
        return index


file_scanner = FilesAstScanning()
file_index = None


def load_index(file_list=None):
    global file_index
    if file_index is None:
        logger.info('Building ast index from scanning every file!')
        file_index = file_scanner.get_files_scan_results(file_list)
    return file_index


if __name__ == '__main__':
    index = load_index()
    print(index)
