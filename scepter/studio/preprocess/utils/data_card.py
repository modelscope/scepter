# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import copy
import datetime
import json
import os.path

from PIL import Image
from scepter.modules.utils.file_system import FS


def find_prefix(file_path):
    for k in FS._prefix_to_clients.keys():
        if file_path.startswith(k):
            return True, file_path, ''
        elif FS.exists(os.path.join(k, file_path)):
            return True, os.path.join(k, file_path), ''
        elif FS.exists(os.path.join(k, 'datasets', file_path)):
            return True, os.path.join(k, 'datasets', file_path), 'datasets'
    return False, None, None


def del_prefix(file_path, prefix=''):
    for k in FS._prefix_to_clients.keys():
        if file_path.startswith(k):
            file_path = file_path.replace(k, '')
            while file_path.startswith('/'):
                file_path = file_path[1:]
            if not prefix == '' and file_path.startswith(prefix):
                file_path = file_path.split(prefix)[-1]
            while file_path.startswith('/'):
                file_path = file_path[1:]
            return True, file_path
    return False, file_path


def get_image_meta(image):
    img = Image.open(image)
    return *img.size, img


class BaseDataCard(object):
    def __init__(self,
                 dataset_folder,
                 dataset_name=None,
                 user_name='admin',
                 batch=8):
        self.dataset_folder = dataset_folder
        self.batch = batch
        self.local_dataset_folder, _ = FS.map_to_local(dataset_folder)

        self.meta_file = os.path.join(dataset_folder, 'meta.json')
        self.local_meta = os.path.join(self.local_dataset_folder, 'meta.json')

        self.train_file = os.path.join(dataset_folder, 'train.csv')
        self.local_train_file = os.path.join(self.local_dataset_folder,
                                             'train.csv')

        self.save_file_list = os.path.join(dataset_folder, 'file.txt')
        self.local_save_file_list = os.path.join(self.local_dataset_folder,
                                                 'file.txt')
        self.edit_list = []
        self.edit_cursor = -1

        current_time = self.get_time()
        '''
            A legal meta is like as follows:
            {
                'dataset_name': '',
                'cursor': cursor,
                'file_list': [], # use to manage all data status
                'train_csv': train_csv,
                'save_file_list': save_file_list,
                'is_valid': True or False ,
                'create_time': 'YYYYMMDD-HHMMSS',
                'update_time': 'YYYYMMDD-HHMMSS',
                'user_name': ''
            }
        '''
        self.default_meta = {
            'dataset_name': dataset_name,
            'cursor': -1,
            'file_list': [],  # use to manage all data status
            'train_csv': self.train_file,
            'save_file_list': self.save_file_list,
            'is_valid': True,
            'create_time': current_time,
            'update_time': current_time,
            'user_name': user_name
        }
        if FS.exists(self.meta_file):
            FS.get_dir_to_local_dir(self.dataset_folder,
                                    self.local_dataset_folder,
                                    sign_key=self.meta_file)
            if FS.exists(self.local_meta):
                self.meta = self.check_legal_dataset(
                    json.load(open(self.local_meta, 'r')))
            else:
                self.meta = {}
        else:
            self.meta = {}
        if len(self.meta) < 1:
            assert dataset_name is not None
            self.meta = self.default_meta
            self.meta['dataset_name'] = dataset_name
            os.makedirs(self.local_dataset_folder, exist_ok=True)
            self.new_dataset = True
        else:
            self.new_dataset = False
        self.meta['local_work_dir'] = self.local_dataset_folder
        self.meta['work_dir'] = dataset_folder
        self.dataset_name = self.meta['dataset_name']
        self.start_cursor = (self.cursor // batch) * batch

    def check_legal_dataset(self, meta):
        # If meta has dataset, this dataset is legal.
        # otherwise, we take it as a new dataset or illegal one.
        if 'dataset_name' not in meta:
            return {}
        is_update = False
        for key, value in self.default_meta.items():
            if key not in meta:
                meta[key] = self.default_meta[key]
                is_update = True
        if is_update:
            self.save_meta(meta)
        return meta

    def get_time(self):
        return '{0:%Y%m%d%-H%M%S}'.format(datetime.datetime.now())

    def update_dataset(self):
        self.write_train_file()
        self.write_data_file()
        self.save_meta()

    def save_meta(self, meta=None):
        meta = self.meta if meta is None else meta
        save_meta = copy.deepcopy(meta)
        if 'local_work_dir' in meta:
            save_meta.pop('local_work_dir')
        if 'work_dir' in meta:
            save_meta.pop('work_dir')
        json.dump(save_meta, open(self.local_meta, 'w'))
        FS.put_object_from_local_file(self.local_meta, self.meta_file)
        return True

    def write_train_file(self):
        raise NotImplementedError

    def write_data_file(self):
        raise NotImplementedError

    @property
    def cursor(self):
        return self.meta['cursor']

    def __len__(self):
        return len(self.meta['file_list'])

    @property
    def data(self):
        return self.meta['file_list']

    def set_cursor(self, cursor):
        if cursor >= len(self):
            self.meta['cursor'] = 0
        elif len(self) == 0:
            self.meta['cursor'] = -1
        else:
            self.meta['cursor'] = cursor
        # if self.cursor is the start of current batch or end of current batch
        # we employ stride forward or backward automaticlly.
        # but now it’s not compatiable with gradio
        # if self.cursor - self.start_cursor == self.batch - 1:
        #     self.start_cursor = self.cursor
        # elif self.cursor - self.start_cursor == 0:
        #     self.start_cursor = self.cursor - self.batch + 1
        #     if self.start_cursor < 0: self.start_cursor = 0
        # else:
        #     self.start_cursor = (self.cursor//self.batch) * self.batch

    def deactive_dataset(self):
        self.meta['is_valid'] = False
        self.update_dataset()

    def active_dataset(self):
        self.meta['is_valid'] = True
        self.update_dataset()

    @property
    def is_valid(self):
        return self.meta['is_valid']

    @property
    def user_name(self):
        return self.meta['user_name']

    @property
    def work_dir(self):
        return self.meta['work_dir']

    @property
    def local_work_dir(self):
        return self.meta['local_work_dir']

    @property
    def get_batch(self):
        end_index = self.start_cursor + self.batch
        if end_index >= len(self):
            extend_data = self.data[0:end_index - len(self) + 1]
        else:
            extend_data = []
        return self.data[self.start_cursor:end_index] + extend_data

    def set_edit_range(self, range_list):
        self.edit_list = []
        illegal_tup = []
        if range_list == -1:
            self.edit_list = list(range(len(self)))
            self.samples_list = self.data
            return True, ''
        for range_tup in range_list.split(','):
            if range_tup.strip() == '':
                continue
            if '-' in range_tup:
                num_tup = range_tup.split('-')
                if not len(num_tup) == 2:
                    illegal_tup.append(
                        f"{range_tup} is illegal, more than one '-'")
                    continue
                try:
                    start_num = int(num_tup[0])
                    end_num = int(num_tup[1])
                except Exception as e:
                    illegal_tup.append(
                        f'{range_tup} is illegal, start number '
                        f'or end number is not int number, error {e}. ')
                    continue
                if start_num < 1 or end_num < 1:
                    illegal_tup.append(f'{range_tup} is illegal, start number '
                                       f'or end number should >= 1. ')
                    continue
                start_num = max(0, start_num)
                end_num = min(len(self), end_num)
                if start_num > len(self) or end_num > len(self):
                    illegal_tup.append(
                        f'{range_tup} is illegal, start number '
                        f'or end number should <= length of this dataset {len(self)}. '
                    )
                    continue
                self.edit_list.extend(
                    list(range(start_num - 1, end_num - 1, 1)))
            else:
                try:
                    num = int(range_tup)
                except Exception as e:
                    illegal_tup.append(
                        f'{range_tup} is illegal, number is not int number,'
                        f' error {e}. ')
                    continue

                if num < 1:
                    illegal_tup.append(f'{range_tup} is illegal, number '
                                       f'should >= 1. ')
                    continue
                if num > len(self):
                    illegal_tup.append(
                        f'{range_tup} is illegal, number '
                        f'should <= length of this dataset {len(self)}. ')
                    continue
                self.edit_list.append(num - 1)
        self.edit_list.sort()
        self.samples_list = [self.data[index] for index in self.edit_list]
        if len(self.samples_list) < 1:
            illegal_tup.append('select list is blank.')
        if len(illegal_tup) > 0:
            return False, ' '.join(illegal_tup)
        else:
            return True, ''

    @property
    def edit_samples(self):
        return self.samples_list

    def edit_index_from_cursor(self, cursor):
        if cursor in self.edit_list:
            return self.edit_list.index(cursor)
        return -1

    def cursor_from_edit_index(self, index):
        if index < 0 or index >= len(self.edit_list):
            return -1
        return self.edit_list[index]

    def apply_changes(self):
        edit_index_list = self.edit_list
        for index in edit_index_list:
            one_data = self.data[index]

            relative_image_path = one_data['relative_path']
            local_image_path = os.path.join(self.meta['local_work_dir'],
                                            relative_image_path)
            image_path = one_data['image_path']

            edit_relative_image_path = one_data.get('edit_relative_path',
                                                    one_data['relative_path'])
            local_edit_image_path = os.path.join(self.meta['local_work_dir'],
                                                 edit_relative_image_path)
            if not relative_image_path == edit_relative_image_path:
                try:
                    os.rename(local_edit_image_path, local_image_path)
                except Exception as e:
                    msg = f'Apply edited image failed, error is {e}'
                    return False, msg
                FS.put_object_from_local_file(local_image_path, image_path)
                try:
                    os.remove(local_edit_image_path)
                except Exception:
                    pass
            self.data[index]['edit_relative_path'] = one_data['relative_path']
            self.data[index]['edit_image_path'] = one_data['image_path']
            self.data[index]['caption'] = one_data['edit_caption']
            self.data[index]['width'] = one_data['edit_width']
            self.data[index]['height'] = one_data['edit_height']
        self.update_dataset()
        return True, ''

    # @property
    # def select_index(self):
    #     return self.cursor - self.start_cursor

    @property
    def current_record(self):
        if self.cursor >= len(self):
            self.set_cursor(0)
        if len(self) < 1:
            self.set_cursor(-1)
        if self.cursor >= 0:
            return self.meta['file_list'][self.cursor]
        else:
            return {}

    def modify_data_name(self, new_dataset_name):
        self.meta['dataset_name'] = new_dataset_name
        self.update_dataset()

    def edit_caption(self, edit_caption):
        if self.cursor >= 0:
            self.data[self.cursor]['edit_caption'] = edit_caption

    def set_caption(self, edit_caption=None):
        if self.cursor >= 0:
            self.data[self.cursor]['caption'] = self.data[self.cursor][
                'edit_caption'] if edit_caption is None else edit_caption
