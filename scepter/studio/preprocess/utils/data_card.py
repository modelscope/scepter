# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import copy
import csv
import datetime
import json
import os.path

import gradio as gr
import imagehash
from PIL import Image
from tqdm import tqdm

from scepter.modules.utils.directory import get_md5
from scepter.modules.utils.file_system import FS
from scepter.studio.preprocess.caption_editor_ui.component_names import \
    Text2ImageDataCardName


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
    return img.size


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
                                    self.local_dataset_folder)
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


class Text2ImageDataCard(BaseDataCard):
    def __init__(self,
                 dataset_folder,
                 dataset_name=None,
                 src_file=None,
                 surfix=None,
                 user_name='admin',
                 language='en'):
        super().__init__(dataset_folder,
                         dataset_name=dataset_name,
                         user_name=user_name)
        self.meta['task_type'] = 'txt2img'
        self.components_name = Text2ImageDataCardName(language)
        if self.new_dataset:
            # new dataset
            if surfix == '.zip':
                file_list = self.load_from_zip(src_file, dataset_folder,
                                               self.local_dataset_folder)
            elif surfix in ['.txt', '.csv']:
                file_list = self.load_from_list(src_file, dataset_folder,
                                                self.local_dataset_folder)
            elif surfix is None:
                file_list = []
            else:
                raise gr.Error(
                    f'{self.components_name.illegal_data_err2} {surfix}')

            is_flag = FS.put_dir_from_local_dir(self.local_dataset_folder,
                                                dataset_folder,
                                                multi_thread=True)
            if not is_flag:
                raise gr.Error(f'{self.components_name.illegal_data_err3}')

            self.meta['cursor'] = 0 if len(file_list) > 0 else -1
            self.meta['file_list'] = file_list
            self.update_dataset()
        else:
            for da_idx, cur_data in enumerate(self.data):
                if 'edit_caption' not in cur_data:
                    self.data[da_idx]['edit_caption'] = cur_data['caption']
                if 'edit_image_path' not in cur_data:
                    self.data[da_idx]['edit_image_path'] = cur_data[
                        'image_path']
                if 'edit_relative_path' not in cur_data:
                    self.data[da_idx]['edit_relative_path'] = cur_data[
                        'relative_path']
                if 'edit_width' not in cur_data:
                    self.data[da_idx]['edit_width'] = cur_data['width']
                if 'edit_height' not in cur_data:
                    self.data[da_idx]['edit_height'] = cur_data['height']

    def load_from_zip(self, save_file, data_folder, local_dataset_folder):
        with FS.get_from(save_file) as local_path:
            res = os.popen(
                f"unzip -o '{local_path}' -d '{local_dataset_folder}'")
            res = res.readlines()
        if not os.path.exists(local_dataset_folder):
            raise gr.Error(f'Unzip {save_file} failed {str(res)}')
        file_folder = None
        train_list = None
        hit_dir = None
        raw_list = {}
        mac_osx = os.path.join(local_dataset_folder, '__MACOSX')
        if os.path.exists(mac_osx):
            res = os.popen(f"rm -rf '{mac_osx}'")
            res = res.readlines()
        for one_dir in FS.walk_dir(local_dataset_folder, recurse=False):
            if one_dir.endswith('__MACOSX'):
                res = os.popen(f"rm -rf '{one_dir}'")
                res = res.readlines()
                continue
            if FS.isdir(one_dir):
                if one_dir.endswith('images') or one_dir.endswith('images/'):
                    file_folder = one_dir
                    hit_dir = one_dir
                else:
                    sub_dir = FS.walk_dir(one_dir)
                    for one_s_dir in sub_dir:
                        if FS.isdir(one_s_dir) and one_s_dir.split(
                                one_dir)[1].replace('/', '') == 'images':
                            file_folder = one_s_dir
                            hit_dir = one_dir
                        if FS.isfile(one_s_dir) and one_s_dir.split(
                                one_dir)[1].replace('/', '') == 'train.csv':
                            train_list = one_s_dir
                        if file_folder is not None and train_list is not None:
                            break
                        if (one_s_dir.endswith('.jpg')
                                or one_s_dir.endswith('.jpeg')
                                or one_s_dir.endswith('.png')
                                or one_s_dir.endswith('.webp')):
                            file_name, surfix = os.path.splitext(one_s_dir)
                            txt_file = file_name + '.txt'
                            if os.path.exists(txt_file):
                                raw_list[one_s_dir] = txt_file
                            else:
                                raw_list[one_s_dir] = None
            elif one_dir.endswith('train.csv'):
                train_list = one_dir
            else:
                if (one_dir.endswith('.jpg') or one_dir.endswith('.jpeg')
                        or one_dir.endswith('.png')
                        or one_dir.endswith('.webp')):
                    file_name, surfix = os.path.splitext(one_dir)
                    txt_file = file_name + '.txt'
                    if os.path.exists(txt_file):
                        raw_list[one_dir] = txt_file
                    else:
                        raw_list[one_dir] = None
            if file_folder is not None and train_list is not None:
                break
        if file_folder is None and len(raw_list) < 1:
            raise gr.Error(
                "images folder or train.csv doesn't exists, or nothing exists in your zip"
            )
        new_file_folder = f'{local_dataset_folder}/images'
        os.makedirs(new_file_folder, exist_ok=True)
        if file_folder is not None:
            _ = FS.get_dir_to_local_dir(file_folder, new_file_folder)
        elif len(raw_list) > 0:
            raw_list = [[k, v] for k, v in raw_list.items()]
            for img_id, cur_image in enumerate(raw_list):
                image_name, surfix = os.path.splitext(cur_image[0])
                if cur_image[1] is not None and os.path.exists(cur_image[1]):
                    prompt = open(cur_image[1], 'r').read()
                else:
                    prompt = image_name.split('/')[-1]
                try:
                    os.rename(
                        os.path.abspath(cur_image[0]),
                        f'{new_file_folder}/{get_md5(cur_image[0])}{surfix}')
                    raw_list[img_id] = [
                        os.path.join('images',
                                     f'{get_md5(cur_image[0])}{surfix}'),
                        prompt
                    ]
                except Exception as e:
                    print(e)

        if not os.path.exists(new_file_folder):
            raise gr.Error(f'{str(res)}')
        new_train_list = f'{local_dataset_folder}/train.csv'
        if train_list is None or not os.path.exists(train_list):
            with open(new_train_list, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['Target:FILE', 'Prompt'])
                for cur_image, cur_prompt in raw_list:
                    writer.writerow([cur_image, cur_prompt])
        else:
            res = os.popen(f"mv '{train_list}' '{new_train_list}'")
            res = res.readlines()
        if not os.path.exists(new_train_list):
            raise gr.Error(f'{str(res)}')
        if not file_folder == hit_dir:
            try:
                res = os.popen(f"rm -rf '{hit_dir}/images/*'")
                _ = res.readlines()
                res = os.popen(f"rm -rf '{hit_dir}'")
                _ = res.readlines()
            except Exception:
                pass
        file_list = self.load_train_file(new_train_list, data_folder)
        # remove unused data
        for one_dir in FS.walk_dir(local_dataset_folder):
            if "images" in one_dir or one_dir.endswith("file.csv") or one_dir.endswith("train.csv"):
                continue
            try:
                os.system(f"rm -rf {one_dir}")
            except:
                pass
        return file_list

    def load_train_file(self, file_path, data_folder):
        base_folder = os.path.dirname(file_path)
        file_list = []
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                image_path, prompt = row[0], row[1]
                if image_path == 'Target:FILE':
                    continue
                local_image_path = os.path.join(base_folder, image_path)
                w, h = get_image_meta(local_image_path)
                file_list.append({
                    'image_path':
                    os.path.join(data_folder, image_path),
                    'relative_path':
                    image_path,
                    'width':
                    w,
                    'height':
                    h,
                    'caption':
                    prompt,
                    'prefix':
                    '',
                    'edit_caption':
                    prompt,
                    'edit_image_path':
                    os.path.join(data_folder, image_path),
                    'edit_relative_path':
                    image_path,
                    'edit_width':
                    w,
                    'edit_height':
                    h,
                })
        return file_list

    def load_from_list(self, save_file, dataset_folder, local_dataset_folder):
        file_list = []
        images_folder = os.path.join(local_dataset_folder, 'images')
        os.makedirs(images_folder, exist_ok=True)
        with FS.get_from(save_file) as local_path:
            all_remote_list, all_local_list = [], []
            all_save_list = []
            with open(local_path, 'r') as f:
                for line in tqdm(f):
                    line = line.strip()
                    if line == '':
                        continue
                    try:
                        image_path, width, height, caption = line.split(
                            '#;#', 3)
                    except Exception:
                        try:
                            image_path, width, height, caption = line.split(
                                ',', 3)
                        except Exception:
                            raise gr.Error(
                                self.components_name.illegal_data_err1)
                    is_legal, new_path, prefix = find_prefix(image_path)
                    try:
                        int(width), int(height)
                    except Exception:
                        raise gr.Error(
                            self.components_name.illegal_data_err4.format(
                                width, height))

                    if not is_legal:
                        raise gr.Error(
                            self.components_name.illegal_data_err5.format(
                                image_path))
                    relative_path = os.path.join('images',
                                                 image_path.split('/')[-1])

                    all_remote_list.append(new_path)
                    all_local_list.append(
                        os.path.join(local_dataset_folder, relative_path))
                    all_save_list.append(
                        os.path.join(dataset_folder, relative_path))
                    file_list.append({
                        'image_path':
                        os.path.join(dataset_folder, relative_path),
                        'relative_path':
                        relative_path,
                        'width':
                        int(width),
                        'height':
                        int(height),
                        'caption':
                        caption,
                        'prefix':
                        prefix,
                        'edit_caption':
                        caption,
                        'edit_image_path':
                        os.path.join(dataset_folder, image_path),
                        'edit_relative_path':
                        image_path,
                        'edit_width':
                        int(width),
                        'edit_height':
                        int(height),
                    })
        cache_file_list = []
        for idx, local_path in enumerate(
                FS.get_batch_objects_from(all_remote_list)):
            if local_path is None:
                raise gr.Error(
                    self.components_name.illegal_data_err6.format(
                        all_remote_list[idx]))
            _ = FS.put_object_from_local_file(local_path, all_local_list[idx])
            cache_file_list.append(local_path)

        for local_path, target_path, flg in FS.put_batch_objects_to(
                cache_file_list, all_save_list):
            if not flg:
                raise gr.Error(
                    self.components_name.illegal_data_err7.format(local_path))
            if os.path.exists(local_path):
                try:
                    os.remove(local_path)
                except Exception:
                    pass

        return file_list

    def write_train_file(self):
        file_list = self.meta['file_list']
        with open(self.local_train_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['Target:FILE', 'Prompt'])
            for one_file in file_list:
                relative_file = one_file['relative_path']
                if relative_file.startswith('/'):
                    relative_file = relative_file[1:]
                writer.writerow([relative_file, one_file['caption']])
        FS.put_object_from_local_file(self.local_train_file, self.train_file)

    def write_data_file(self):
        file_list = self.meta['file_list']
        with open(self.local_save_file_list, 'w') as f:
            for one_file in file_list:
                is_flag, file_path = del_prefix(one_file['image_path'],
                                                prefix=one_file['prefix'])
                f.write('{}#;#{}#;#{}#;#{}\n'.format(file_path,
                                                     one_file['width'],
                                                     one_file['height'],
                                                     one_file['caption']))
        FS.put_object_from_local_file(self.local_save_file_list,
                                      self.save_file_list)

    def add_record(self, image, caption, **kwargs):
        local_work_dir = self.meta['local_work_dir']
        work_dir = self.meta['work_dir']

        save_folder = os.path.join(local_work_dir, 'images')
        os.makedirs(save_folder, exist_ok=True)

        w, h = image.size
        relative_path = os.path.join('images', f'{imagehash.phash(image)}.png')
        image_path = os.path.join(work_dir, relative_path)
        local_image_path = os.path.join(local_work_dir, relative_path)
        with FS.put_to(image_path) as local_path:
            image.save(local_path)

        image.save(local_image_path)

        FS.put_object_from_local_file(local_image_path, image_path)

        self.data.append({
            'image_path': image_path,
            'relative_path': relative_path,
            'width': w,
            'height': h,
            'caption': caption,
            'prefix': '',
            'edit_caption': caption,
            'edit_image_path': image_path,
            'edit_relative_path': relative_path,
            'edit_width': w,
            'edit_height': h
        })

        self.set_cursor(len(self.meta['file_list']) - 1)
        self.update_dataset()
        return True

    def delete_record(self):
        if len(self) < 1:
            raise gr.Error(self.components_name.delete_err1)
        current_file = self.data.pop(self.cursor)
        local_file = os.path.join(self.meta['local_work_dir'],
                                  current_file['relative_path'])
        try:
            os.remove(local_file)
        except Exception:
            print(f'remove file {local_file} error')
        if self.cursor >= len(self.meta['file_list']):
            self.set_cursor(0)
        if len(self.meta['file_list']) == 0:
            self.set_cursor(-1)
        self.update_dataset()

    def export_zip(self, export_folder):
        self.update_dataset()
        zip_path = os.path.join(export_folder, f'{self.dataset_name}.zip')
        local_zip, _ = FS.map_to_local(zip_path)
        os.makedirs(os.path.dirname(local_zip), exist_ok=True)
        res = os.popen(
            f"cd '{self.local_work_dir}' && mkdir -p '{self.dataset_name}' "
            f"&& cp -rf images '{self.dataset_name}/images' "
            f"&& cp -rf train.csv '{self.dataset_name}/train.csv' "
            f"&& zip -r '{os.path.abspath(local_zip)}' '{self.dataset_name}'/* "
            f"&& rm -rf '{self.dataset_name}'")
        print(res.readlines())
        if not FS.exists(zip_path):
            raise gr.Error(self.components_name.export_zip_err1)
        return local_zip


class Image2ImageDataCard(BaseDataCard):
    def __init__(self,
                 dataset_folder,
                 dataset_name=None,
                 src_file=None,
                 surfix=None,
                 user_name='admin',
                 language='en'):
        super().__init__(dataset_folder,
                         dataset_name=dataset_name,
                         user_name=user_name)
        self.meta['task_type'] = 'txt2img'
        self.components_name = Text2ImageDataCardName(language)
        if self.new_dataset:
            # new dataset
            if surfix == '.zip':
                file_list = self.load_from_zip(src_file, dataset_folder,
                                               self.local_dataset_folder)
            elif surfix in ['.txt', '.csv']:
                file_list = self.load_from_list(src_file, dataset_folder,
                                                self.local_dataset_folder)
            elif surfix is None:
                file_list = []
            else:
                raise gr.Error(
                    f'{self.components_name.illegal_data_err2} {surfix}')

            is_flag = FS.put_dir_from_local_dir(self.local_dataset_folder,
                                                dataset_folder,
                                                multi_thread=True)
            if not is_flag:
                raise gr.Error(f'{self.components_name.illegal_data_err3}')

            self.meta['cursor'] = 0 if len(file_list) > 0 else -1
            self.meta['file_list'] = file_list
            self.update_dataset()

    def load_from_zip(self, save_file, data_folder, local_dataset_folder):
        with FS.get_from(save_file) as local_path:
            res = os.popen(
                f"unzip -o '{local_path}' -d '{local_dataset_folder}'")
            res = res.readlines()
        if not os.path.exists(local_dataset_folder):
            raise gr.Error(f'Unzip {save_file} failed {str(res)}')
        file_folder = None
        train_list = None
        hit_dir = None
        raw_list = {}
        mac_osx = os.path.join(local_dataset_folder, '__MACOSX')
        if os.path.exists(mac_osx):
            res = os.popen(f"rm -rf '{mac_osx}'")
            res = res.readlines()
        for one_dir in FS.walk_dir(local_dataset_folder, recurse=False):
            if one_dir.endswith('__MACOSX'):
                res = os.popen(f"rm -rf '{one_dir}'")
                res = res.readlines()
                continue
            if FS.isdir(one_dir):
                if one_dir.endswith('images') or one_dir.endswith('images/'):
                    file_folder = one_dir
                    hit_dir = one_dir
                else:
                    sub_dir = FS.walk_dir(one_dir)
                    for one_s_dir in sub_dir:
                        if FS.isdir(one_s_dir) and one_s_dir.split(
                                one_dir)[1].replace('/', '') == 'images':
                            file_folder = one_s_dir
                            hit_dir = one_dir
                        if FS.isfile(one_s_dir) and one_s_dir.split(
                                one_dir)[1].replace('/', '') == 'train.csv':
                            train_list = one_s_dir
                        if file_folder is not None and train_list is not None:
                            break
                        if (one_s_dir.endswith('.jpg')
                                or one_s_dir.endswith('.jpeg')
                                or one_s_dir.endswith('.png')
                                or one_s_dir.endswith('.webp')):
                            file_name, surfix = os.path.splitext(one_s_dir)
                            txt_file = file_name + '.txt'
                            if os.path.exists(txt_file):
                                raw_list[one_s_dir] = txt_file
                            else:
                                raw_list[one_s_dir] = None
            elif one_dir.endswith('train.csv'):
                train_list = one_dir
            else:
                if (one_dir.endswith('.jpg') or one_dir.endswith('.jpeg')
                        or one_dir.endswith('.png')
                        or one_dir.endswith('.webp')):
                    file_name, surfix = os.path.splitext(one_dir)
                    txt_file = file_name + '.txt'
                    if os.path.exists(txt_file):
                        raw_list[one_dir] = txt_file
                    else:
                        raw_list[one_dir] = None
            if file_folder is not None and train_list is not None:
                break
        if file_folder is None and len(raw_list) < 1:
            raise gr.Error(
                "images doesn't exist, or nothing exists in your zip")

        if train_list is None:
            raise gr.Error("pair list doesn't exist")
        new_file_folder = f'{local_dataset_folder}/images'
        os.makedirs(new_file_folder, exist_ok=True)

        images_dict = {}
        if file_folder is not None:
            _ = FS.get_dir_to_local_dir(file_folder, new_file_folder)
        elif len(raw_list) > 0:
            raw_list = [[k, v] for k, v in raw_list.items()]
            for img_id, cur_image in enumerate(raw_list):
                image_name, surfix = os.path.splitext(cur_image[0])
                try:
                    os.rename(
                        os.path.abspath(cur_image[0]),
                        f'{new_file_folder}/{get_md5(cur_image[0])}{surfix}')
                    images_dict[os.path.basename(image_name)] = os.path.join(
                        'images', f'{get_md5(cur_image[0])}{surfix}'),
                except Exception as e:
                    print(e)

        if not os.path.exists(new_file_folder):
            raise gr.Error(f'{str(res)}')
        new_train_list = f'{local_dataset_folder}/train.csv'
        res = os.popen(f"mv '{train_list}' '{new_train_list}'")
        res = res.readlines()
        if not os.path.exists(new_train_list):
            raise gr.Error(f'{str(res)}')
        if not file_folder == hit_dir:
            try:
                res = os.popen(f"rm -rf '{hit_dir}/images/*'")
                _ = res.readlines()
                res = os.popen(f"rm -rf '{hit_dir}'")
                _ = res.readlines()
            except Exception:
                pass
        file_list = self.load_train_file(new_train_list, data_folder,
                                         images_dict)
        return file_list

    def load_train_file(self, file_path, data_folder, images_dict={}):
        base_folder = os.path.dirname(file_path)
        file_list = []
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                src_image_path, image_path, prompt = row[0], row[1], row[2]
                if image_path == 'Target:FILE':
                    continue

                src_image_name, surfix = os.path.splitext(src_image_path)
                src_image_path = images_dict.get(
                    src_image_path, os.path.basename(src_image_name))
                local_src_image_path = os.path.join(base_folder,
                                                    src_image_path)
                src_w, src_h = get_image_meta(local_src_image_path)

                image_name, surfix = os.path.splitext(image_path)
                image_path = images_dict.get(image_path,
                                             os.path.basename(image_name))
                local_image_path = os.path.join(base_folder, image_path)
                w, h = get_image_meta(local_image_path)

                file_list.append({
                    'image_path':
                    os.path.join(data_folder, image_path),
                    'relative_path':
                    image_path,
                    'width':
                    w,
                    'height':
                    h,
                    'src_image_path':
                    os.path.join(data_folder, src_image_path),
                    'src_relative_path':
                    src_image_path,
                    'src_width':
                    src_w,
                    'src_height':
                    src_h,
                    'caption':
                    prompt,
                    'prefix':
                    '',
                    'edit_caption':
                    prompt
                })
        return file_list

    def load_from_list(self, save_file, dataset_folder, local_dataset_folder):
        file_list = []
        images_folder = os.path.join(local_dataset_folder, 'images')
        os.makedirs(images_folder, exist_ok=True)
        with FS.get_from(save_file) as local_path:
            all_remote_list, all_local_list = [], []
            all_src_remote_list, all_src_local_list = [], []
            all_save_list, all_src_save_list = [], []
            with open(local_path, 'r') as f:
                for line in tqdm(f):
                    line = line.strip()
                    if line == '':
                        continue
                    try:
                        image_path, width, height, src_image_path, src_width, src_height, caption = line.split(
                            '#;#', 6)
                    except Exception:
                        try:
                            image_path, width, height, src_image_path, src_width, src_height, caption = line.split(
                                ',', 6)
                        except Exception:
                            raise gr.Error(
                                self.components_name.illegal_data_err1)
                    is_legal, new_path, prefix = find_prefix(image_path)
                    try:
                        int(width), int(height)
                    except Exception:
                        raise gr.Error(
                            self.components_name.illegal_data_err4.format(
                                width, height))

                    is_legal, new_src_path, src_prefix = find_prefix(
                        src_image_path)
                    try:
                        int(src_width), int(src_height)
                    except Exception:
                        raise gr.Error(
                            self.components_name.illegal_data_err4.format(
                                src_width, src_height))

                    if not is_legal:
                        raise gr.Error(
                            self.components_name.illegal_data_err5.format(
                                image_path + ' ' + src_image_path))
                    relative_path = os.path.join('images',
                                                 image_path.split('/')[-1])

                    src_relative_path = os.path.join(
                        'images',
                        src_image_path.split('/')[-1])

                    all_remote_list.append(new_path)
                    all_local_list.append(
                        os.path.join(local_dataset_folder, relative_path))
                    all_save_list.append(
                        os.path.join(dataset_folder, relative_path))

                    all_src_remote_list.append(new_src_path)
                    all_src_local_list.append(
                        os.path.join(local_dataset_folder, src_relative_path))
                    all_src_save_list.append(
                        os.path.join(dataset_folder, src_relative_path))

                    file_list.append({
                        'image_path':
                        os.path.join(dataset_folder, relative_path),
                        'src_image_path':
                        os.path.join(dataset_folder, src_relative_path),
                        'relative_path':
                        relative_path,
                        'src_relative_path':
                        src_relative_path,
                        'width':
                        int(width),
                        'height':
                        int(height),
                        'src_width':
                        int(src_width),
                        'src_height':
                        int(src_height),
                        'caption':
                        caption,
                        'prefix':
                        prefix,
                        'edit_caption':
                        caption
                    })
        cache_file_list = []
        for idx, local_path in enumerate(
                FS.get_batch_objects_from(all_remote_list)):
            if local_path is None:
                raise gr.Error(
                    self.components_name.illegal_data_err6.format(
                        all_remote_list[idx]))
            _ = FS.put_object_from_local_file(local_path, all_local_list[idx])
            cache_file_list.append(local_path)

        for local_path, target_path, flg in FS.put_batch_objects_to(
                cache_file_list, all_save_list):
            if not flg:
                raise gr.Error(
                    self.components_name.illegal_data_err7.format(local_path))
            if os.path.exists(local_path):
                try:
                    os.remove(local_path)
                except Exception:
                    pass

        cache_src_file_list = []
        for idx, local_path in enumerate(
                FS.get_batch_objects_from(all_src_remote_list)):
            if local_path is None:
                raise gr.Error(
                    self.components_name.illegal_data_err6.format(
                        all_src_remote_list[idx]))
            _ = FS.put_object_from_local_file(local_path,
                                              all_src_local_list[idx])
            cache_src_file_list.append(local_path)

        for local_path, target_path, flg in FS.put_batch_objects_to(
                cache_src_file_list, all_src_save_list):
            if not flg:
                raise gr.Error(
                    self.components_name.illegal_data_err7.format(local_path))
            if os.path.exists(local_path):
                try:
                    os.remove(local_path)
                except Exception:
                    pass
        return file_list

    def write_train_file(self):
        file_list = self.meta['file_list']
        with open(self.local_train_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['Target:FILE', 'Source:FILE', 'Prompt'])
            for one_file in file_list:
                relative_file = one_file['relative_path']
                if relative_file.startswith('/'):
                    relative_file = relative_file[1:]

                src_relative_file = one_file['src_relative_path']
                if src_relative_file.startswith('/'):
                    src_relative_file = src_relative_file[1:]

                writer.writerow(
                    [relative_file, src_relative_file, one_file['caption']])
        FS.put_object_from_local_file(self.local_train_file, self.train_file)

    def write_data_file(self):
        file_list = self.meta['file_list']
        with open(self.local_save_file_list, 'w') as f:
            for one_file in file_list:
                is_flag, file_path = del_prefix(one_file['image_path'],
                                                prefix=one_file['prefix'])
                is_flag, src_file_path = del_prefix(one_file['src_image_path'],
                                                    prefix=one_file['prefix'])
                f.write('{}#;#{}#;#{}#;#{}#;#{}#;#{}#;#{}\n'.format(
                    file_path, one_file['width'], one_file['height'],
                    src_file_path, one_file['src_width'],
                    one_file['src_height'], one_file['caption']))
        FS.put_object_from_local_file(self.local_save_file_list,
                                      self.save_file_list)

    def add_record(self, images, caption, **kwargs):
        local_work_dir = self.meta['local_work_dir']
        work_dir = self.meta['work_dir']

        save_folder = os.path.join(local_work_dir, 'images')
        os.makedirs(save_folder, exist_ok=True)
        image, src_image = images

        w, h = image.size
        relative_path = os.path.join('images', f'{imagehash.phash(image)}.png')
        image_path = os.path.join(work_dir, relative_path)
        local_image_path = os.path.join(local_work_dir, relative_path)
        image.save(local_image_path)
        FS.put_object_from_local_file(local_image_path, image_path)

        src_w, src_h = src_image.size
        src_relative_path = os.path.join('images',
                                         f'{imagehash.phash(src_image)}.png')
        src_image_path = os.path.join(work_dir, src_relative_path)
        local_src_image_path = os.path.join(local_work_dir, src_relative_path)
        image.save(local_src_image_path)
        FS.put_object_from_local_file(local_src_image_path, src_image_path)

        self.data.append({
            'image_path': image_path,
            'relative_path': relative_path,
            'width': w,
            'height': h,
            'src_image_path': src_image_path,
            'src_relative_path': src_relative_path,
            'src_width': src_w,
            'src_height': src_h,
            'caption': caption,
            'prefix': '',
            'edit_caption': caption
        })

        self.set_cursor(len(self.meta['file_list']) - 1)
        self.update_dataset()
        return True

    def delete_record(self):
        if len(self) < 1:
            raise gr.Error(self.components_name.delete_err1)
        current_file = self.data.pop(self.cursor)
        local_file = os.path.join(self.meta['local_work_dir'],
                                  current_file['relative_path'])
        try:
            os.remove(local_file)
        except Exception:
            print(f'remove file {local_file} error')
        if self.cursor >= len(self.meta['file_list']):
            self.set_cursor(0)
        if len(self.meta['file_list']) == 0:
            self.set_cursor(-1)
        self.update_dataset()

    def export_zip(self, export_folder):
        self.update_dataset()
        zip_path = os.path.join(export_folder, f'{self.dataset_name}.zip')
        local_zip, _ = FS.map_to_local(zip_path)
        os.makedirs(os.path.dirname(local_zip), exist_ok=True)
        res = os.popen(
            f"cd '{self.local_work_dir}' && mkdir -p '{self.dataset_name}' "
            f"&& cp -rf images '{self.dataset_name}/images' "
            f"&& cp -rf train.csv '{self.dataset_name}/train.csv' "
            f"&& zip -r '{os.path.abspath(local_zip)}' '{self.dataset_name}'/* "
            f"&& rm -rf '{self.dataset_name}'")
        print(res.readlines())
        if not FS.exists(zip_path):
            raise gr.Error(self.components_name.export_zip_err1)
        return local_zip
