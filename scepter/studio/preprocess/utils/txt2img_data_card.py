# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import csv
import os
import time

from tqdm import tqdm

import gradio as gr
import imagehash
from scepter.modules.utils.directory import get_md5
from scepter.modules.utils.file_system import FS
from scepter.studio.preprocess.caption_editor_ui.component_names import \
    Text2ImageDataCardName
from scepter.studio.preprocess.utils.data_card import (BaseDataCard,
                                                       find_prefix,
                                                       get_image_meta)


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
                    new_name = f'{get_md5(cur_image[0])}_{int(time.time())}{surfix}'
                    os.rename(os.path.abspath(cur_image[0]),
                              f'{new_file_folder}/{new_name}')
                    raw_list[img_id] = [
                        os.path.join('images', new_name), prompt
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
            if 'images' in one_dir or one_dir.endswith(
                    'file.csv') or one_dir.endswith('train.csv'):
                continue
            # try:
            #     os.system(f'rm -rf {one_dir}')
            # except:
            #     pass
            os.system(f'rm -rf {one_dir}')
        return file_list

    def load_train_file(self, file_path, data_folder):
        base_folder = os.path.dirname(file_path)
        file_list = []
        image_set = set()
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                image_path, prompt = row[0], row[1]
                if image_path == 'Target:FILE':
                    continue
                local_image_path = os.path.join(base_folder, image_path)
                w, h, img = get_image_meta(local_image_path)
                # deal with the duplication
                if image_path in image_set:
                    basename, surfix = os.path.splitext(local_image_path)
                    image_path = f'{basename}_{int(time.time())}{surfix}'
                    new_local_image_path = os.path.join(
                        base_folder, image_path)
                    img.save(new_local_image_path)
                image_set.add(image_path)
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
                    relative_path = os.path.join(
                        'images',
                        f'{int(time.time())}_' + image_path.split('/')[-1])

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
                writer.writerow([relative_file, one_file['caption'].strip().replace("\n", "")])
        FS.put_object_from_local_file(self.local_train_file, self.train_file)

    def write_data_file(self):
        file_list = self.meta['file_list']
        with open(self.local_save_file_list, 'w') as f:
            for one_file in file_list:
                file_path = os.path.join(self.local_work_dir,
                                         one_file['relative_path'])
                f.write('{}#;#{}#;#{}#;#{}\n'.format(file_path,
                                                     one_file['width'],
                                                     one_file['height'],
                                                     one_file['caption'].strip().replace("\n", "")))
        FS.put_object_from_local_file(self.local_save_file_list,
                                      self.save_file_list)

    def add_record(self, image, caption, **kwargs):
        local_work_dir = self.meta['local_work_dir']
        work_dir = self.meta['work_dir']

        save_folder = os.path.join(local_work_dir, 'images')
        os.makedirs(save_folder, exist_ok=True)

        w, h = image.size
        relative_path = os.path.join(
            'images', f'{imagehash.phash(image)}_{int(time.time())}.jpg')
        image_path = os.path.join(work_dir, relative_path)
        local_image_path = os.path.join(local_work_dir, relative_path)
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
        self.set_cursor(self.cursor + 1)
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
        FS.put_object_from_local_file(local_zip, zip_path)
        if not FS.exists(zip_path):
            raise gr.Error(self.components_name.export_zip_err1)
        return local_zip
