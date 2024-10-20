# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import csv
import os
import time

from PIL import Image
from tqdm import tqdm

import gradio as gr
import imagehash
from scepter.modules.utils.file_system import FS
from scepter.studio.preprocess.caption_editor_ui.component_names import \
    Image2ImageDataCardName
from scepter.studio.preprocess.utils.data_card import (BaseDataCard,
                                                       del_prefix, find_prefix,
                                                       get_image_meta)


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
        self.meta['task_type'] = 'img2img'
        self.components_name = Image2ImageDataCardName(language)
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

                if 'src_mask_path' not in cur_data:
                    src_image_name, surfix = os.path.splitext(cur_data['src_relative_path'])
                    src_mask_path = f'{src_image_name}_mask_{int(time.time()*100)}{surfix}'
                    self.meta['local_work_dir'] = self.local_dataset_folder
                    self.meta['work_dir'] = dataset_folder
                    local_src_mask_path = os.path.join(self.meta['local_work_dir'], src_mask_path)
                    src_mask_img = self.default_mask(cur_data['src_height'], cur_data['src_width'])
                    src_mask_img.save(local_src_mask_path)
                    FS.put_object_from_local_file(local_src_mask_path, os.path.join(self.meta['work_dir'], src_mask_path))
                    self.data[da_idx]['src_mask_path'] = os.path.join(self.meta['work_dir'], src_mask_path)
                    self.data[da_idx]['src_mask_relative_path'] = src_mask_path
                    self.data[da_idx]['src_mask_width'] = cur_data['src_width']
                    self.data[da_idx]['src_mask_height'] = cur_data['src_height']
                else:
                    if self.data[da_idx]['src_mask_path'].startswith(self.meta['work_dir']):
                        self.data[da_idx]['src_mask_relative_path'] = self.data[da_idx]['src_mask_relative_path'].replace(
                            self.meta['work_dir'], ""
                        )
                    if self.data[da_idx]['src_mask_relative_path'].startswith("/"):
                        self.data[da_idx]['src_mask_relative_path'] = self.data[da_idx]['src_mask_relative_path'][1:]

                if 'edit_src_mask_path' not in cur_data:
                    self.data[da_idx]['edit_src_mask_path'] = cur_data[
                        'src_mask_path']
                if 'edit_src_mask_relative_path' not in cur_data:
                    self.data[da_idx]['edit_src_mask_relative_path'] = cur_data[
                        'src_mask_relative_path']
                if 'edit_src_mask_width' not in cur_data:
                    self.data[da_idx]['edit_src_mask_width'] = cur_data['src_mask_width']
                if 'edit_src_mask_height' not in cur_data:
                    self.data[da_idx]['edit_src_mask_height'] = cur_data['src_mask_height']

                if 'edit_src_image_path' not in cur_data:
                    self.data[da_idx]['edit_src_image_path'] = cur_data[
                        'src_image_path']
                if 'edit_src_relative_path' not in cur_data:
                    self.data[da_idx]['edit_src_relative_path'] = cur_data[
                        'src_relative_path']
                if 'edit_src_width' not in cur_data:
                    self.data[da_idx]['edit_src_width'] = cur_data['src_width']
                if 'edit_src_height' not in cur_data:
                    self.data[da_idx]['edit_src_height'] = cur_data[
                        'src_height']
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
            elif one_dir.endswith('train.csv'):
                train_list = one_dir
            else:
                continue
            if file_folder is not None and train_list is not None:
                break
        if file_folder is None and len(raw_list) < 1:
            raise gr.Error(
                "images doesn't exist, or nothing exists in your zip")

        if train_list is None:
            raise gr.Error("pair list doesn't exist")
        new_file_folder = f'{local_dataset_folder}/images'
        os.makedirs(new_file_folder, exist_ok=True)

        if file_folder is not None:
            _ = FS.get_dir_to_local_dir(file_folder, new_file_folder)

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

    def default_mask(self, h, w):
        mode = 'L'  # 'L' mode is for grayscale images
        color = 0  # Color value for grayscale (0 = black, 255 = white)
        # Create the image
        image = Image.new(mode, (w, h), color)
        return image

    def load_train_file(self, file_path, data_folder):
        base_folder = os.path.dirname(file_path)
        file_list = []
        image_set = set()
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) == 3:
                    src_image_path, image_path, prompt = row[0], row[1], row[2]
                    src_mask_path = None
                elif len(row) == 4:
                    src_image_path, src_mask_path, image_path, prompt = row[0], row[1], row[2], row[3]
                else:
                    continue
                if image_path == 'Target:FILE':
                    continue
                local_src_image_path = os.path.join(base_folder,
                                                    src_image_path)
                src_w, src_h, src_img = get_image_meta(local_src_image_path)
                if src_image_path in image_set:
                    src_image_name, surfix = os.path.splitext(src_image_path)
                    src_image_path = f'{src_image_name}_{int(time.time()*100)}{surfix}'
                    new_local_image_path = os.path.join(
                        base_folder, src_image_path)
                    src_img.save(new_local_image_path)
                image_set.add(src_image_path)

                if src_mask_path is not None and not src_mask_path.strip() == '':
                    local_mask_image_path = os.path.join(base_folder, src_mask_path)
                    src_mask_w, src_mask_h, src_mask_img = get_image_meta(local_mask_image_path)
                    if src_mask_path in image_set:
                        src_mask_name, surfix = os.path.splitext(src_mask_path)
                        src_mask_path = f'{src_mask_name}_{int(time.time()*100)}{surfix}'
                        new_local_src_mask_path = os.path.join(
                            base_folder, src_mask_path)
                        src_img.save(new_local_src_mask_path)
                    image_set.add(src_mask_path)
                else:
                    src_mask_name, surfix = os.path.splitext(src_image_path)
                    src_mask_path = f'{src_mask_name}_mask_{int(time.time()*100)}{surfix}'
                    local_mask_image_path = os.path.join(base_folder, src_mask_path)
                    src_mask_img = self.default_mask(src_h, src_w)
                    src_mask_img.save(local_mask_image_path)
                    src_mask_w, src_mask_h = src_w, src_h

                local_image_path = os.path.join(base_folder, image_path)
                w, h, img = get_image_meta(local_image_path)
                if image_path in image_set:
                    image_name, surfix = os.path.splitext(image_path)
                    image_path = f'{image_name}_{int(time.time()*100)}{surfix}'
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
                    'src_image_path':
                    os.path.join(data_folder, src_image_path),
                    'src_relative_path':
                    src_image_path,
                    'src_width':
                    src_w,
                    'src_height':
                    src_h,
                    'src_mask_path':
                        os.path.join(data_folder, src_mask_path),
                    'src_mask_relative_path':
                        src_mask_path,
                    'src_mask_width':
                        src_mask_w,
                    'src_mask_height':
                        src_mask_h,
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
                    'edit_src_image_path':
                    os.path.join(data_folder, src_image_path),
                    'edit_src_relative_path':
                    src_image_path,
                    'edit_src_width':
                    src_w,
                    'edit_src_height':
                    src_h,
                    'edit_src_mask_path':
                        os.path.join(data_folder, src_mask_path),
                    'edit_src_mask_relative_path':
                        src_mask_path,
                    'edit_src_mask_width':
                        src_mask_w,
                    'edit_src_mask_height':
                        src_h,
                })
        return file_list

    def load_from_list(self, save_file, dataset_folder, local_dataset_folder):
        file_list = []
        images_folder = os.path.join(local_dataset_folder, 'images')
        os.makedirs(images_folder, exist_ok=True)
        with FS.get_from(save_file) as local_path:
            all_remote_list, all_local_list = [], []
            all_src_remote_list, all_src_local_list = [], []
            all_src_mask_remote_list, all_src_mask_local_list = [], []
            all_save_list, all_src_save_list, all_src_mask_save_list = [], [], []
            with open(local_path, 'r') as f:
                for line in tqdm(f):
                    line = line.strip()
                    if line == '':
                        continue
                    try:
                        src_image_path, src_width, src_height, src_mask_path, src_mask_width, src_mask_height, image_path, width, height, caption = line.split(
                            '#;#', 9)
                    except Exception:
                        try:
                            src_image_path, src_width, src_height, src_mask_path, src_mask_width, src_mask_height, image_path, width, height, caption = line.split(
                                ',', 9)
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

                    is_legal, new_src_mask_path, src_mask_prefix = find_prefix(
                        src_mask_path)
                    try:
                        int(src_mask_width), int(src_mask_height)
                    except Exception:
                        raise gr.Error(
                            self.components_name.illegal_data_err4.format(
                                src_mask_width, src_mask_height))


                    if not is_legal:
                        raise gr.Error(
                            self.components_name.illegal_data_err5.format(
                                image_path + ' ' + src_image_path + ' ' + src_mask_path))
                    relative_path = os.path.join(
                        'images',
                        f'{int(time.time()*100)}_' + image_path.split('/')[-1])

                    src_relative_path = os.path.join(
                        'images',
                        f'{int(time.time()*100)}_' + src_image_path.split('/')[-1])

                    src_mask_relative_path = os.path.join(
                        'images',
                        f'{int(time.time()*100)}_' + src_mask_path.split('/')[-1])

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

                    all_src_mask_remote_list.append(new_src_mask_path)
                    all_src_mask_local_list.append(
                        os.path.join(local_dataset_folder, src_mask_relative_path))
                    all_src_mask_save_list.append(
                        os.path.join(dataset_folder, src_mask_relative_path))

                    file_list.append({
                        'image_path':
                        os.path.join(dataset_folder, relative_path),
                        'src_image_path':
                        os.path.join(dataset_folder, src_relative_path),
                        'src_mask_path':
                            os.path.join(dataset_folder, src_mask_relative_path),
                        'relative_path':
                        relative_path,
                        'src_relative_path':
                        src_relative_path,
                        'src_mask_relative_path':
                        src_mask_relative_path,
                        'width':
                        int(width),
                        'height':
                        int(height),
                        'src_width':
                        int(src_width),
                        'src_height':
                        int(src_height),
                        'src_mask_width':
                            int(src_mask_width),
                        'src_mask_height':
                            int(src_mask_height),
                        'caption':
                        caption,
                        'prefix':
                        prefix,
                        'edit_caption':
                        caption,
                        'edit_image_path':
                        os.path.join(dataset_folder, relative_path),
                        'edit_relative_path':
                        relative_path,
                        'edit_width':
                        int(width),
                        'edit_height':
                        int(height),
                        'edit_src_image_path':
                        os.path.join(dataset_folder, src_relative_path),
                        'edit_src_relative_path':
                        src_image_path,
                        'edit_src_width':
                        int(src_width),
                        'edit_src_height':
                        int(src_height),
                        'edit_src_mask_path':
                            os.path.join(dataset_folder, src_mask_relative_path),
                        'edit_src_mask_relative_path':
                            src_mask_path,
                        'edit_src_mask_width':
                            int(src_mask_width),
                        'edit_src_mask_height':
                            int(src_mask_height)
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
        cache_src_mask_file_list = []
        for idx, local_path in enumerate(
                FS.get_batch_objects_from(all_src_mask_remote_list)):
            if local_path is None:
                raise gr.Error(
                    self.components_name.illegal_data_err6.format(
                        all_src_mask_remote_list[idx]))
            _ = FS.put_object_from_local_file(local_path,
                                              all_src_mask_local_list[idx])
            cache_src_mask_file_list.append(local_path)

        for local_path, target_path, flg in FS.put_batch_objects_to(
                cache_src_mask_file_list, all_src_mask_save_list):
            if not flg:
                raise gr.Error(
                    self.components_name.illegal_data_err7.format(local_path))
            if os.path.exists(local_path):
                try:
                    os.remove(local_path)
                except Exception:
                    pass
        return file_list

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

            src_relative_path = one_data['src_relative_path']
            local_src_path = os.path.join(self.meta['local_work_dir'],
                                            src_relative_path)
            src_image_path = one_data['src_image_path']

            edit_src_relative_path = one_data.get('edit_src_relative_path',
                                                    one_data['src_relative_path'])
            local_edit_src_image_path = os.path.join(self.meta['local_work_dir'],
                                                 edit_src_relative_path)
            if not src_relative_path == edit_src_relative_path:
                try:
                    os.rename(local_edit_src_image_path, local_src_path)
                except Exception as e:
                    msg = f'Apply edited image failed, error is {e}'
                    return False, msg
                FS.put_object_from_local_file(local_src_path, src_image_path)
                try:
                    os.remove(local_edit_image_path)
                except Exception:
                    pass

            src_mask_relative_path = one_data['src_mask_relative_path']
            local_src_mask_path = os.path.join(self.meta['local_work_dir'],
                                          src_mask_relative_path)
            src_mask_path = one_data['src_mask_path']

            edit_src_mask_relative_path = one_data.get('edit_src_mask_relative_path',
                                                  one_data['src_mask_relative_path'])
            local_edit_src_mask_path = os.path.join(self.meta['local_work_dir'],
                                                     edit_src_mask_relative_path)
            if not src_mask_relative_path == edit_src_mask_relative_path:
                try:
                    os.rename(local_edit_src_mask_path, local_src_mask_path)
                except Exception as e:
                    msg = f'Apply edited image failed, error is {e}'
                    return False, msg
                FS.put_object_from_local_file(local_src_mask_path, src_mask_path)
                try:
                    os.remove(local_edit_src_mask_path)
                except Exception:
                    pass


            self.data[index]['edit_relative_path'] = one_data['relative_path']
            self.data[index]['edit_image_path'] = one_data['image_path']
            self.data[index]['edit_src_relative_path'] = one_data['src_relative_path']
            self.data[index]['edit_src_image_path'] = one_data['src_image_path']
            self.data[index]['edit_src_mask_relative_path'] = one_data['src_mask_relative_path']
            self.data[index]['edit_src_mask_path'] = one_data['src_mask_path']
            self.data[index]['caption'] = one_data['edit_caption']
            self.data[index]['width'] = one_data['edit_width']
            self.data[index]['height'] = one_data['edit_height']
        self.update_dataset()
        return True, ''

    def write_train_file(self):
        file_list = self.meta['file_list']
        with open(self.local_train_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['Source:FILE', 'SourceMASK:FILE', 'Target:FILE', 'Prompt'])
            for one_file in file_list:
                relative_file = one_file['relative_path']
                if relative_file.startswith('/'):
                    relative_file = relative_file[1:]

                src_relative_file = one_file['src_relative_path']
                if src_relative_file.startswith('/'):
                    src_relative_file = src_relative_file[1:]

                src_mask_relative_file = one_file['src_mask_relative_path']
                if src_mask_relative_file.startswith('/'):
                    src_mask_relative_file = src_mask_relative_file[1:]

                writer.writerow(
                    [src_relative_file, src_mask_relative_file, relative_file, one_file['caption'].strip().replace("\n", "")])
        FS.put_object_from_local_file(self.local_train_file, self.train_file)

    def write_data_file(self):
        file_list = self.meta['file_list']
        with open(self.local_save_file_list, 'w') as f:
            for one_file in file_list:
                is_flag, file_path = del_prefix(one_file['image_path'],
                                                prefix=one_file['prefix'])
                is_flag, src_file_path = del_prefix(one_file['src_image_path'],
                                                    prefix=one_file['prefix'])
                is_flag, src_mask_file_path = del_prefix(one_file['src_mask_path'],
                                                    prefix=one_file['prefix'])
                f.write('{}#;#{}#;#{}#;#{}#;#{}#;#{}#;#{}#;#{}#;#{}#;#{}\n'.format(
                    src_file_path, one_file['src_width'],
                    one_file['src_height'], src_mask_file_path,
                    one_file['src_mask_width'], one_file['src_mask_height'],
                    file_path, one_file['width'],
                    one_file['height'], one_file['caption']))
        FS.put_object_from_local_file(self.local_save_file_list,
                                      self.save_file_list)

    def add_record(self, image, caption, **kwargs):
        local_work_dir = self.meta['local_work_dir']
        work_dir = self.meta['work_dir']

        save_folder = os.path.join(local_work_dir, 'images')
        os.makedirs(save_folder, exist_ok=True)
        w, h = image.size
        relative_path = os.path.join(
            'images', f'{imagehash.phash(image)}_{int(time.time()*100)}.jpg')
        image_path = os.path.join(work_dir, relative_path)
        local_image_path = os.path.join(local_work_dir, relative_path)
        image.save(local_image_path)
        FS.put_object_from_local_file(local_image_path, image_path)

        src_image = kwargs.pop('src_image')
        src_w, src_h = src_image.size
        src_relative_path = os.path.join(
            'images', f'{imagehash.phash(src_image)}_{int(time.time()*100)}.jpg')
        src_image_path = os.path.join(work_dir, src_relative_path)
        local_src_image_path = os.path.join(local_work_dir, src_relative_path)
        src_image.save(local_src_image_path)
        FS.put_object_from_local_file(local_src_image_path, src_image_path)

        if 'src_mask' not in kwargs:
            src_mask_image = None
        else:
            src_mask_image = kwargs.pop('src_mask')
        if src_mask_image is None:
            src_mask_image = self.default_mask(src_h, src_w)
        src_mask_w, src_mask_h = src_mask_image.size
        src_mask_relative_path = os.path.join(
            'images', f'{imagehash.phash(src_mask_image)}_{int(time.time()*100)}.jpg')
        src_mask_path = os.path.join(work_dir, src_mask_relative_path)
        local_src_mask_path = os.path.join(local_work_dir, src_mask_relative_path)
        src_mask_image.save(local_src_mask_path)
        FS.put_object_from_local_file(local_src_mask_path, src_mask_path)

        self.data.append({
            'image_path': image_path,
            'relative_path': relative_path,
            'width': w,
            'height': h,
            'src_image_path': src_image_path,
            'src_relative_path': src_relative_path,
            'src_width': src_w,
            'src_height': src_h,
            'src_mask_path': src_mask_path,
            'src_mask_relative_path': src_mask_relative_path,
            'src_mask_width': src_mask_w,
            'src_mask_height': src_mask_h,
            'caption': caption,
            'prefix': '',
            'edit_caption': caption,
            'edit_image_path': image_path,
            'edit_relative_path': relative_path,
            'edit_width': w,
            'edit_height': h,
            'edit_src_image_path': src_image_path,
            'edit_src_relative_path': src_relative_path,
            'edit_src_width': src_w,
            'edit_src_height': src_h,
            'edit_src_mask_path': src_mask_path,
            'edit_src_mask_relative_path': src_mask_relative_path,
            'edit_src_mask_width': src_mask_w,
            'edit_src_mask_height': src_mask_h
        })
        self.set_cursor(len(self.meta['file_list']) - 1)
        self.update_dataset()
        return True

    def delete_record(self):
        if len(self) < 1:
            raise gr.Error(self.components_name.delete_err1)
        current_file = self.data.pop(self.cursor)
        self.set_cursor(self.cursor - 1)
        local_file = os.path.join(self.meta['local_work_dir'],
                                  current_file['relative_path'])
        try:
            os.remove(local_file)
        except Exception:
            print(f'remove file {local_file} error')

        local_src_file = os.path.join(self.meta['local_work_dir'],
                                      current_file['src_relative_path'])
        try:
            os.remove(local_src_file)
        except Exception:
            print(f'remove file {local_src_file} error')

        local_src_mask_file = os.path.join(self.meta['local_work_dir'],
                                      current_file['src_mask_relative_path'])
        try:
            os.remove(local_src_mask_file)
        except Exception:
            print(f'remove file {local_src_mask_file} error')

        if self.cursor >= len(self.meta['file_list']):
            self.set_cursor(0)
        if self.cursor < 0:
            self.set_cursor(len(self) - 1)
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
