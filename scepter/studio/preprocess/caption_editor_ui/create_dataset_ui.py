# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from __future__ import annotations

import copy
import csv
import datetime
import json
import os.path

import gradio as gr
from PIL import Image
from tqdm import tqdm

from scepter.modules.utils.directory import get_md5
from scepter.modules.utils.file_system import FS
from scepter.studio.preprocess.caption_editor_ui.component_names import \
    CreateDatasetUIName
from scepter.studio.utils.uibase import UIBase

refresh_symbol = '\U0001f504'  # 🔄


class CreateDatasetUI(UIBase):
    def __init__(self, cfg, is_debug=False, language='en'):
        self.work_dir = cfg.WORK_DIR
        self.dir_list = FS.walk_dir(self.work_dir, recurse=False)
        self.cache_file = {}
        self.meta_dict = {}
        self.dataset_list = self.load_history()
        self.components_name = CreateDatasetUIName(language)

    def load_meta(self, meta_file):
        dataset_meta = json.load(open(meta_file, 'r'))
        return dataset_meta

    def write_csv(self, file_list, save_csv, data_folder):
        with FS.put_to(save_csv) as local_path:
            print(local_path)
            with open(local_path, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['Target:FILE', 'Prompt'])
                for one_file in file_list:
                    relative_file = one_file['relative_path']
                    if relative_file.startswith('/'):
                        relative_file = relative_file[1:]
                    writer.writerow([relative_file, one_file['caption']])
        return save_csv

    def write_file_list(self, file_list, save_csv):
        with FS.put_to(save_csv) as local_path:
            print(local_path)
            with open(local_path, 'w') as f:
                for one_file in file_list:
                    is_flag, file_path = self.del_prefix(
                        one_file['image_path'], prefix=one_file['prefix'])
                    f.write('{},{},{},{}\n'.format(file_path,
                                                   one_file['width'],
                                                   one_file['height'],
                                                   one_file['caption']))
        return save_csv

    def save_meta(self, meta, dataset_folder):
        meta_file = os.path.join(dataset_folder, 'meta.json')
        save_meta = copy.deepcopy(meta)
        if 'local_work_dir' in meta:
            save_meta.pop('local_work_dir')
        if 'work_dir' in meta:
            save_meta.pop('work_dir')
        with FS.put_to(meta_file) as local_path:
            json.dump(save_meta, open(local_path, 'w'))
        return meta_file

    def construct_meta(self, cursor, file_list, dataset_folder, user_name):
        '''
            {
                "dataset_name": "xxxx",
                "dataset_scale": 100,
                "file_list": "xxxxx", # image_path#;#width#;#height#;#caption
                "update_time": "",
                "create_time": ""
            }
        '''
        train_csv = os.path.join(dataset_folder, 'train.csv')
        train_csv = self.write_csv(file_list, train_csv, dataset_folder)
        save_file_list = os.path.join(dataset_folder, 'file.csv')
        save_file_list = self.write_file_list(file_list, save_file_list)
        meta = {
            'dataset_name': user_name,
            'cursor': cursor,
            'file_list': file_list,
            'train_csv': train_csv,
            'save_file_list': save_file_list
        }
        self.save_meta(meta, dataset_folder)
        return meta

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
                            raise gr.Error('列表只支持,或#;#作为分割符，四列分别为图像路径/宽/高/描述')
                    is_legal, new_path, prefix = self.find_prefix(image_path)
                    try:
                        int(width), int(height)
                    except Exception:
                        raise gr.Error(f'不合法的width({width}),height({height})')

                    if not is_legal:
                        raise gr.Error(
                            f'路径不支持{image_path}，应该为oss路径（oss://）或者省略前缀（xxx/xxx）'
                        )
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
                        caption
                    })
        cache_file_list = []
        for idx, local_path in enumerate(
                FS.get_batch_objects_from(all_remote_list)):
            if local_path is None:
                raise gr.Error(f'下载图像失败{all_remote_list[idx]}')
            _ = FS.put_object_from_local_file(local_path, all_local_list[idx])
            cache_file_list.append(local_path)

        for local_path, target_path, flg in FS.put_batch_objects_to(
                cache_file_list, all_save_list):
            if not flg:
                raise gr.Error(f'上传图像失败{local_path}')
            if os.path.exists(local_path):
                try:
                    os.remove(local_path)
                except Exception:
                    pass

        return file_list

    def load_from_zip(self, save_file, data_folder, local_dataset_folder):
        with FS.get_from(save_file) as local_path:
            res = os.popen(
                f"unzip -o '{local_path}' -d '{local_dataset_folder}'")
            res = res.readlines()
        if not os.path.exists(local_dataset_folder):
            raise gr.Error(f'解压{save_file}失败{str(res)}')
        file_folder = None
        train_list = None
        hit_dir = None
        raw_list = []
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
                        raw_list.append(one_s_dir)
            else:
                if (one_dir.endswith('.jpg') or one_dir.endswith('.jpeg')
                        or one_dir.endswith('.png')
                        or one_dir.endswith('.webp')):
                    raw_list.append(one_dir)

        if file_folder is None and len(raw_list) < 1:
            raise gr.Error(
                "images folder or train.csv doesn't exists, or nothing exists in your zip"
            )
        new_file_folder = f'{local_dataset_folder}/images'
        os.makedirs(new_file_folder, exist_ok=True)
        if file_folder is not None:
            _ = FS.get_dir_to_local_dir(file_folder, new_file_folder)
        elif len(raw_list) > 0:
            raw_list = list(set(raw_list))
            for img_id, cur_image in enumerate(raw_list):
                _, surfix = os.path.splitext(cur_image)
                try:
                    os.rename(
                        os.path.abspath(cur_image),
                        f'{new_file_folder}/{get_md5(cur_image)}{surfix}')
                    raw_list[img_id] = [
                        os.path.join('images',
                                     f'{get_md5(cur_image)}{surfix}'),
                        cur_image.split('/')[-1]
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
        try:
            res = os.popen(f"rm -rf '{hit_dir}/images/*'")
            _ = res.readlines()
            res = os.popen(f"rm -rf '{hit_dir}'")
            _ = res.readlines()
        except Exception:
            pass
        file_list = self.load_train_csv(new_train_list, data_folder)
        return file_list

    def get_image_meta(self, image):
        img = Image.open(image)
        return img.size

    def load_train_csv(self, file_path, data_folder):
        base_folder = os.path.dirname(file_path)
        file_list = []
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                image_path, prompt = row[0], row[1]
                if image_path == 'Target:FILE':
                    continue
                local_image_path = os.path.join(base_folder, image_path)
                w, h = self.get_image_meta(local_image_path)
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
                    prompt
                })
        return file_list

    def find_prefix(self, file_path):
        for k in FS._prefix_to_clients.keys():
            if file_path.startswith(k):
                return True, file_path, ''
            elif FS.exists(os.path.join(k, file_path)):
                return True, os.path.join(k, file_path), ''
            elif FS.exists(os.path.join(k, 'datasets', file_path)):
                return True, os.path.join(k, 'datasets', file_path), 'datasets'
        return False, None, None

    def del_prefix(self, file_path, prefix=''):
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

    def load_history(self):
        dataset_list = []
        for one_dir in self.dir_list:
            if FS.isdir(one_dir):
                meta_file = os.path.join(one_dir, 'meta.json')
                if FS.exists(meta_file):
                    local_dataset_folder, _ = FS.map_to_local(one_dir)
                    local_dataset_folder = FS.get_dir_to_local_dir(
                        one_dir, local_dataset_folder, multi_thread=True)
                    meta_data = self.load_meta(
                        os.path.join(local_dataset_folder, 'meta.json'))
                    meta_data['local_work_dir'] = local_dataset_folder
                    meta_data['work_dir'] = one_dir
                    dataset_list.append(meta_data['dataset_name'])
                    self.meta_dict[meta_data['dataset_name']] = meta_data
        return dataset_list

    def create_ui(self):
        with gr.Box():
            gr.Markdown(self.components_name.user_direction)
        with gr.Box():
            with gr.Row():
                with gr.Column(scale=1, min_width=0):
                    self.dataset_name = gr.Dropdown(
                        label=self.components_name.dataset_name,
                        choices=self.dataset_list,
                        interactive=True)
                with gr.Column(scale=1, min_width=0):
                    self.refresh_dataset_name = gr.Button(
                        value=self.components_name.refresh_list_button)
                    self.btn_create_datasets = gr.Button(
                        value=self.components_name.btn_create_datasets)
                    self.btn_create_datasets_from_file = gr.Button(
                        value=self.components_name.
                        btn_create_datasets_from_file)
                    self.panel_state = gr.Checkbox(label='panel_state',
                                                   value=False,
                                                   visible=False)
                with gr.Column(scale=2, min_width=0):
                    with gr.Row(equal_height=True):
                        with gr.Column(visible=False, min_width=0) as panel:
                            self.user_data_name = gr.Text(
                                label=self.components_name.user_data_name,
                                value='',
                                interactive=True)
                            self.user_data_name_state = gr.State(value='')
                            self.create_mode = gr.State(value=0)
                        with gr.Column(visible=False,
                                       min_width=0) as file_panel:
                            self.use_link = gr.Checkbox(
                                label=self.components_name.use_link,
                                value=False,
                                visible=False)
                            self.file_path = gr.File(
                                label=self.components_name.zip_file,
                                min_width=0,
                                file_types=['.zip', '.txt', '.csv'],
                                visible=False)

                            self.file_path_url = gr.Text(
                                label=self.components_name.zip_file_url,
                                value='',
                                visible=False)
                        with gr.Column(visible=False,
                                       min_width=0) as btn_panel:
                            self.random_data_button = gr.Button(
                                value=refresh_symbol)
                            self.confirm_data_button = gr.Button(
                                value=self.components_name.confirm_data_button)
                        with gr.Column(visible=False,
                                       min_width=0) as modify_panel:
                            self.modify_data_button = gr.Button(
                                value=self.components_name.modify_data_button)

        self.dataset_panel = panel
        self.btn_panel = btn_panel
        self.file_panel = file_panel
        self.modify_panel = modify_panel

    def set_callbacks(self, gallery_dataset, export_dataset):
        def show_dataset_panel():
            return (gr.Column(visible=False), gr.Column(visible=True),
                    gr.Column(visible=True),
                    gr.Checkbox(value=False, visible=False),
                    gr.Text(value=get_random_dataset_name(),
                            interactive=True), 1)

        def show_file_panel():
            return (gr.Column(visible=True), gr.Column(visible=True),
                    gr.Column(visible=True),
                    gr.Checkbox(value=False, visible=False),
                    gr.Text(value=get_random_dataset_name(),
                            interactive=True), gr.File(value=None,
                                                       visible=True),
                    gr.Text(value='', visible=False), 2,
                    gr.Checkbox(value=False, visible=True))

        def get_random_dataset_name():
            data_name = 'name-version-{0:%Y%m%d_%H_%M_%S}'.format(
                datetime.datetime.now())
            return data_name

        def refresh():
            return gr.Dropdown(value=self.dataset_list[-1]
                               if len(self.dataset_list) > 0 else '',
                               choices=self.dataset_list)

        self.refresh_dataset_name.click(refresh,
                                        outputs=[self.dataset_name],
                                        queue=False)

        def confirm_create_dataset(user_name, create_mode, file_url, file_path,
                                   panel_state):
            if user_name.strip() == '' or ' ' in user_name or '/' in user_name:
                raise gr.Error(self.components_name.illegal_data_name_err1)

            if len(user_name.split('-')) < 3:
                raise gr.Error(self.components_name.illegal_data_name_err2)

            if '.' in user_name:
                raise gr.Error(self.components_name.illegal_data_name_err3)

            if not file_url.strip() == '' and file_path is not None:
                raise gr.Error(self.components_name.illegal_data_name_err4)
            if create_mode == 3 and not file_url.strip() == '':
                file_name, surfix = os.path.splitext(file_url.split('?')[0])
                save_file = os.path.join(self.work_dir, f'{user_name}{surfix}')
                local_path, _ = FS.map_to_local(save_file)
                res = os.popen(f"wget -c '{file_url}' -O '{local_path}'")
                res.readlines()
                FS.put_object_from_local_file(local_path, save_file)
                if not FS.exists(save_file):
                    raise gr.Error(
                        f'{self.components_name.illegal_data_err1} {str(res)}')
            elif create_mode == 2 and file_path is not None and file_path.name:
                self.cache_file[user_name] = {
                    'file_name': file_path.name,
                    'surfix': os.path.splitext(file_path.name)[-1]
                }
                cache_file = self.cache_file.pop(user_name)
                surfix = cache_file['surfix']
                ori_file = cache_file['file_name']
                save_file = os.path.join(self.work_dir, f'{user_name}{surfix}')
                with FS.put_to(save_file) as local_path:
                    res = os.popen(f"cp '{ori_file}' '{local_path}'")
                    res = res.readlines()
                if not FS.exists(save_file):
                    raise gr.Error(
                        f'{self.components_name.illegal_data_err1}{str(res)}')
            else:
                surfix = None
            # untar file or create blank dataset
            dataset_folder = os.path.join(self.work_dir, user_name)
            local_dataset_folder, _ = FS.map_to_local(dataset_folder)
            if surfix == '.zip':
                file_list = self.load_from_zip(save_file, dataset_folder,
                                               local_dataset_folder)
            elif surfix in ['.txt', '.csv']:
                file_list = self.load_from_list(save_file, dataset_folder,
                                                local_dataset_folder)
            elif surfix is None:
                file_list = []
            else:
                raise gr.Error(
                    f'{self.components_name.illegal_data_err2} {surfix}')
            is_flag = FS.put_dir_from_local_dir(local_dataset_folder,
                                                dataset_folder,
                                                multi_thread=True)
            if not is_flag:
                raise gr.Error(f'{self.components_name.illegal_data_err3}')

            cursor = 0 if len(file_list) > 0 else -1
            meta = self.construct_meta(cursor, file_list, dataset_folder,
                                       user_name)

            meta['local_work_dir'] = local_dataset_folder
            meta['work_dir'] = dataset_folder

            self.meta_dict[meta['dataset_name']] = meta
            if meta['dataset_name'] not in self.dataset_list:
                self.dataset_list.append(meta['dataset_name'])
            return (
                gr.Checkbox(value=True, visible=False),
                gr.Dropdown(value=user_name, choices=self.dataset_list),
            )

        def clear_file():
            return gr.Text(visible=True)

        # Click Create
        self.btn_create_datasets.click(show_dataset_panel, [], [
            self.file_panel, self.dataset_panel, self.btn_panel,
            self.panel_state, self.user_data_name, self.create_mode
        ],
                                       queue=False)

        self.btn_create_datasets_from_file.click(show_file_panel, [], [
            self.file_panel, self.dataset_panel, self.btn_panel,
            self.panel_state, self.user_data_name, self.file_path,
            self.file_path_url, self.create_mode, self.use_link
        ],
                                                 queue=False)

        def use_link_change(use_link):
            if use_link:
                create_mode = 3
                return (gr.File(value=None, visible=False),
                        gr.Text(value='', visible=True), create_mode)
            else:
                create_mode = 2
                return (gr.File(value=None, visible=True),
                        gr.Text(value='', visible=False), create_mode)

        self.use_link.change(
            use_link_change, [self.use_link],
            [self.file_path, self.file_path_url, self.create_mode])
        # Click Refresh
        self.random_data_button.click(get_random_dataset_name, [],
                                      [self.user_data_name],
                                      queue=False)

        self.file_path.clear(clear_file,
                             outputs=[self.file_path_url],
                             queue=False)

        # Click Confirm
        self.confirm_data_button.click(confirm_create_dataset, [
            self.user_data_name, self.create_mode, self.file_path_url,
            self.file_path, self.panel_state
        ], [self.panel_state, self.dataset_name],
                                       queue=True)

        def show_edit_panel(panel_state, data_name):
            if panel_state:
                return (gr.Row(visible=True), gr.Row(visible=True),
                        gr.Row(visible=True), gr.Column(visible=True),
                        gr.Column(visible=False), gr.Column(visible=False),
                        data_name)
            else:
                return (gr.Row(visible=False), gr.Row(visible=False),
                        gr.Row(visible=False), gr.Column(visible=False),
                        gr.Column(), gr.Column(), data_name)

        self.panel_state.change(
            show_edit_panel, [self.panel_state, self.dataset_name], [
                gallery_dataset.gallery_panel, gallery_dataset.upload_panel,
                export_dataset.export_panel, self.modify_panel,
                self.file_panel, self.btn_panel, self.user_data_name_state
            ],
            queue=False)

        def modify_data_name(user_name, prev_data_name):
            print(
                f'Current file name {prev_data_name}, new file name {user_name}.'
            )
            if user_name.strip() == '' or ' ' in user_name or '/' in user_name:
                raise gr.Error(self.components_name.illegal_data_name_err1)
            if len(user_name.split('-')) < 3:
                raise gr.Error(self.components_name.illegal_data_name_err2)
            if '.' in user_name:
                raise gr.Error(self.components_name.illegal_data_name_err3)
            if user_name != prev_data_name:
                if prev_data_name in self.meta_dict:
                    ori_meta = self.meta_dict[prev_data_name]
                    dataset_folder = os.path.join(self.work_dir, user_name)
                    local_dataset_folder, _ = FS.map_to_local(dataset_folder)
                    os.makedirs(local_dataset_folder, exist_ok=True)
                    is_flag = FS.get_dir_to_local_dir(ori_meta['work_dir'],
                                                      local_dataset_folder,
                                                      multi_thread=True)
                    file_list = ori_meta['file_list']
                    is_flag = FS.put_dir_from_local_dir(local_dataset_folder,
                                                        dataset_folder,
                                                        multi_thread=True)
                    if not is_flag:
                        raise gr.Error(self.components_name.illegal_data_err3)
                    is_flag = FS.put_dir_from_local_dir(local_dataset_folder,
                                                        dataset_folder,
                                                        multi_thread=True)
                    if not is_flag:
                        raise gr.Error(self.components_name.illegal_data_err3)
                    cursor = ori_meta['cursor']
                    meta = self.construct_meta(cursor, file_list,
                                               dataset_folder, user_name)
                    meta['local_work_dir'] = local_dataset_folder
                    meta['work_dir'] = dataset_folder

                    if prev_data_name in self.dataset_list:
                        self.dataset_list.remove(prev_data_name)
                        self.dataset_list.append(user_name)
                    self.meta_dict.pop(prev_data_name)
                    self.meta_dict[user_name] = meta
                    _ = FS.delete_object(
                        os.path.join(ori_meta['work_dir'], 'meta.json'))
                    _ = FS.delete_object(
                        os.path.join(ori_meta['local_work_dir'], 'meta.json'))
                else:
                    raise gr.Error(self.components_name.modify_data_name_err1)
                return user_name, gr.Dropdown(
                    choices=self.dataset_list,
                    value=user_name,
                    select_index=len(self.dataset_list) - 1)
            else:
                return user_name, gr.Dropdown()

        self.modify_data_button.click(
            modify_data_name,
            inputs=[self.user_data_name, self.user_data_name_state],
            outputs=[self.user_data_name_state, self.dataset_name],
            queue=False)

        def dataset_change(user_name):
            if user_name is None or user_name == '':
                raise gr.Error(self.components_name.illegal_data_name_err5 +
                               f'{user_name}')
            if user_name not in self.meta_dict:
                raise gr.Error(self.components_name.refresh_data_list_info1)
            return (gr.Column(visible=True), gr.Column(visible=False),
                    gr.Checkbox(value=True, visible=False),
                    gr.Text(value=user_name,
                            interactive=True), gr.Text(value=user_name))

        self.dataset_name.change(dataset_change,
                                 inputs=[self.dataset_name],
                                 outputs=[
                                     self.dataset_panel, self.file_panel,
                                     self.panel_state, self.user_data_name,
                                     gallery_dataset.gallery_state
                                 ],
                                 queue=False)
