# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import csv
import os
import shutil
import time
import decord
from tqdm import tqdm
import gradio as gr

from scepter.modules.utils.directory import get_md5
from scepter.modules.utils.file_system import FS
from scepter.studio.preprocess.caption_editor_ui.component_names import \
    Text2VideoDataCardName
from scepter.studio.preprocess.utils.data_card import (BaseDataCard, find_prefix)


class Text2VideoDataCard(BaseDataCard):
    def __init__(self,
                 dataset_folder,
                 dataset_name=None,
                 src_file=None,
                 surfix=None,
                 user_name='admin',
                 language='en'
                 ):
        super().__init__(dataset_folder,
                         dataset_name=dataset_name,
                         user_name=user_name)
        self.meta['task_type'] = 'txt2vid'
        self.components_name = Text2VideoDataCardName(language)
        if self.new_dataset:
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

    def apply_changes(self):
        edit_index_list = self.edit_list
        for index in edit_index_list:
            one_data = self.data[index]
            self.data[index]['caption'] = one_data['edit_caption']

        self.update_dataset()
        return True, ''

    def load_from_list(self, save_file, dataset_folder, local_dataset_folder):
        file_list = []
        videos_folder = os.path.join(local_dataset_folder, 'videos')
        os.makedirs(videos_folder, exist_ok=True)
        with FS.get_from(save_file) as local_path:
            with open(local_path, 'r') as f:
                for line in tqdm(f):
                    line = line.strip()
                    if line == '':
                        continue
                    try:
                        src_video_path, caption = line.split(
                            '#;#', 1)
                    except Exception:
                        try:
                            src_video_path, caption = line.split(
                                ',', 1)
                        except Exception:
                            raise gr.Error(
                                self.components_name.illegal_data_err1)
                    relative_path = os.path.join(
                        'videos', f'{get_md5(src_video_path)[:18]}_{int(time.time())}.mp4')
                    video_path = os.path.join(dataset_folder, relative_path)
                    FS.get_from(src_video_path, local_path=video_path)
                    is_legal, new_path, prefix = find_prefix(src_video_path)
                    w, h, fps, duration = self.get_video_meta(video_path)

                    file_list.append({
                        'video_path':
                        video_path,
                        'relative_path':
                        relative_path,
                        'width':
                        w,
                        'height':
                        h,
                        'fps':
                        fps,
                        'duration':
                        duration,
                        'caption':
                        caption,
                        'edit_caption':
                        caption,
                        'prefix':
                        prefix
                    })
        return file_list

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
                if one_dir.endswith('videos') or one_dir.endswith('videos/'):
                    file_folder = one_dir
                    hit_dir = one_dir
                else:
                    sub_dir = FS.walk_dir(one_dir)
                    for one_s_dir in sub_dir:
                        if FS.isdir(one_s_dir) and one_s_dir.split(
                                one_dir)[1].replace('/', '') == 'videos':
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
                "video doesn't exist, or nothing exists in your zip")

        if train_list is None:
            raise gr.Error("pair list doesn't exist")
        new_file_folder = f'{local_dataset_folder}/videos'
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
                res = os.popen(f"rm -rf '{hit_dir}/videos/*'")
                _ = res.readlines()
                res = os.popen(f"rm -rf '{hit_dir}'")
                _ = res.readlines()

            except Exception:
                pass
        file_list = self.load_train_file(new_train_list)
        # remove unused data
        for one_dir in FS.walk_dir(local_dataset_folder):
            if 'videos' in one_dir or one_dir.endswith(
                    'file.csv') or one_dir.endswith('train.csv'):
                continue
            os.system(f'rm -rf {one_dir}')
        return file_list

    def load_train_file(self, file_path):
        base_folder = os.path.dirname(file_path)
        file_list = []
        video_set = set()

        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) == 2:
                   src_video_path, prompt = row[0], row[1]
                else:
                    return gr.Error(self.components_name.illegal_data_err2)
                if src_video_path == 'Target:FILE':
                    continue

                local_video_path = os.path.join(base_folder, src_video_path)
                w, h, fps, duration = self.get_video_meta(local_video_path)
                if src_video_path in video_set:
                    src_video_path, surfix = os.path.splitext(src_video_path)
                    src_video_path = f'{src_video_path}_{int(time.time() * 100)}{surfix}'
                    new_local_video_path = os.path.join(
                        base_folder, src_video_path)
                    self.copy_video(src_video_path, new_local_video_path)
                video_set.add(src_video_path)

                file_list.append({
                    'video_path':
                        local_video_path,
                    'relative_path':
                        src_video_path,
                    'width':
                        w,
                    'height':
                        h,
                    'fps':
                        fps,
                    "duration":
                        duration,
                    'caption':
                        prompt,
                    'edit_caption':
                        prompt,
                    'prefix':
                    ''
                })
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
                writer.writerow([relative_file, one_file['caption'].
                                strip().replace("\n", "")])
        FS.put_object_from_local_file(self.local_train_file, self.train_file)

    def write_data_file(self):
        file_list = self.meta['file_list']
        with open(self.local_save_file_list, 'w') as f:
            for one_file in file_list:
                f.write('{}#;#{}#;#{}#;#{}\n'.format(one_file['relative_path'],
                                                     one_file['width'],
                                                     one_file['height'],
                                                     one_file['caption'].strip().replace("\n", ""))
                        )
        FS.put_object_from_local_file(self.local_save_file_list,
                                      self.save_file_list)

    def add_record(self, video, caption, **kwargs):
        local_work_dir = self.meta['local_work_dir']
        work_dir = self.meta['work_dir']

        save_folder = os.path.join(local_work_dir, 'videos')
        os.makedirs(save_folder, exist_ok=True)
        w, h, fps, duration = self.get_video_meta(video)

        relative_path = os.path.join(
            'videos', f'{get_md5(video)[:18]}_{int(time.time())}.mp4')
        video_path = os.path.join(work_dir, relative_path)
        local_video_path = os.path.join(local_work_dir, relative_path)
        self.copy_video(video, local_video_path)

        self.data.append({
            'video_path': video_path,
            'relative_path': relative_path,
            'width': w,
            'height': h,
            'fps': fps,
            'duration': duration,
            'caption': caption,
            'edit_caption': caption,
            'prefix': ''
        })

        self.set_cursor(len(self.meta['file_list']) - 1)
        self.update_dataset()
        return True

    def copy_video(self, source_path, target_path):
        if not os.path.isfile(source_path):
            raise gr.Error('Video path not exist.')
        try:
            shutil.copy2(source_path, target_path)
        except Exception as e:
            raise gr.Error(str(e))

    def get_video_meta(self, video):
        video_reader = decord.VideoReader(video)
        w = video_reader[0].shape[1]
        h = video_reader[0].shape[0]
        fps = video_reader.get_avg_fps()
        video_length = len(video_reader)
        duration = video_length / fps

        return w, h, fps, duration

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
            f"&& cp -rf videos '{self.dataset_name}/videos' "
            f"&& cp -rf train.csv '{self.dataset_name}/train.csv' "
            f"&& zip -r '{os.path.abspath(local_zip)}' '{self.dataset_name}'/* "
            f"&& rm -rf '{self.dataset_name}'")
        print(res.readlines())
        FS.put_object_from_local_file(local_zip, zip_path)
        if not FS.exists(zip_path):
            raise gr.Error(self.components_name.export_zip_err1)
        return local_zip
