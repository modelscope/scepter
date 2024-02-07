# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from __future__ import annotations

import os.path

import gradio as gr
import imagehash

from scepter.modules.utils.file_system import FS
from scepter.studio.preprocess.caption_editor_ui.component_names import \
    DatasetGalleryUIName
from scepter.studio.preprocess.caption_editor_ui.create_dataset_ui import \
    CreateDatasetUI
from scepter.studio.utils.uibase import UIBase


class DatasetGalleryUI(UIBase):
    def __init__(self, cfg, is_debug=False, language='en'):
        self.selected_path = ''
        self.selected_index = -1
        self.selected_index_prev = -1
        self.component_names = DatasetGalleryUIName(language)

    def create_ui(self):
        with gr.Row(variant='panel', visible=False,
                    equal_height=True) as upload_panel:
            with gr.Column():
                self.upload_image = gr.Image(
                    label=self.component_names.upload_image,
                    tool='sketch',
                    type='pil')
            with gr.Column(min_width=80):
                self.caption = gr.Textbox(
                    label=self.component_names.image_caption,
                    placeholder='',
                    value='',
                    lines=5)
                self.upload_button = gr.Button(
                    value=self.component_names.upload_image_btn)

        with gr.Row(visible=False, equal_height=True) as gallery_panel:
            with gr.Row(visible=False):
                # self.gallery_state = gr.Checkbox(label='gallery_state', value=False, visible=False)
                self.cbg_hidden_dataset_filter = gr.CheckboxGroup(
                    label='Dataset Filter')
                self.nb_hidden_dataset_filter_apply = gr.Number(
                    label='Filter Apply', value=-1)
                self.btn_hidden_set_index = gr.Button(
                    elem_id='dataset_tag_editor_btn_hidden_set_index')
                self.nb_hidden_image_index = gr.Number(value=None,
                                                       label='hidden_idx_next')
                self.nb_hidden_image_index_prev = gr.Number(
                    value=None, label='hidden_idx_prev')
                self.gallery_state = gr.Text(label='gallery_state',
                                             value='',
                                             visible=False)

            # with gr.Row(variant='panel', equal_height=True):
            with gr.Column(scale=1):
                self.gl_dataset_images = gr.Gallery(
                    label=self.component_names.dataset_images,
                    elem_id='dataset_tag_editor_dataset_gallery',
                    columns=4)
            with gr.Column(scale=1):
                with gr.Row(equal_height=True):
                    self.info = gr.Text(value='',
                                        label=None,
                                        show_label=False,
                                        interactive=False)
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1, min_width=0):
                        self.ori_caption = gr.Textbox(
                            label=self.component_names.ori_caption,
                            placeholder='',
                            value='',
                            lines=10,
                            autoscroll=False,
                            interactive=False)
                    with gr.Column(scale=1, min_width=0):
                        self.edit_caption = gr.Textbox(
                            label=self.component_names.edit_caption,
                            placeholder='',
                            value='',
                            lines=10,
                            autoscroll=False,
                            interactive=True)
                with gr.Row(equal_height=True):
                    self.modify_button = gr.Button(
                        value=self.component_names.btn_modify)
                with gr.Row(equal_height=True):
                    self.delete_button = gr.Button(
                        value=self.component_names.btn_delete)

        self.upload_panel = upload_panel
        self.gallery_panel = gallery_panel

    def set_callbacks(self, create_dataset: CreateDatasetUI):
        def change_gallery(dataset_name):
            meta_data = create_dataset.meta_dict[dataset_name]
            if len(meta_data['file_list']) > 0:
                cursor = create_dataset.meta_dict[dataset_name]['cursor']
            else:
                cursor = -1
            image_list = [
                os.path.join(meta_data['local_work_dir'], v['relative_path'])
                for v in meta_data['file_list']
            ]
            if cursor >= 0:
                return gr.Gallery(label=dataset_name,
                                  value=image_list,
                                  selected_index=cursor)
            else:
                return gr.Gallery(label=dataset_name,
                                  value=image_list,
                                  selected_index=None)

        self.gallery_state.change(change_gallery,
                                  inputs=[create_dataset.user_data_name],
                                  outputs=[self.gl_dataset_images],
                                  queue=False)

        def select_image(dataset_name, evt: gr.SelectData):
            meta_data = create_dataset.meta_dict[dataset_name]
            if len(meta_data['file_list']) > 0:
                current_info = meta_data['file_list'][evt.index]
                create_dataset.meta_dict[dataset_name]['cursor'] = evt.index
                cursor = evt.index
            else:
                current_info = {'caption': ''}
                cursor = -1

            all_number = len(meta_data['file_list'])
            if cursor >= 0:
                return (gr.Gallery(selected_index=cursor),
                        gr.Textbox(value=current_info['caption']),
                        gr.Textbox(value=current_info['edit_caption']),
                        gr.Text(value=f'{cursor+1}/{all_number}'))
            else:
                return (gr.Gallery(value=[], selected_index=None),
                        gr.Textbox(value=current_info['caption']),
                        gr.Textbox(value=current_info['edit_caption']),
                        gr.Text(value=f'{cursor + 1}/{all_number}'))

        def change_image(dataset_name):
            meta_data = create_dataset.meta_dict[dataset_name]
            cursor = create_dataset.meta_dict[dataset_name]['cursor']
            if cursor >= 0:
                current_info = meta_data['file_list'][cursor]
            else:
                current_info = {'caption': '', 'edit_caption': ''}
            all_number = len(meta_data['file_list'])
            return (gr.Textbox(value=current_info['caption']),
                    gr.Textbox(value=current_info['edit_caption']),
                    gr.Text(value=f'{cursor+1}/{all_number}'))

        def change_caption(dataset_name, edit_caption):
            cursor = create_dataset.meta_dict[dataset_name]['cursor']
            create_dataset.meta_dict[dataset_name]['file_list'][cursor][
                'caption'] = edit_caption
            create_dataset.save_meta(
                create_dataset.meta_dict[dataset_name],
                create_dataset.meta_dict[dataset_name]['work_dir'])
            return gr.Textbox(value=edit_caption)

        self.gl_dataset_images.select(select_image,
                                      inputs=[create_dataset.user_data_name],
                                      outputs=[
                                          self.gl_dataset_images,
                                          self.ori_caption, self.edit_caption,
                                          self.info
                                      ],
                                      queue=False)
        self.gl_dataset_images.change(
            change_image,
            inputs=[create_dataset.user_data_name],
            outputs=[self.ori_caption, self.edit_caption, self.info],
            queue=False)

        self.modify_button.click(
            change_caption,
            inputs=[create_dataset.user_data_name, self.edit_caption],
            outputs=[self.ori_caption],
            queue=False)

        def delete_file(dataset_name):
            cursor = create_dataset.meta_dict[dataset_name]['cursor']
            if len(create_dataset.meta_dict[dataset_name]['file_list']) < 1:
                raise gr.Error(self.component_names.delete_err1)
            current_file = create_dataset.meta_dict[dataset_name][
                'file_list'].pop(cursor)
            local_file = os.path.join(
                create_dataset.meta_dict[dataset_name]['local_work_dir'],
                current_file['relative_path'])
            try:
                os.remove(local_file)
            except Exception:
                print(f'remove file {local_file} error')
            if cursor >= len(
                    create_dataset.meta_dict[dataset_name]['file_list']):
                cursor = 0
            create_dataset.meta_dict[dataset_name]['cursor'] = cursor
            create_dataset.save_meta(
                create_dataset.meta_dict[dataset_name],
                create_dataset.meta_dict[dataset_name]['work_dir'])
            current_info = create_dataset.meta_dict[dataset_name]['file_list'][
                cursor]
            image_list = [
                os.path.join(
                    create_dataset.meta_dict[dataset_name]['local_work_dir'],
                    v['relative_path'])
                for v in create_dataset.meta_dict[dataset_name]['file_list']
            ]
            return (gr.Gallery(value=image_list, selected_index=cursor),
                    gr.Textbox(value=current_info['caption']),
                    gr.Textbox(value=current_info['caption']
                               if current_info['edit_caption'] ==
                               '' else current_info['edit_caption']),
                    gr.Text(value=f'{cursor + 1}/{len(image_list)}'))

        self.delete_button.click(delete_file,
                                 inputs=[create_dataset.user_data_name],
                                 outputs=[
                                     self.gl_dataset_images, self.ori_caption,
                                     self.edit_caption, self.info
                                 ],
                                 queue=False)

        def add_file(dataset_name, upload_image, caption):
            if 'image' in upload_image:
                image = upload_image['image']

            else:
                image = upload_image
            w, h = image.size
            meta = create_dataset.meta_dict[dataset_name]
            local_work_dir = meta['local_work_dir']
            work_dir = meta['work_dir']

            save_folder = os.path.join(local_work_dir, 'images')
            os.makedirs(save_folder, exist_ok=True)

            relative_path = os.path.join('images',
                                         f'{imagehash.phash(image)}.png')
            image_path = os.path.join(work_dir, relative_path)

            local_image_path = os.path.join(local_work_dir, relative_path)
            with FS.put_to(image_path) as local_path:
                image.save(local_path)

            image.save(local_image_path)

            meta['file_list'].append({
                'image_path': image_path,
                'relative_path': relative_path,
                'width': w,
                'height': h,
                'caption': caption,
                'prefix': '',
                'edit_caption': caption
            })

            meta['cursor'] = len(meta['file_list']) - 1
            create_dataset.meta_dict[dataset_name] = meta
            image_list = [
                os.path.join(
                    create_dataset.meta_dict[dataset_name]['local_work_dir'],
                    v['relative_path'])
                for v in create_dataset.meta_dict[dataset_name]['file_list']
            ]
            return (gr.Gallery(value=image_list,
                               selected_index=meta['cursor']),
                    gr.Textbox(value=caption), gr.Textbox(value=caption),
                    gr.Text(value=f"{meta['cursor'] + 1}/{len(image_list)}"))

        self.upload_button.click(add_file,
                                 inputs=[
                                     create_dataset.user_data_name,
                                     self.upload_image, self.caption
                                 ],
                                 outputs=[
                                     self.gl_dataset_images, self.ori_caption,
                                     self.edit_caption, self.info
                                 ],
                                 queue=False)

        def edit_caption_change(dataset_name, edit_caption):
            meta = create_dataset.meta_dict[dataset_name]
            cursor = meta['cursor']
            if cursor >= 0:
                meta['file_list'][cursor]['edit_caption'] = edit_caption

        self.edit_caption.change(
            edit_caption_change,
            inputs=[create_dataset.user_data_name, self.edit_caption],
            queue=False)
