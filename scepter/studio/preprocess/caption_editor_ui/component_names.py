# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
# For dataset manager
class CreateDatasetUIName():
    def __init__(self, language='en'):
        if language == 'en':
            self.dataset_name = 'All Dataset'
            self.btn_create_datasets = 'Create Dataset'
            self.user_data_name = 'Current Dataset Name'
            self.modify_data_button = 'Modify Name'
            self.confirm_data_button = 'Confirm'
            self.refresh_list_button = 'Refresh List'
            self.zip_file = 'Upload Dataset(Zip/Txt)'
            self.zip_file_url = 'Dataset Url'
            self.btn_create_datasets_from_file = 'Create Dataset From File'
            self.user_direction = (
                '### User Guide: \n' +
                f'* {self.btn_create_datasets} button is used to create a new dataset '
                "from scratch. Please make sure to modify the dataset's name and version. After creation, "
                'you can upload images one by one. \n'
                f'* The {self.btn_create_datasets_from_file} button supports creating a new dataset from '
                'a file, currently supporting zip files. For zip files, the format should be consistent'
                " with the one used during training, ensuring it contains an 'images/' folder and a '"
                "train.csv' (which will use the image paths in this file); "
                'The first line is Target:FILE, Prompt, followed by the format of each line: image path, description.'
                'we also surpport the zip of '
                'one level subfolder of images whose format are in jpg, jpeg, png, webp.\n'
                f'* If you have refreshed the page, please click the {self.refresh_list_button} '
                'button to ensure all previously created datasets are visible in the dropdown menu.\n'
                '* ZIP example: https://modelscope.cn/api/v1/models/damo/scepter/repo?Revision=master&FilePath=datasets/3D_example_csv.zip \n'  # noqa
                '* For processing and training with large-scale data, it is recommended to use the command line.'
            )
            # Error or Warning
            self.illegal_data_name_err1 = (
                'The data name is empty or contains illegal '
                "characters ' ' (space) or '/' (slash).")
            self.illegal_data_name_err2 = "Please follow the format '{name}-{version}-{randomstr}'"
            self.illegal_data_name_err3 = "Do not include '.' in the dataset name."
            self.illegal_data_name_err4 = 'Please do not upload files and set dataset links simultaneously.'
            self.illegal_data_name_err5 = 'Invalid dataset name, please switch datasets or create a new one.'
            self.illegal_data_err1 = 'File download failed'
            self.illegal_data_err2 = 'Illegal file format'
            self.illegal_data_err3 = 'File decompression failed, failed to upload to storage!'
            self.modify_data_name_err1 = 'Failed to change dataset name!'
            self.refresh_data_list_info1 = (
                'The dataset name has been changed, '
                'please refresh the list and try again.')
            self.use_link = 'Use File Link'
        elif language == 'zh':
            self.dataset_name = '数据集'
            self.btn_create_datasets = '新建'
            self.user_data_name = '当前数据集名称'
            self.modify_data_button = '修改数据集名称'
            self.confirm_data_button = '确认'
            self.refresh_list_button = '刷新列表'
            self.zip_file = '上传数据集'
            self.zip_file_url = '数据集链接'
            self.btn_create_datasets_from_file = '从文件新建'
            self.user_direction = (
                '### 使用说明 \n' +
                f'* {self.btn_create_datasets} 按钮用于从零新建数据集，请注意修改数据集的name和version，'
                '新建完成后可以逐个上传图片。\n' +
                f'* {self.btn_create_datasets_from_file} 按钮支持从文件中来新建数据集，目前支持zip文件，'
                '需要保证在文件夹外进行打包，并包含images/文件夹和train.csv(会使用该文件中的图片路径)，首行为Target:FILE,Prompt，'
                '其次每行格式为：图片路径,描述；'
                '同时我们也支持图像文件的zip包，格式在jpg、jpeg、png或webp \n' +
                f'* 如果刷新了页面，请点击{self.refresh_list_button} 按钮以确保所有以往创建的数据集在下拉框中可见。\n'
                '* ZIP样例路径：https://modelscope.cn/api/v1/models/damo/scepter/repo?Revision=master&FilePath=datasets/3D_example_csv.zip \n'  # noqa
                '* 对于大规模数据的处理和训练，建议使用命令行形式')
            # Error or Warning
            self.illegal_data_name_err1 = "数据名称为空或包含非法字符' '或者'/'"
            self.illegal_data_name_err2 = '请按照{name}-{version}-{randomstr}'
            self.illegal_data_name_err3 = "数据集名称中不要包含'.'"
            self.illegal_data_name_err4 = '请不要同时上传文件和设置数据集链接'
            self.illegal_data_name_err5 = '不合法的数据集名称，请切换数据集或新建数据集。'
            self.illegal_data_err1 = '文件下载失败'
            self.illegal_data_err2 = '非法的文件格式'
            self.illegal_data_err3 = '文件解压失败，上传存储器失败！'
            self.modify_data_name_err1 = '变更数据集名称失败！'
            self.refresh_data_list_info1 = '该数据集名称发生了变更，请刷新列表试一下。'
            self.use_link = '使用文件链接'


class DatasetGalleryUIName():
    def __init__(self, language='en'):
        if language == 'en':
            self.upload_image = 'Upload Image'
            self.upload_image_btn = 'Upload'
            self.image_caption = 'Image Caption'
            self.dataset_images = 'Dataset Images'
            self.ori_caption = 'Original Caption'
            self.edit_caption = 'Editable Caption'
            self.btn_modify = 'Replace Caption'
            self.btn_delete = 'Delete Image'
            # Error or Warning
            self.delete_err1 = 'Deletion failed, the data is already empty.'
        elif language == 'zh':
            self.upload_image = '上传图片'
            self.upload_image_btn = '上传'
            self.image_caption = '图片描述'
            self.dataset_images = '图片集'
            self.ori_caption = '原始描述'
            self.edit_caption = '编辑描述'
            self.btn_modify = '替换描述'
            self.btn_delete = '删除图片'
            # Error or Warning
            self.delete_err1 = '删除失败，数据已经为空了'


class ExportDatasetUIName():
    def __init__(self, language='en'):
        if language == 'en':
            self.btn_export_zip = 'Download Data'
            self.btn_export_list = 'Export List'
            self.export_file = 'Download Data'
            # Error or Warning
            self.export_err1 = 'The dataset is empty, export is not possible!'
            self.export_zip_err1 = 'Failed to compress the file!'
            self.upload_err1 = 'Failed to compress the file!'
            self.go_to_train = 'Go to train...'
        elif language == 'zh':
            self.btn_export_zip = '导出数据'
            self.btn_export_list = '导出列表'
            self.export_file = '下载数据'
            self.export_err1 = '数据集为空，无法导出!'
            self.export_zip_err1 = '压缩文件失败!'
            self.upload_err1 = '压缩文件失败!'
            self.go_to_train = '去训练...'
