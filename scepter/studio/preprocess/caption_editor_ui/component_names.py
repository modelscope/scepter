# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
# For dataset manager
class CreateDatasetUIName():
    def __init__(self, language='en'):
        if language == 'en':
            self.system_log = '<span style="color: blue;">System Log: {}</span> '
            self.btn_create_datasets = '\U00002795'  # ➕
            self.get_data_name_button = '\U0001F3B2'  # 🎲
            self.new_data_name = (
                'New dataset name, replace "name" and "version" with easy-to-remember identifiers.'
                f'Also get a random name by clicking {self.get_data_name_button}'
            )
            self.modify_data_button = '\U0001F4DD'  # 📝
            self.confirm_data_button = '\U00002714'  # ✔️
            self.cancel_create_button = '\U00002716'  # ✖️
            self.refresh_list_button = '\U0001f504'  # 🔄
            self.delete_dataset_button = '\U0001f5d1'  # 🗑️
            self.dataset_name = (
                f'All Dataset，click{self.btn_create_datasets}to create new dataset，'
                f'click{self.delete_dataset_button}to delete this dataset.')
            self.dataset_type = 'Dataset Type'
            self.dataset_type_name = {
                'scepter_txt2img': 'Text2Image Generation',
                'scepter_img2img': 'Image Edit Generation'
            }
            self.user_data_name = (
                f'Current Dataset Name. Changes of dataset name take '
                f'effect after clicking {self.modify_data_button}')
            self.zip_file = 'Upload Dataset(Zip/Txt)'
            self.zip_file_url = 'Dataset Url'
            self.default_dataset_repo = 'https://www.modelscope.cn/api/v1/models/iic/scepter/'
            self.default_dataset_zip = \
                self.default_dataset_repo + 'repo?Revision=master&FilePath=datasets/3D_example_csv.zip'
            self.default_dataset_name = '3D_example'
            self.btn_create_datasets_from_file = 'Create Dataset From File'
            self.user_direction = (
                '### User Guide: \n' +
                f'* {self.btn_create_datasets} button is used to create a new dataset '
                ". Please make sure to modify the dataset's name and version. After creation, "
                'you can upload images one by one. \n'
                f'* The "{self.btn_create_datasets_from_file}" button supports creating a new dataset from '
                'a file, currently supporting zip files. For zip files, the format should be consistent'
                " with the one used during training, ensuring it contains an 'images/' folder and a '"
                "train.csv' (which will use the image paths in this file); "
                'The first line is Target:FILE, Prompt, followed by the format of each line: image path, description.'
                'we also surpport the zip of '
                'one level subfolder of images whose format are in jpg, jpeg, png, webp.'
                f'The ZIP example is: {self.default_dataset_zip}. \n'  # noqa
                f'* If you have refreshed the page, please click the {self.refresh_list_button} '
                'button to ensure all previously created datasets are visible in the dropdown menu.\n'
                '* For processing and training with large-scale data(for example more than 10K samples), '
                'it is recommended to use the command line to train the model.'
                '* <span style="color: blue;">Please pay attention to the output of '
                'the system logs to help improve operations.</span> \n')
            # Error or Warning
            self.illegal_data_name_err1 = (
                'The data name is empty or contains illegal '
                "characters ' ' (space) or '/' (slash).")
            self.illegal_data_name_err2 = "Please follow the format '{name}-{version}-{randomstr}'"
            self.illegal_data_name_err3 = "Do not include '.' in the dataset name."
            self.illegal_data_name_err4 = 'Please do not upload files and set dataset links simultaneously.'
            self.illegal_data_name_err5 = 'Invalid dataset name, please switch datasets or create a new one.'
            self.illegal_data_err1 = 'File download failed'
            self.illegal_data_err3 = 'File decompression failed, failed to upload to storage!'
            self.delete_data_err1 = 'The example dataset is not allowed delete!'
            self.modify_data_err1 = 'The example dataset is not allowed modify!'
            self.modify_data_name_err1 = 'Failed to change dataset name!'
            self.refresh_data_list_info1 = (
                'The dataset name has been changed, '
                'please refresh the list and try again.')
            self.use_link = 'Use File Link'
        elif language == 'zh':
            self.system_log = '<span style="color: blue;">系统日志: {}</span> '
            self.btn_create_datasets = '\U00002795'  # ➕
            self.get_data_name_button = '\U0001F3B2'  # 🎲
            self.new_data_name = ('新数据集名称，替换"name"和"version"为方便记忆名称.'
                                  f'可以通过点击{self.get_data_name_button}获取随机名称')
            self.modify_data_button = '\U0001F4DD'  # 📝
            self.confirm_data_button = '\U00002714'  # ✔️
            self.cancel_create_button = '\U00002716'  # ✖️
            self.refresh_list_button = '\U0001f504'  # 🔄
            self.delete_dataset_button = '\U0001f5d1'  # 🗑️
            self.dataset_name = (f'数据集，点击{self.btn_create_datasets}新建数据集，'
                                 f'点击{self.delete_dataset_button}删除数据集')
            self.dataset_type = '数据集类型'
            self.dataset_type_name = {
                'scepter_txt2img': '文生图数据',
                'scepter_img2img': '图像编辑（图生图）数据'
            }

            self.user_data_name = f'当前数据集名称，修改后点{self.modify_data_button}生效'
            self.zip_file = '上传数据集'
            self.zip_file_url = '数据集链接'
            self.default_dataset_repo = 'https://www.modelscope.cn/api/v1/models/iic/scepter/'
            self.default_dataset_zip = \
                self.default_dataset_repo + 'repo?Revision=master&FilePath=datasets/3D_example_csv.zip'
            self.default_dataset_name = '3D_example'
            self.btn_create_datasets_from_file = '从文件新建'
            self.user_direction = (
                '### 使用说明 \n' +
                f'* {self.btn_create_datasets} 按钮用于从零新建数据集，请注意修改数据集的name和version，'
                '新建完成后可以逐个上传图片。\n' +
                f'* {self.btn_create_datasets_from_file} 按钮支持从文件中来新建数据集，目前支持zip文件，'
                '需要保证在文件夹外进行打包，并包含images/文件夹和train.csv(会使用该文件中的图片路径)，首行为Target:FILE,Prompt，'
                '其次每行格式为：图片路径,描述；'
                f'同时我们也支持图像文件的zip包，格式在jpg、jpeg、png或webp。数据ZIP样例路径：{self.default_dataset_zip}. \n'
                +
                f'* 如果刷新了页面，请点击{self.refresh_list_button} 按钮以确保所有以往创建的数据集在下拉框中可见。\n'
                '* 对于大规模数据的处理和训练（数据规模大于1万），建议使用命令行形式\n'
                '* <span style="color: blue;">请注意观察系统日志的输出以帮助改进操作。</span> \n')
            # Error or Warning
            self.illegal_data_name_err1 = "数据名称为空或包含非法字符' '或者'/'"
            self.illegal_data_name_err2 = '数据名称应该按照{name}-{version}-{randomstr}'
            self.illegal_data_name_err3 = "数据集名称中不要包含'.'"
            self.illegal_data_name_err4 = '请不要同时上传文件和设置数据集链接'
            self.illegal_data_name_err5 = '不合法的数据集名称，请切换数据集或新建数据集。'
            self.illegal_data_err1 = '文件下载失败'
            self.illegal_data_err3 = '文件解压失败，上传存储器失败！'

            self.delete_data_err1 = '示例数据集不允许删除!'
            self.modify_data_err1 = '示例数据集不允许修改!'

            self.modify_data_name_err1 = '变更数据集名称失败！'
            self.refresh_data_list_info1 = '该数据集名称发生了变更，请刷新列表试一下。'
            self.use_link = '使用文件链接'


class DatasetGalleryUIName():
    def __init__(self, language='en'):
        if language == 'en':
            self.system_log = '<span style="color: blue;">System Log: {}</span> '
            self.upload_image = 'Upload Image'
            self.upload_image_btn = '\U00002714'  # ✔️
            self.cancel_upload_btn = '\U00002716'  # ✖️
            self.image_caption = 'Image Caption'

            self.ori_caption = 'Original Caption'
            self.btn_modify = '\U0001F4DD'  # 📝
            self.btn_delete = '\U0001f5d1'  # 🗑️
            self.btn_add = '\U00002795'  # ➕
            self.dataset_images = f'Original Images，click{self.btn_modify} into editable mode.'
            self.edit_caption = f'Editable Caption，click{self.btn_modify} into editable mode.'
            self.dataset_images = f'Original Images，click{self.btn_modify} into editable mode.'

            self.ori_dataset = 'Original Data Height({}) * Width({}) and Image Format({})'
            self.edit_dataset = 'Editable Data Height({}) * Width({}) and Image Format({})'
            self.upload_image_info = 'Image Information: Height({}) * Width({}) and Image Format({})'

            self.range_mode_name = [
                'Current sample', 'All samples', 'Samples in range'
            ]
            self.samples_range = 'The samples range to be process.'
            self.samples_range_placeholder = (
                '"1,4,6" indicates to process 1st, 4th and 6th sample;'
                '"1-6" indicates to process samples from 1st to 6th.'
                '"1-4,6-8"indicates to process samples from 1st to 4th '
                'and from 6th to 8th.')
            self.set_range_name = 'Samples Range to be edited'
            self.btn_confirm_edit = '\U00002714'  # ✔️
            self.btn_cancel_edit = '\U00002716'  # ✖️
            self.btn_reset_edit = '\U000021BA'  # ↺
            self.confirm_direction = (
                f'click{self.btn_confirm_edit} to apply all changes，'
                f'click{self.btn_reset_edit} to reset edited data，'
                f'click{self.btn_cancel_edit} to out of editing mode.')
            self.preprocess_choices = [
                'Image Preprocess', 'Caption Preprocess'
            ]
            self.image_processor_type = 'Image Preprocessors'
            self.caption_processor_type = 'Caption Preprocessors'
            self.image_preprocess_btn = 'Run'
            self.caption_preprocess_btn = 'Run'
            self.caption_update_mode = 'Caption Update Mode'
            self.caption_update_choices = ['Append', 'Replace']

            self.used_device = 'Used Device'
            self.used_memory = 'Used Memory'
            self.caption_language = "Caption's Language"
            self.advance_setting = 'Generation Setting'
            self.system_prompt = 'System Prompt'
            self.max_new_tokens = 'Max New Tokens'
            self.min_new_tokens = 'Min New Tokens'
            self.num_beams = 'Beams Num'
            self.repetition_penalty = 'Repetition Penalty'
            self.temperature = 'Temperature'

            self.height_ratio = 'Height side scale'
            self.width_ratio = 'Width side scale'
            # Error or Warning

        elif language == 'zh':
            self.system_log = '<span style="color: blue;">系统日志: {}</span> '
            self.upload_image = '上传图片'
            self.upload_image_btn = '\U00002714'  # ✔️
            self.cancel_upload_btn = '\U00002716'  # ✖️
            self.image_caption = '图片描述'

            # self.image_height = '高度'
            # self.image_width = '宽度'
            # self.image_format = '格式'

            self.btn_modify = '\U0001F4DD'  # 📝
            self.dataset_images = f'图片数据，点击{self.btn_modify}进入编辑模式'

            self.btn_delete = '\U0001f5d1'  # 🗑️
            self.btn_add = '\U00002795'  # ➕

            self.ori_caption = f'原始描述，点击{self.btn_modify}进入编辑模式'
            self.edit_caption = '编辑描述'
            self.batch_caption_generate = '处理范围'

            self.ori_dataset = '原始数据 高({}) * 宽({}) 图像格式({})'
            self.edit_dataset = '可编辑数据 高({}) * 宽({}) 图像格式({})'
            self.upload_image_info = '图像信息 高({}) * 宽({})'

            self.range_mode_name = ['当前样本', '全部样本', '指定范围']
            self.samples_range = '处理样本范围'
            self.samples_range_placeholder = (
                '"1,4,6"代表处理第1，4，6个样本;'
                '"1-6" 代表处理从第1个到第6个的全部样本;'
                '"1-4,6-8" 代表处理从第1个到第4个，第6到第8个样本。')
            self.set_range_name = '编辑数据范围'

            self.btn_confirm_edit = '\U00002714'  # ✔️
            self.btn_cancel_edit = '\U00002716'  # ✖️
            self.btn_reset_edit = '\U000021BA'  # ↺
            self.confirm_direction = (f'点击{self.btn_confirm_edit}使所有编辑内容生效，'
                                      f'点击{self.btn_cancel_edit}取消编辑，'
                                      f'点击{self.btn_reset_edit}重置数据，'
                                      f'修改编辑范围可以批量编辑不同范围的数据。')
            self.preprocess_choices = ['图像预处理', '描述生成']
            self.image_processor_type = '图像预处理器'
            self.caption_processor_type = '描述生成器'
            self.image_preprocess_btn = '运行'
            self.caption_preprocess_btn = '运行'
            self.caption_update_mode = '描述更新方式'
            self.caption_update_choices = ['追加', '替换']
            self.used_device = '使用设备'
            self.used_memory = '使用内存'
            self.caption_language = '描述语言'
            self.advance_setting = '生成设置'
            self.system_prompt = '系统提示'
            self.max_new_tokens = '描述最大长度'
            self.min_new_tokens = '描述最小长度'
            self.num_beams = 'Beams数'
            self.repetition_penalty = '重复惩罚'
            self.temperature = '温度系数'
            self.height_ratio = '高度比例'
            self.width_ratio = '宽度比例'
            # Error or Warning


class ExportDatasetUIName():
    def __init__(self, language='en'):
        if language == 'en':
            self.btn_export_zip = 'Download Data'
            self.btn_export_list = 'Export List'
            self.export_file = 'Download Data'
            # Error or Warning
            self.export_err1 = 'The dataset is empty, export is not possible!'

            self.upload_err1 = 'Failed to compress the file!'
            self.go_to_train = 'Go to train...'
        elif language == 'zh':
            self.btn_export_zip = '导出数据'
            self.btn_export_list = '导出列表'
            self.export_file = '下载数据'
            self.export_err1 = '数据集为空，无法导出!'

            self.upload_err1 = '压缩文件失败!'
            self.go_to_train = '去训练...'


class Text2ImageDataCardName():
    def __init__(self, language='en'):
        if language == 'en':
            self.illegal_data_err1 = (
                'The list supports only "," or "#;#" as delimiters. '
                'The four columns represent image path, width, height, '
                'and description, respectively.')
            self.illegal_data_err2 = 'Illegal file format'
            self.illegal_data_err3 = 'File decompression failed, failed to upload to storage!'
            self.illegal_data_err4 = 'Illegal width({}),height({})'
            self.illegal_data_err5 = (
                'The path should not contain "{}". '
                'It should be an OSS path (oss://) or the prefix '
                'can be omitted (xxx/xxx)."')
            self.illegal_data_err6 = 'Image download failed {}'
            self.illegal_data_err7 = 'Image upload failed {}'
            self.delete_err1 = 'Deletion failed, the data is already empty.'
            self.export_zip_err1 = 'Failed to compress the file!'
        elif language == 'zh':
            self.illegal_data_err1 = '列表只支持,或#;#作为分割符，四列分别为图像路径/宽/高/描述'
            self.illegal_data_err2 = '非法的文件格式'
            self.illegal_data_err3 = '文件解压失败，上传存储器失败！'
            self.illegal_data_err4 = '不合法的width({}),height({})'
            self.illegal_data_err5 = '路径不支持{}，应该为oss路径（oss://）或者省略前缀（xxx/xxx）'
            self.illegal_data_err6 = '下载图像失败{}'
            self.illegal_data_err7 = '上传图像失败{}'
            self.delete_err1 = '删除失败，数据已经为空了'
            self.export_zip_err1 = '压缩文件失败!'


class Image2ImageDataCardName():
    def __init__(self, language='en'):
        if language == 'en':
            self.illegal_data_err1 = (
                'The list supports only "," or "#;#" as delimiters. '
                'The four columns represent image path, width, height, '
                'and description, respectively.')
            self.illegal_data_err2 = 'Illegal file format'
            self.illegal_data_err3 = 'File decompression failed, failed to upload to storage!'
            self.illegal_data_err4 = 'Illegal width({}),height({})'
            self.illegal_data_err5 = (
                'The path should not contain "{}". '
                'It should be an OSS path (oss://) or the prefix '
                'can be omitted (xxx/xxx)."')
            self.illegal_data_err6 = 'Image download failed {}'
            self.illegal_data_err7 = 'Image upload failed {}'
            self.delete_err1 = 'Deletion failed, the data is already empty.'
            self.export_zip_err1 = 'Failed to compress the file!'
        elif language == 'zh':
            self.illegal_data_err1 = '列表只支持,或#;#作为分割符，四列分别为图像路径/宽/高/描述'
            self.illegal_data_err2 = '非法的文件格式'
            self.illegal_data_err3 = '文件解压失败，上传存储器失败！'
            self.illegal_data_err4 = '不合法的width({}),height({})'
            self.illegal_data_err5 = '路径不支持{}，应该为oss路径（oss://）或者省略前缀（xxx/xxx）'
            self.illegal_data_err6 = '下载图像失败{}'
            self.illegal_data_err7 = '上传图像失败{}'
            self.delete_err1 = '删除失败，数据已经为空了'
            self.export_zip_err1 = '压缩文件失败!'
