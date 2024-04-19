# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
# For dataset manager
class ModelUIName():
    def __init__(self, language='en'):
        if language == 'en':
            self.output_model_block = 'Model Output'
            self.output_model_name = 'Output Model Name'
            self.output_ckpt_name = 'Output Ckpt Name'
            self.test_prompt = 'Test Prompt'
            self.test_prefix = 'Test Prefix'
            self.test_n_prompt = 'Negative Prompt'
            self.sampler = 'Sampler'
            self.num_inference_steps = 'Sampling Step Length'
            self.inference_num = 'Number of Inferences'
            self.generator_seed = 'Sampling Seed'
            self.tuner_method = 'Tuning Method'
            self.inference_resolution = 'Inference Resolution'
            self.output_image = 'Output Result'
            self.display_button = 'Infer'
            self.extra_model_gtxt = 'Extra Model'
            self.extra_model_gbtn = 'Add Model'
            self.refresh_model_gbtn = 'Refresh Model'
            self.go_to_inference = 'Go to inference'
            self.btn_export_log = 'Export Log'
            self.export_file = 'Log File'
            self.log_block = 'Training Log...'
            self.gallery_block = 'Gallery Log...'
            self.eval_gallery = 'Eval Gallery'
            # Error or Warning
            self.inference_err1 = 'Inference failed, please try again.'
            self.inference_err2 = 'Test prompt is empty.'
            self.model_err3 = "Doesn't surpport this base model"
            self.model_err4 = "This model maybe not finish training, because model doesn't exist."
            self.model_err5 = "Model {} doesn't exist."
            self.training_warn1 = 'No log message util now.'

        elif language == 'zh':
            self.output_model_block = '模型产出'
            self.output_model_name = '产出名称'
            self.output_ckpt_name = '产出检查点名称'
            self.test_prompt = '测试提示词'
            self.test_prefix = '测试前缀'
            self.test_n_prompt = '负向提示词'
            self.sampler = '采样器'
            self.num_inference_steps = '采样步长'
            self.inference_num = '推理数'
            self.generator_seed = '采样种子'
            self.tuner_method = '训练方式'
            self.inference_resolution = '推理分辨率'
            self.output_image = '输出结果'
            self.display_button = '推理'
            self.extra_model_gtxt = '额外模型'
            self.extra_model_gbtn = '添加模型'
            self.refresh_model_gbtn = '刷新模型'
            self.btn_export_log = '导出日志'
            self.export_file = '日志文件'
            self.log_block = '训练日志...'
            self.training_button = '开始训练'
            self.gallery_block = '图像日志...'
            self.eval_gallery = '评测图像'
            # Error or Warning
            self.inference_err1 = '推理失败，请重试。'
            self.inference_err2 = '测试提示词为空。'
            self.model_err3 = '不支持的基础模型'
            self.go_to_inference = '使用模型'
            self.model_err4 = '模型可能没有训练完成或者模型不存在'
            self.model_err5 = '模型{}不存在'
            self.training_warn1 = '暂时没有日志文件；任务启动中或失败！'


class TrainerUIName():
    def __init__(self, language='en'):
        if language == 'en':
            self.user_direction = '''
                ### User Guide
                - Data: Select the template data from Examples, or prepare your custom data
                for upload according to the format of the example dog.zip package.
                - Parameters: You can try modifying the related parameters.
                - Training: Click on [Start Training].
                - Testing: After completing the training, click [Go to inference ].
                - Note: Timeouts may cause the connection to disconnect (an Error may occur).
                 After waiting for the time when the training is likely to be almost complete,
                 refresh the interface and then click [Refresh Model] at the bottom of the page.
                 The trained model should appear in the [Output Model Name] if training was successful;
                 if not, the training may be incomplete or have failed.
                - zip example: https://www.modelscope.cn/api/v1/models/iic/scepter/repo?Revision=master&FilePath=datasets/3D_example_csv.zip
                - For processing and training with large-scale data, it is recommended to use the command line.
            '''  # noqa
            self.data_type_choices = ['Dataset zip', 'MaaS Dataset']
            self.data_type_value = 'Dataset zip'
            self.data_type_name = 'Data Source'
            self.ori_data_name = 'Data Name'
            self.ms_data_name_place_hold = 'Supports MaaS dataset/local/HTTP Zip package'
            self.ms_data_space = 'ModelScope Space'
            self.ms_data_subname = 'MaaS Dataset - Subset'
            self.training_block = 'Training Parameters'
            self.base_model = 'Base Model'
            self.tuner_name = 'Tuner Method'
            self.base_model_revision = 'Model Version Number'
            self.resolution_height = 'Resolution Height'
            self.resolution_width = 'Resolution Width'
            self.resolution_height_max = 'Resolution Height Max'
            self.resolution_width_max = 'Resolution Width Max'
            self.train_epoch = 'Number of Training Epochs'
            self.learning_rate = 'Learning Rate'
            self.save_interval = 'Save Interval'
            self.train_batch_size = 'Training Batch Size'
            self.prompt_prefix = 'Prefix'
            self.replace_keywords = 'Trigger Keywords'
            self.work_name = 'Save Model Name (refresh to get a random value)'
            self.push_to_hub = 'Push to hub'
            self.training_button = 'Start Training'
            self.eval_prompts = 'Eval Prompts'
            self.tuner_param = 'Tuner Parameters'
            self.enable_resolution_bucket = 'Enable Resolution Bucket'
            self.resolution_param = 'Resolution Parameters'
            self.min_bucket_resolution = 'Min Bucket Resolution'
            self.max_bucket_resolution = 'Max Bucket Resolution'
            self.bucket_resolution_steps = 'Bucket Resolution Steps'
            self.bucket_no_upscale = 'Bucket No Upscale'
            # Error or Warning
            self.training_err1 = 'CUDA is unavailable.'
            self.training_err2 = 'Currently insufficient VRAM, training failed!'
            self.training_err3 = 'You need to prepare training data.'
            self.training_err4 = 'Save model name already exists or is None, please regenerate this name.'
            self.training_err5 = 'Training failed.'
            self.training_err6 = "Can't process training data"

        elif language == 'zh':
            self.user_direction = '''
                ### 使用说明
                - 数据: 选择Example的模版数据或可以按照样例中dog.zip包的格式准备自定义数据进行上传
                - 参数: 可尝试进行相关参数的修改
                - 训练: 点击【开始训练】
                - 测试: 完成训练后点击【使用模型】
                - 注意：超时可能导致连接断开(出现Error)，可以等差不多可能训完后，刷新界面再点击页面最后的[刷新模型]，即可在[产出模型名称中]出现已经完成训练的模型，若不存在则没有完成训练或训练失败
                - ZIP样例：https://www.modelscope.cn/api/v1/models/iic/scepter/repo?Revision=master&FilePath=datasets/3D_example_csv.zip
                - 对于大规模数据的处理和训练，建议使用命令行形式
                '''  # noqa
            self.data_type_choices = ['数据集zip', 'MaaS数据集']
            self.data_type_value = '数据集zip'
            self.data_type_name = '数据集来源'
            self.ori_data_name = '数据集名称'
            self.ms_data_name_place_hold = '支持MaaS数据集/本地/Http Zip包'
            self.ms_data_space = 'ModelScope 空间'
            self.ms_data_subname = 'MaaS数据集-子集'
            self.training_block = '训练参数'
            self.base_model = '基础模型'
            self.tuner_name = '微调方法'
            self.base_model_revision = '模型版本号'
            self.resolution_height = '训练高度'
            self.resolution_width = '训练宽度'
            self.resolution_height_max = '最大训练高度'
            self.resolution_width_max = '最大训练宽度'
            self.train_epoch = '训练轮数'
            self.learning_rate = '学习率'
            self.save_interval = '存储间隔'
            self.train_batch_size = '训练批次'
            self.prompt_prefix = '前缀'
            self.replace_keywords = '触发关键词'
            self.work_name = '保存模型名称（刷新获得随机值）'
            self.push_to_hub = '推送魔搭社区'
            self.eval_prompts = '评测文本'
            self.tuner_param = '微调参数'
            self.enable_resolution_bucket = '开启分辨率分桶'
            self.resolution_param = '分辨率参数'
            self.min_bucket_resolution = '最小分桶分辨率'
            self.max_bucket_resolution = '最大分桶分辨率'
            self.bucket_resolution_steps = '分桶分辨率步长'
            self.bucket_no_upscale = '分桶分辨率不做放大'
            # Error or Warning
            self.training_err1 = 'CUDA不可用.'
            self.training_err2 = '目前显存不足，训练失败！'
            self.training_err3 = '您需要准备训练数据'
            self.training_err4 = '保存模型未生成或名称已经存在，请重新生成名称'
            self.training_err5 = '训练失败'
            self.training_err6 = '无法处理的数据'
