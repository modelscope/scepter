# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
# For dataset manager
from scepter.modules.utils.directory import get_md5
from scepter.modules.utils.file_system import FS


def download_image(image):
    if image is not None:
        name = get_md5(image)
        local_path = FS.get_from(image, f'/tmp/gradio/scepter_examples/{name}')
        return local_path
    else:
        return image


class InferenceUIName():
    def __init__(self, language='en'):
        if language == 'en':
            self.advance_block_name = 'Advance Setting'
            self.check_box_for_setting = [
                'Use Mantra', 'Use Tuners', 'Use Controller', 'LAR-Gen',
                'StyleBooth'
            ]
            self.diffusion_paras = 'Generation Setting'
            self.mantra_paras = 'Mantra Book'
            self.tuner_paras = 'Tuners'
            self.control_paras = 'Controlable Generation'
            self.refiner_paras = 'Refiner Setting'
            self.largen_paras = 'LAR-Gen'
            self.stylebooth_paras = 'StyleBooth'
        elif language == 'zh':
            self.advance_block_name = '生成选项'
            self.check_box_for_setting = [
                '使用咒语', '使用微调', '使用控制', 'LAR-Gen', 'StyleBooth'
            ]
            self.diffusion_paras = '生成参数设置'
            self.mantra_paras = '咒语书'
            self.tuner_paras = '微调模型'
            self.control_paras = '可控生成'
            self.refiner_paras = 'Refine设置'
            self.largen_paras = 'LAR-Gen'
            self.stylebooth_paras = 'StyleBooth'


class ModelManageUIName():
    def __init__(self, language='en'):
        if language == 'en':
            self.model_block_name = 'Model Management'
            self.postprocess_model_name = 'Refiners and Tuners'
            self.diffusion_model = 'Unet'
            self.first_stage_model = 'Vae'
            self.cond_stage_model = 'Condition Model'
            self.refine_diffusion_model = 'Refine Unet'
            self.refine_cond_model = 'Refine Condition Model'
            self.load_lora_tuner = 'Load Lora Tuner'
            self.load_swift_tuner = 'Load swift Tuner'
        elif language == 'zh':
            self.model_block_name = '模型管理'
            self.postprocess_model_name = 'Refiners and Tuners'
            self.diffusion_model = 'Unet'
            self.first_stage_model = 'Vae'
            self.cond_stage_model = 'Condition Model'
            self.refine_diffusion_model = 'Refine Unet'
            self.refine_cond_model = 'Refine Condition Model'
            self.load_lora_tuner = '加载 Lora 微调模型'
            self.load_swift_tuner = '加载 Swift 微调模型'


class GalleryUIName():
    def __init__(self, language='en'):
        if language == 'en':
            self.gallery_block_name = "Model's Output"
            self.gallery_diffusion_output = 'Diffusion Output'
            self.gallery_before_refine_output = 'Before Refine Output'
            self.prompt_input = 'Type Prompt Here'
            self.generate = 'Generate'
            self.control_err1 = 'The conditional image does not exist'
        elif language == 'zh':
            self.gallery_block_name = '模型的输出'
            self.gallery_diffusion_output = '扩散输出'
            self.gallery_before_refine_output = '精炼前输出'
            self.prompt_input = '在此输入提示'
            self.generate = '生成'
            self.control_err1 = '条件图片不存在'


class DiffusionUIName():
    def __init__(self, language='en'):
        if language == 'en':

            self.sample = 'Sampler'
            self.sample_steps = 'Sample Steps'
            self.image_number = 'Images Number'
            self.resolutions_height = 'Output Height'
            self.resolutions_width = 'Output Width'
            self.negative_prompt = 'Negative Prompt'
            self.negative_prompt_placeholder = 'Type Prompt Here'
            self.negative_prompt_description = 'Describing what you do not want to see.'
            self.prompt_prefix = 'Prompt Prefix'
            self.guide_scale = 'Guide Scale For Uncondition'
            self.guide_rescale = 'Guide ReScale'
            self.discretization = 'Discretization'
            self.random_seed = 'Use Random Seed'
            self.seed = 'Used Seed'
            self.example_block_name = 'Prompt Examples'
            self.examples = [
                'dream dandelion', 'Mount Everest', 'a boy wearing a jacket',
                'Spring, Birds, Cawing, Branches',
                'Cyberpunk, Maiden, Heavy Machinery', 'A Frog'
            ]

        elif language == 'zh':

            self.sample = '采样器'
            self.sample_steps = '采样步数'
            self.image_number = '图片数量'
            self.resolutions_height = '输出高度'
            self.resolutions_width = '输出宽度'
            self.negative_prompt = '负向提示'
            self.negative_prompt_placeholder = '在此输入提示'
            self.negative_prompt_description = '描述您不希望看到的内容。'
            self.prompt_prefix = '提示前缀词'
            self.guide_scale = '条件引导比例'
            self.guide_rescale = '引导缩放'
            self.discretization = '离散化'
            self.random_seed = '使用随机种子'
            self.seed = '使用的种子'
            self.example_block_name = '提示词样例'
            self.examples = [
                'dream dandelion', 'Mount Everest', 'a boy wearing a jacket',
                'Spring, Birds, Cawing, Branches',
                'Cyberpunk, Maiden, Heavy Machinery', 'A Frog'
            ]


class MantraUIName():
    def __init__(self, language='en'):
        if language == 'en':

            self.mantra_styles = 'Mantra Styles'
            self.style_name = 'Mantra Name'
            self.style_prompt = 'Mantra Prompt'
            self.style_source = 'Mantra Source'
            self.style_negative_prompt = 'Mantra Negative Prompt'
            self.select_style = 'Selected Mantra'
            self.style_desc = 'Mantra Description'
            self.style_template = 'Mantra Prompt Template'
            self.style_negative_template = 'Mantra Negative Prompt Template'
            self.style_example = 'Mantra Results Example'
            self.style_example_prompt = 'Mantra Example Prompt'
            self.example_block_name = 'Examples'
            self.examples = [
                [['Adorable 3D Character'], 'a girl'],
                [['Watercolor 2'], 'a single flower'],
                [['Action Figure'],
                 'a close up of a small rabbit wearing a hat and scarf']
            ]

        elif language == 'zh':
            self.mantra_styles = '咒语风格'
            self.style_name = '咒语名称'
            self.style_prompt = '咒语提示'
            self.style_source = '咒语来源'
            self.style_negative_prompt = '咒语负向提示'
            self.select_style = '选定的咒语'
            self.style_desc = '咒语描述'
            self.style_template = '咒语提示模板'
            self.style_negative_template = '咒语负向提示模板'
            self.style_example = '咒语示例图'
            self.style_example_prompt = '咒语示例提示词'
            self.example_block_name = '样例'
            self.examples = [
                [['可爱的3D角色'], 'a girl'], [['水彩'], 'a single flower'],
                [['动作人偶'],
                 'a close up of a small rabbit wearing a hat and scarf']
            ]


class RefinerUIName():
    def __init__(self, language='en'):
        if language == 'en':

            self.refine_sample = 'Refine Sampler'
            self.refine_discretization = 'Refine Discretization'
            self.refine_guide_scale = 'Refine Guide Scale For Uncondition'
            self.refine_guide_rescale = 'Guide ReScale'
            self.refine_strength = 'Refine Strength'
            self.refine_strength_description = 'Use to control how many steps used to refine.'
            self.refine_diffusion_model = 'Refiner'
            self.refine_cond_model = 'Refine Cond'

        elif language == 'zh':

            self.refine_sample = 'Refine采样器'
            self.refine_discretization = 'Refine离散化'
            self.refine_guide_scale = 'Refine条件引导比例'
            self.refine_guide_rescale = 'Refine引导缩放'
            self.refine_strength = 'Refine强度'
            self.refine_strength_description = '用于控制用于Refine步数。'
            self.refine_diffusion_model = 'Refiner'
            self.refine_cond_model = 'Refine Cond'


class TunerUIName():
    def __init__(self, language='en'):
        if language == 'en':

            self.tuner_model = 'Tuner'
            self.tuner_name = 'Tuner Name'
            self.tuner_type = 'Tuner Type'
            self.tuner_desc = 'Tuner Description'
            self.tuner_example = 'Results Example'
            self.tuner_prompt_example = 'Prompt Example'
            self.base_model = 'Base Model Name'
            self.custom_tuner_model = 'Customized Model'
            self.advance_block_name = 'Advance Setting'
            self.tuner_scale = 'Tuner Scale'
            self.example_block_name = 'Examples'
            self.examples = [[['Pencil Sketch Drawing'], 'a girl in a jacket'],
                             [['Flat 2D Art'], 'a cat']]
            self.save_button = 'Save'

        elif language == 'zh':
            self.tuner_model = '微调模型'
            self.tuner_name = '微调模型名'
            self.tuner_type = '微调模型类型'
            self.tuner_desc = '微调模型描述'
            self.tuner_example = '示例结果'
            self.tuner_prompt_example = '示例提示词'
            self.base_model = '基础模型'
            self.custom_tuner_model = '自定义模型'
            self.advance_block_name = '高级设置'
            self.tuner_scale = '微调强度'
            self.example_block_name = '样例'
            self.examples = [[['铅笔素描'], 'a girl in a jacket'],
                             [['扁平2D艺术'], 'a cat']]
            self.save_button = '保存'


class ControlUIName():
    def __init__(self, language='en'):
        self.examples = [
            [
                'Canny',
                download_image(
                    'https://modelscope.cn/api/v1/models/iic/scepter/repo?Revision=master&FilePath=examples/control/canny_turtle.jpeg'  # noqa
                ),
                'sea turtle'
            ],
            [
                'Canny',
                download_image(
                    'https://modelscope.cn/api/v1/models/iic/scepter/repo?Revision=master&FilePath=examples/control/canny_starwar.jpeg'  # noqa
                ),
                'star wars stormtrooper with weapons'
            ],
            [
                'Hed',
                download_image(
                    'https://modelscope.cn/api/v1/models/iic/scepter/repo?Revision=master&FilePath=examples/control/hed_kingfisher.jpeg'  # noqa
                ),
                'a kingfisher coming out of the water, photorealistic hyperrealistic'
            ],
            [
                'Hed',
                download_image(
                    'https://modelscope.cn/api/v1/models/iic/scepter/repo?Revision=master&FilePath=examples/control/hed_lemon.jpeg'  # noqa
                ),
                'lemon and branches, simple background'
            ],
            [
                'Openpose',
                download_image(
                    'https://modelscope.cn/api/v1/models/iic/scepter/repo?Revision=master&FilePath=examples/control/pose_panda.jpeg'  # noqa
                ),
                'panda wearing pink suite sitting on iron throne with a sword'
            ],
            [
                'Openpose',
                download_image(
                    'https://modelscope.cn/api/v1/models/iic/scepter/repo?Revision=master&FilePath=examples/control/pose_girl.jpeg'  # noqa
                ),
                'a beautiful little girl walking on the grass, pixar style characters'
            ],
            [
                'Midas',
                download_image(
                    'https://modelscope.cn/api/v1/models/iic/scepter/repo?Revision=master&FilePath=examples/control/depth_rose.jpeg'  # noqa
                ),
                'beautiful red rose'
            ],
            [
                'Midas',
                download_image(
                    'https://modelscope.cn/api/v1/models/iic/scepter/repo?Revision=master&FilePath=examples/control/depth_c.jpg'  # noqa
                ),
                'three-dimensional letter C with fire'
            ],
            [
                'Color',
                download_image(
                    'https://modelscope.cn/api/v1/models/iic/scepter/repo?Revision=master&FilePath=examples/control/color_lilypad.jpeg'  # noqa
                ),
                'lilypad flower floating on a tiny pond surrounded by ferns morning'
            ],
            [
                'Color',
                download_image(
                    'https://modelscope.cn/api/v1/models/iic/scepter/repo?Revision=master&FilePath=examples/control/color_gooseberry.jpeg'  # noqa
                ),
                'gooseberry watercolor isolated'
            ]
        ]

        if language == 'en':

            self.source_image = 'Source Image'
            self.cond_image = 'Conditional Image'
            self.control_preprocessor = 'Control Preprocessor'
            self.crop_type = 'Crop type'
            self.direction = (
                "1) After clicking 'Extract', you can extract the conditional image from the original image on the "
                "left, then click the 'Generate' button above;\n"
                "2) You can also directly transfer the conditional image on the right, then click the 'Generate' "
                'button above;')
            self.preprocess = 'Preprocess'
            self.control_model = 'Generation Model'

            self.control_err1 = "Condition preprocessor doesn't exist."
            self.control_err2 = 'Condition preprocessor failed.'
            self.advance_block_name = 'Advance Setting'
            self.control_scale = 'Control Scale'
            self.example_block_name = 'Examples'

        elif language == 'zh':
            self.preprocess = '条件预处理'
            self.source_image = '源图片'
            self.cond_image = '条件图片'
            self.control_preprocessor = '图像预处理器'
            self.crop_type = '抠图方式'
            self.direction = ('1）点击【Extract】后可从左侧原始图像提取出条件图像，再点击上方运行；\n'
                              '2）也可直接传输右侧条件图像，再点击上方运行；\n')
            self.control_err1 = '预处理器不存在。'
            self.control_err2 = '预处理失败。'
            self.control_model = '生成模型'
            self.advance_block_name = '高级设置'
            self.control_scale = '控制强度'
            self.example_block_name = '样例'


class LargenUIName():
    def __init__(self, language='en'):

        self.tasks = [
            'Text_Guided_Outpainting', 'Subject_Guided_Inpainting',
            'Text_Guided_Inpainting', 'Text_Subject_Guided_Inpainting'
        ]

        if language == 'en':
            self.apps = [
                'Zoom Out', 'Virtual Try On', 'Inpainting (text guided)',
                'Inpainting (text + reference image guided)'
            ]
            self.dropdown_name = 'Application'
            self.subject_image = 'Reference Image'
            self.subject_mask = 'Reference Mask'
            self.scene_image = 'Scene Image'
            self.scene_mask = 'Scene Mask'
            self.prompt = 'Prompt'
            self.masked_image = 'Masked Image'
            self.preprocess = 'Input Preprocess'
            self.button_name = 'Data Preprocess'
            self.direction = (
                'Instruction: \n\n'
                'For customized data: \n\n'
                'a.1) Select the task; \n\n'
                'a.1) Upload scene image; \n\n'
                'a.2) Upload reference image (if needed); \n\n'
                'a.3) Use the brush tool to cover the areas on the scene'
                'image and reference image (to generate corresponding scene'
                'mask and reference mask); \n\n'
                'a.4) Click Data Preprocess button; \n\n'
                'a.5) Input text prompt; \n\n'
                'For example data: \n\n'
                'b.1) click example row \n\n'
                'Finally, click Generate button and get the output image!')
            self.out_direction_label = 'Out Direction'
            self.out_directions = [
                'CenterAround',
                'RightDown',
                'LeftDown',
                'RightUp',
                'LeftUp',
            ]
        elif language == 'zh':
            self.apps = ['图像扩展', '虚拟试衣', '图像补全（文本引导）', '图像补全（文本+参考图引导）']
            self.dropdown_name = '应用'
            self.subject_image = '参考图片'
            self.subject_mask = '参考图掩码'
            self.scene_image = '背景图片'
            self.scene_mask = '背景图掩码'
            self.prompt = '提示文本'
            self.masked_image = '掩码图片'
            self.preprocess = '输入图片预处理'
            self.button_name = '数据预处理'
            self.direction = ('使用说明：\n'
                              '针对自定义数据：\n'
                              'a.1）上传背景图片（待编辑）；\n'
                              'a.2）上传参考图片（如需要）；\n'
                              'a.3）使用笔刷功能涂抹图像中的特定位置（获取对应的掩码图像）；\n'
                              'a.4）点击数据预处理按钮；\n'
                              'a.5）输入文本提示；\n'
                              '针对提供的样例数据: \n'
                              'b.1）点击样例数据 \n'
                              '最后，点击生成按钮获取生成图片')
            self.out_direction_label = '扩展方向'
            self.out_directions = [
                '中心向外',
                '右下',
                '左下',
                '右上',
                '左上',
            ]

        self.examples = [
            [
                self.apps[0],
                'a temple on fire',
                download_image(
                    'https://modelscope.cn/api/v1/models/iic/LARGEN/repo?Revision=master&FilePath=examples/ex1_scene_im.png'  # noqa
                ),
                None,
                None,
                None,
                1.0,
                0.75,
                'CenterAround',
                1024,
                1024
            ],
            [
                self.apps[1],
                '',
                download_image(
                    'https://modelscope.cn/api/v1/models/iic/LARGEN/repo?Revision=master&FilePath=examples/ex2_scene_im.jpg'  # noqa
                ),
                download_image(
                    'https://modelscope.cn/api/v1/models/iic/LARGEN/repo?Revision=master&FilePath=examples/ex2_scene_mask.png'  # noqa
                ),
                download_image(
                    'https://modelscope.cn/api/v1/models/iic/LARGEN/repo?Revision=master&FilePath=examples/ex2_subject_im.jpg'  # noqa
                ),
                download_image(
                    'https://modelscope.cn/api/v1/models/iic/LARGEN/repo?Revision=master&FilePath=examples/ex2_subject_mask.jpg'  # noqa
                ),
                1.0,
                0.0,
                '',
                1024,
                1024
            ],
            [
                self.apps[2],
                'a blue and white porcelain',
                download_image(
                    'https://modelscope.cn/api/v1/models/iic/LARGEN/repo?Revision=master&FilePath=examples/ex3_scene_im.png'  # noqa
                ),
                download_image(
                    'https://modelscope.cn/api/v1/models/iic/LARGEN/repo?Revision=master&FilePath=examples/ex3_scene_mask.png'  # noqa
                ),
                None,
                None,
                1.0,
                0.0,
                '',
                1024,
                1024
            ],
            [
                self.apps[3],
                'a dog wearing sunglasses',
                download_image(
                    'https://modelscope.cn/api/v1/models/iic/LARGEN/repo?Revision=master&FilePath=examples/ex4_scene_im.png'  # noqa
                ),
                download_image(
                    'https://modelscope.cn/api/v1/models/iic/LARGEN/repo?Revision=master&FilePath=examples/ex4_scene_mask.png'  # noqa
                ),
                download_image(
                    'https://modelscope.cn/api/v1/models/iic/LARGEN/repo?Revision=master&FilePath=examples/ex4_subject_im.png'  # noqa
                ),
                download_image(
                    'https://modelscope.cn/api/v1/models/iic/LARGEN/repo?Revision=master&FilePath=examples/ex4_subject_mask.png'  # noqa
                ),
                0.45,
                0.0,
                '',
                1024,
                1024
            ],
        ]


class StyleboothUIName():
    def __init__(self, language='en'):
        if language == 'en':
            self.dropdown_name = 'Application'
            self.apps = ['Text-based Style Editing']
            # self.apps = ["Text-based Style Editing", "Exemplar-based Style Editing"]
            self.source_image = 'Source Image'
            self.exemplar_image = 'Exemplar Image'
            self.ins_format = 'Instruction Format (select or rewrite, en only)'
            self.style_format = '{} (select or rewrite, en only)'
            self.guide_scale_image = 'Guide Scale For Uncondition Image'
            self.guide_scale_text = 'Guide Scale For Uncondition Text'
            self.guide_rescale = 'Guide Rescale'
            self.resolution = 'Resolution of Short Edge'
            self.compose_button = 'Assemble Style Editing Instruction to Prompt'
        elif language == 'zh':
            self.dropdown_name = '应用'
            self.apps = ['根据文本编辑风格']
            # self.apps = ["根据文本编辑风格", "根据风格样例编辑风格"]
            self.source_image = '源图片'
            self.exemplar_image = '样例图片'
            self.ins_format = '指令模版 (选择或者新写，仅英文)'
            self.style_format = '{} (选择或者新写，仅英文)'
            self.guide_scale_image = '图片条件引导比例'
            self.guide_scale_text = '文本条件引导比例'
            self.guide_rescale = '引导缩放'
            self.resolution = '短边分辨率'
            self.compose_button = '组装风格编辑指令到Prompt栏'
        self.tb_ins_format_choice = [
            'Let this image be in the style of <style>',
            'Please edit this image to embody the characteristics of <style> style.',
            'Transform this image to reflect the distinct aesthetic of <style>.',
            'Can you infuse this image with the signature techniques representative of <style>?',
            'Adjust the visual elements of this image to emulate the <style> style.',
            'Reinterpret this image through the artistic lens of <style>.',
            'Apply the <style> style to this image to capture its unique essence.',
            'Modify this photograph to mirror the thematic qualities of <style>.',
            "I'd like you to rework this image to pay homage to the <style> movement.",
            'Ensure that this image adopts the brushwork and color palette typical of <style>.',
            'Give this image a makeover so that it aligns with the <style> stylistic approach.',
            'Retouch this image to channel the spirit and technique of <style>.',
            'Merge this image with the foundational elements of <style>.',
            'Re-envision this image to fit within the <style> genre.',
            'Adapt this image to exhibit the soft edges and vibrant light of <style>.',
            'Craft this image to resonate with the visual themes found in <style>.',
        ]
        self.eb_ins_format_choice = [
            'Let this image be in the style of <image>',
            'Please match the aesthetic of this image to that of <image>.',
            'Adjust the current image to mimic the visual style of <image>.',
            'Edit this photo so that it reflects the artistic style found in <image>.',
            'Transform this picture to be stylistically similar to <image>.',
            'Recreate the ambiance and look of <image> in this one.',
            'Harmonize the visual elements of this image with those in <image>.',
            'Ensure the editing of this image captures the essence of the style in <image>.',
            'Can you make this image resonate with the artistic flair of <image>?',
            "I'd like this image to have the same feel and tone as the style reference in <image>.",
            'Retouch this image to align with the creative direction of <image>.',
            'Replicate the style characteristics of <image> onto this one.',
            'Adapt the visual theme of this image to be consistent with <image>.',
            "Conform this image's aesthetic to the distinctive look of <image>.",
            'Make over this image so it conforms to the stylistic cues of <image>.',
            "Modify this image's style to echo the artistic qualities of <image>.",
        ]
        self.tb_target_style = [
            'Adorable 3D Character', 'Color Field Painting',
            'Colored Pencil Art', 'Graffiti Art', 'futuristic-retro futurism',
            'game-retro arcade', 'Simple Vector Art', 'Sketchup', 'mre-comic',
            'sai-comic book', 'sai-lowpoly', 'misc-stained glass',
            'misc-zentangle', 'papercraft-papercut collage', 'Pop Art 2',
            'photo-silhouette', 'Adorable Kawaii', 'sai-anime', 'sai-origami',
            'futuristic-vaporwave', 'game-retro game', 'misc-disco',
            'papercraft-flat papercut'
        ]
        self.eb_target_style = ['<image>']
        self.tb_identifier = '<style>'
        self.eb_identifier = '<image>'
