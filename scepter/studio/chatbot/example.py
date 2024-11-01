# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import os

from scepter.modules.utils.file_system import FS


def download_image(image, local_path=None):
    if not FS.exists(local_path):
        local_path = FS.get_from(image, local_path=local_path)
    return local_path


def get_examples(cache_dir):
    print('Downloading Examples ...')
    examples = [
        [
            'Image Segmentation',
            download_image(
                'https://github.com/ali-vilab/ace-page/blob/main/assets/examples/db3ebaa81899.png?raw=true',
                os.path.join(cache_dir, 'examples/db3ebaa81899.png')), None,
            None, '{image} Segmentation', 6666
        ],
        [
            'Depth Estimation',
            download_image(
                'https://github.com/ali-vilab/ace-page/blob/main/assets/examples/f1927c4692ba.png?raw=true',
                os.path.join(cache_dir, 'examples/f1927c4692ba.png')), None,
            None, '{image} Depth Estimation', 6666
        ],
        [
            'Pose Estimation',
            download_image(
                'https://github.com/ali-vilab/ace-page/blob/main/assets/examples/014e5bf3b4d1.png?raw=true',
                os.path.join(cache_dir, 'examples/014e5bf3b4d1.png')), None,
            None, '{image} distinguish the poses of the figures', 999999
        ],
        [
            'Scribble Extraction',
            download_image(
                'https://github.com/ali-vilab/ace-page/blob/main/assets/examples/5f59a202f8ac.png?raw=true',
                os.path.join(cache_dir, 'examples/5f59a202f8ac.png')), None,
            None, 'Generate a scribble of {image}, please.', 6666
        ],
        [
            'Mosaic',
            download_image(
                'https://github.com/ali-vilab/ace-page/blob/main/assets/examples/3a2f52361eea.png?raw=true',
                os.path.join(cache_dir, 'examples/3a2f52361eea.png')), None,
            None, 'Adapt {image} into a mosaic representation.', 6666
        ],
        [
            'Edge map Extraction',
            download_image(
                'https://github.com/ali-vilab/ace-page/blob/main/assets/examples/b9d1e519d6e5.png?raw=true',
                os.path.join(cache_dir, 'examples/b9d1e519d6e5.png')), None,
            None, 'Get the edge-enhanced result for {image}.', 6666
        ],
        [
            'Grayscale',
            download_image(
                'https://github.com/ali-vilab/ace-page/blob/main/assets/examples/c4ebbe2ba29b.png?raw=true',
                os.path.join(cache_dir, 'examples/c4ebbe2ba29b.png')), None,
            None, 'transform {image} into a black and white one', 6666
        ],
        [
            'Contour Extraction',
            download_image(
                'https://github.com/ali-vilab/ace-page/blob/main/assets/examples/19652d0f6c4b.png?raw=true',
                os.path.join(cache_dir,
                             'examples/19652d0f6c4b.png')), None, None,
            'Would you be able to make a contour picture from {image} for me?',
            6666
        ],
        [
            'Controllable Generation',
            download_image(
                'https://github.com/ali-vilab/ace-page/blob/main/assets/examples/249cda2844b7.png?raw=true',
                os.path.join(cache_dir,
                             'examples/249cda2844b7.png')), None, None,
            'Following the segmentation outcome in mask of {image}, develop a real-life image using the explanatory note in "a mighty cat lying on the bed”.',
            6666
        ],
        [
            'Controllable Generation',
            download_image(
                'https://github.com/ali-vilab/ace-page/blob/main/assets/examples/411f6c4b8e6c.png?raw=true',
                os.path.join(cache_dir,
                             'examples/411f6c4b8e6c.png')), None, None,
            'use the depth map {image} and the text caption "a cut white cat" to create a corresponding graphic image',
            999999
        ],
        [
            'Controllable Generation',
            download_image(
                'https://github.com/ali-vilab/ace-page/blob/main/assets/examples/a35c96ed137a.png?raw=true',
                os.path.join(cache_dir,
                             'examples/a35c96ed137a.png')), None, None,
            'help translate this posture schema {image} into a colored image based on the context I provided "A beautiful woman Climbing the climbing wall, wearing a harness and climbing gear, skillfully maneuvering up the wall with her back to the camera, with a safety rope."',
            3599999
        ],
        [
            'Controllable Generation',
            download_image(
                'https://github.com/ali-vilab/ace-page/blob/main/assets/examples/dcb2fc86f1ce.png?raw=true',
                os.path.join(cache_dir,
                             'examples/dcb2fc86f1ce.png')), None, None,
            'Transform and generate an image using mosaic {image} and "Monarch butterflies gracefully perch on vibrant purple flowers, showcasing their striking orange and black wings in a lush garden setting." description',
            6666
        ],
        [
            'Controllable Generation',
            download_image(
                'https://github.com/ali-vilab/ace-page/blob/main/assets/examples/4cd4ee494962.png?raw=true',
                os.path.join(cache_dir,
                             'examples/4cd4ee494962.png')), None, None,
            'make this {image} colorful as per the "beautiful sunflowers"',
            6666
        ],
        [
            'Controllable Generation',
            download_image(
                'https://github.com/ali-vilab/ace-page/blob/main/assets/examples/a47e3a9cd166.png?raw=true',
                os.path.join(cache_dir,
                             'examples/a47e3a9cd166.png')), None, None,
            'Take the edge conscious {image} and the written guideline "A whimsical animated character is depicted holding a delectable cake adorned with blue and white frosting and a drizzle of chocolate. The character wears a yellow headband with a bow, matching a cozy yellow sweater. Her dark hair is styled in a braid, tied with a yellow ribbon. With a golden fork in hand, she stands ready to enjoy a slice, exuding an air of joyful anticipation. The scene is creatively rendered with a charming and playful aesthetic." and produce a realistic image.',
            613725
        ],
        [
            'Controllable Generation',
            download_image(
                'https://github.com/ali-vilab/ace-page/blob/main/assets/examples/d890ed8a3ac2.png?raw=true',
                os.path.join(cache_dir,
                             'examples/d890ed8a3ac2.png')), None, None,
            'creating a vivid image based on {image} and description "This image features a delicious rectangular tart with a flaky, golden-brown crust. The tart is topped with evenly sliced tomatoes, layered over a creamy cheese filling. Aromatic herbs are sprinkled on top, adding a touch of green and enhancing the visual appeal. The background includes a soft, textured fabric and scattered white flowers, creating an elegant and inviting presentation. Bright red tomatoes in the upper right corner hint at the fresh ingredients used in the dish."',
            6666
        ],
        [
            'Controllable Generation',
            download_image(
                'https://github.com/ali-vilab/ace-page/blob/main/assets/examples/131ca90fd2a9.png?raw=true',
                os.path.join(cache_dir,
                             'examples/131ca90fd2a9.png')), None, None,
            '"A person sits contemplatively on the ground, surrounded by falling autumn leaves. Dressed in a green sweater and dark blue pants, they rest their chin on their hand, exuding a relaxed demeanor. Their stylish checkered slip-on shoes add a touch of flair, while a black purse lies in their lap. The backdrop of muted brown enhances the warm, cozy atmosphere of the scene." , generate the image that corresponds to the given scribble {image}.',
            613725
        ],
        [
            'Image Denoising',
            download_image(
                'https://github.com/ali-vilab/ace-page/blob/main/assets/examples/0844a686a179.png?raw=true',
                os.path.join(cache_dir,
                             'examples/0844a686a179.png')), None, None,
            'Eliminate noise interference in {image} and maximize the crispness to obtain superior high-definition quality',
            6666
        ],
        [
            'Inpainting',
            download_image(
                'https://github.com/ali-vilab/ace-page/blob/main/assets/examples/fa91b6b7e59b.png?raw=true',
                os.path.join(cache_dir, 'examples/fa91b6b7e59b.png')),
            download_image(
                'https://github.com/ali-vilab/ace-page/blob/main/assets/examples/fa91b6b7e59b_mask.png?raw=true',
                os.path.join(cache_dir,
                             'examples/fa91b6b7e59b_mask.png')), None,
            'Ensure to overhaul the parts of the {image} indicated by the mask.',
            6666
        ],
        [
            'Inpainting',
            download_image(
                'https://github.com/ali-vilab/ace-page/blob/main/assets/examples/632899695b26.png?raw=true',
                os.path.join(cache_dir, 'examples/632899695b26.png')),
            download_image(
                'https://github.com/ali-vilab/ace-page/blob/main/assets/examples/632899695b26_mask.png?raw=true',
                os.path.join(cache_dir,
                             'examples/632899695b26_mask.png')), None,
            'Refashion the mask portion of {image} in accordance with "A yellow egg with a smiling face painted on it"',
            6666
        ],
        [
            'Outpainting',
            download_image(
                'https://github.com/ali-vilab/ace-page/blob/main/assets/examples/f2b22c08be3f.png?raw=true',
                os.path.join(cache_dir, 'examples/f2b22c08be3f.png')),
            download_image(
                'https://github.com/ali-vilab/ace-page/blob/main/assets/examples/f2b22c08be3f_mask.png?raw=true',
                os.path.join(cache_dir,
                             'examples/f2b22c08be3f_mask.png')), None,
            'Could the {image} be widened within the space designated by mask, while retaining the original?',
            6666
        ],
        [
            'General Editing',
            download_image(
                'https://github.com/ali-vilab/ace-page/blob/main/assets/examples/354d17594afe.png?raw=true',
                os.path.join(cache_dir,
                             'examples/354d17594afe.png')), None, None,
            '{image} change the dog\'s posture to walking in the water, and change the background to green plants and a pond.',
            6666
        ],
        [
            'General Editing',
            download_image(
                'https://github.com/ali-vilab/ace-page/blob/main/assets/examples/38946455752b.png?raw=true',
                os.path.join(cache_dir,
                             'examples/38946455752b.png')), None, None,
            '{image} change the color of the dress from white to red and the model\'s hair color red brown to blonde.Other parts remain unchanged',
            6669
        ],
        [
            'Facial Editing',
            download_image(
                'https://github.com/ali-vilab/ace-page/blob/main/assets/examples/3ba5202f0cd8.png?raw=true',
                os.path.join(cache_dir,
                             'examples/3ba5202f0cd8.png')), None, None,
            'Keep the same facial feature in @3ba5202f0cd8, change the woman\'s clothing from a Blue denim jacket to a white turtleneck sweater and adjust her posture so that she is supporting her chin with both hands. Other aspects, such as background, hairstyle, facial expression, etc, remain unchanged.',
            99999
        ],
        [
            'Facial Editing',
            download_image(
                'https://github.com/ali-vilab/ace-page/blob/main/assets/examples/369365b94725.png?raw=true',
                os.path.join(cache_dir, 'examples/369365b94725.png')), None,
            None, '{image} Make her looking at the camera', 6666
        ],
        [
            'Facial Editing',
            download_image(
                'https://github.com/ali-vilab/ace-page/blob/main/assets/examples/92751f2e4a0e.png?raw=true',
                os.path.join(cache_dir, 'examples/92751f2e4a0e.png')), None,
            None, '{image} Remove the smile from his face', 9899999
        ],
        [
            'Render Text',
            download_image(
                'https://github.com/ali-vilab/ace-page/blob/main/assets/examples/33e9f27c2c48.png?raw=true',
                os.path.join(cache_dir, 'examples/33e9f27c2c48.png')),
            download_image(
                'https://github.com/ali-vilab/ace-page/blob/main/assets/examples/33e9f27c2c48_mask.png?raw=true',
                os.path.join(cache_dir,
                             'examples/33e9f27c2c48_mask.png')), None,
            'Put the text "C A T" at the position marked by mask in the {image}',
            6666
        ],
        [
            'Remove Text',
            download_image(
                'https://github.com/ali-vilab/ace-page/blob/main/assets/examples/8530a6711b2e.png?raw=true',
                os.path.join(cache_dir, 'examples/8530a6711b2e.png')), None,
            None, 'Aim to remove any textual element in {image}', 6666
        ],
        [
            'Remove Text',
            download_image(
                'https://github.com/ali-vilab/ace-page/blob/main/assets/examples/c4d7fb28f8f6.png?raw=true',
                os.path.join(cache_dir, 'examples/c4d7fb28f8f6.png')),
            download_image(
                'https://github.com/ali-vilab/ace-page/blob/main/assets/examples/c4d7fb28f8f6_mask.png?raw=true',
                os.path.join(cache_dir,
                             'examples/c4d7fb28f8f6_mask.png')), None,
            'Rub out any text found in the mask sector of the {image}.', 6666
        ],
        [
            'Remove Object',
            download_image(
                'https://github.com/ali-vilab/ace-page/blob/main/assets/examples/e2f318fa5e5b.png?raw=true',
                os.path.join(cache_dir,
                             'examples/e2f318fa5e5b.png')), None, None,
            'Remove the unicorn in this {image}, ensuring a smooth edit.',
            99999
        ],
        [
            'Remove Object',
            download_image(
                'https://github.com/ali-vilab/ace-page/blob/main/assets/examples/1ae96d8aca00.png?raw=true',
                os.path.join(cache_dir, 'examples/1ae96d8aca00.png')),
            download_image(
                'https://github.com/ali-vilab/ace-page/blob/main/assets/examples/1ae96d8aca00_mask.png?raw=true',
                os.path.join(cache_dir, 'examples/1ae96d8aca00_mask.png')),
            None, 'Discard the contents of the mask area from {image}.', 99999
        ],
        [
            'Add Object',
            download_image(
                'https://github.com/ali-vilab/ace-page/blob/main/assets/examples/80289f48e511.png?raw=true',
                os.path.join(cache_dir, 'examples/80289f48e511.png')),
            download_image(
                'https://github.com/ali-vilab/ace-page/blob/main/assets/examples/80289f48e511_mask.png?raw=true',
                os.path.join(cache_dir,
                             'examples/80289f48e511_mask.png')), None,
            'add a Hot Air Balloon into the {image}, per the mask', 613725
        ],
        [
            'Style Transfer',
            download_image(
                'https://github.com/ali-vilab/ace-page/blob/main/assets/examples/d725cb2009e8.png?raw=true',
                os.path.join(cache_dir, 'examples/d725cb2009e8.png')), None,
            None, 'Change the style of {image} to colored pencil style', 99999
        ],
        [
            'Style Transfer',
            download_image(
                'https://github.com/ali-vilab/ace-page/blob/main/assets/examples/e0f48b3fd010.png?raw=true',
                os.path.join(cache_dir, 'examples/e0f48b3fd010.png')), None,
            None, 'make {image} to Walt Disney Animation style', 99999
        ],
        [
            'Style Transfer',
            download_image(
                'https://github.com/ali-vilab/ace-page/blob/main/assets/examples/9e73e7eeef55.png?raw=true',
                os.path.join(cache_dir, 'examples/9e73e7eeef55.png')), None,
            download_image(
                'https://github.com/ali-vilab/ace-page/blob/main/assets/examples/2e02975293d6.png?raw=true',
                os.path.join(cache_dir, 'examples/2e02975293d6.png')),
            'edit {image} based on the style of {image1} ', 99999
        ],
        [
            'Try On',
            download_image(
                'https://github.com/ali-vilab/ace-page/blob/main/assets/examples/ee4ca60b8c96.png?raw=true',
                os.path.join(cache_dir, 'examples/ee4ca60b8c96.png')),
            download_image(
                'https://github.com/ali-vilab/ace-page/blob/main/assets/examples/ee4ca60b8c96_mask.png?raw=true',
                os.path.join(cache_dir, 'examples/ee4ca60b8c96_mask.png')),
            download_image(
                'https://github.com/ali-vilab/ace-page/blob/main/assets/examples/ebe825bbfe3c.png?raw=true',
                os.path.join(cache_dir, 'examples/ebe825bbfe3c.png')),
            'Change the cloth in {image} to the one in {image1}', 99999
        ],
        [
            'Workflow',
            download_image(
                'https://github.com/ali-vilab/ace-page/blob/main/assets/examples/cb85353c004b.png?raw=true',
                os.path.join(cache_dir, 'examples/cb85353c004b.png')), None,
            None, '<workflow> ice cream {image}', 99999
        ],
    ]
    print('Finish. Start building UI ...')
    return examples
