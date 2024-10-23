# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from PIL import Image
from torchvision import transforms


def load_example_image(content, k):
    from scepter.modules.utils.file_system import FS

    image_path = FS.get_from(content[k])
    image = Image.open(image_path).convert('RGB')

    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(image)
    example_image = image_tensor.permute(1, 2, 0).unsqueeze(0)

    return example_image
