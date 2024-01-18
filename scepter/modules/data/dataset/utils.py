# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

from PIL import Image
from torch.utils.data.dataloader import default_collate


def pil_collate_fn(self, batch):
    batch_data = {}
    for items in batch:
        for key, item in items.items():
            if isinstance(item, Image.Image):
                if key not in batch_data:
                    batch_data[key] = []
                batch_data[key].append(item)
            else:
                if key not in batch_data:
                    batch_data[key] = []
                batch_data[key].append(item)

    for key, item in batch_data.items():
        if not all(isinstance(x, Image.Image) for x in item):
            batch_data[key] = default_collate(item)

    return batch_data
