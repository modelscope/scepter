# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

from scepter.modules.data.dataset.base_dataset import BaseDataset
from scepter.modules.data.dataset.dataset import (Image2ImageDataset,
                                                  ImageClassifyPublicDataset,
                                                  ImageTextPairDataset,
                                                  Text2ImageDataset)
from scepter.modules.data.dataset.ms_dataset import (
    ImageTextPairFolderDataset, ImageTextPairMSDataset)
from scepter.modules.data.dataset.registry import DATASETS
