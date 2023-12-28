# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

from scepter.modules.model.backbone.video.bricks import stems
from scepter.modules.model.backbone.video.bricks.csn_branch import CSNBranch
from scepter.modules.model.backbone.video.bricks.non_local import NonLocal
from scepter.modules.model.backbone.video.bricks.r2d3d_branch import \
    R2D3DBranch
from scepter.modules.model.backbone.video.bricks.r2plus1d_branch import \
    R2Plus1DBranch
from scepter.modules.model.backbone.video.bricks.tada_conv import \
    TAdaConvBlockAvgPool
from scepter.modules.model.backbone.video.bricks.transformer_branch import (
    BaseTransformerLayer, TimesformerLayer)
