# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import re


def is_valid_filename(filename):
    if re.match('^[A-Za-z0-9_@]+$', filename):
        return True
    else:
        return False
