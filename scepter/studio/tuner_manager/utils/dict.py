# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
def update_2level_dict(d, new_dict):
    for first, v in new_dict.items():
        if first in d:
            d[first].update(v)
        else:
            d[first] = v
    return d


def delete_2level_dict(d, first_key, second_key):
    first = d.pop(first_key)
    second = first.pop(second_key)
    if len(first) > 0:
        d.update({first_key: first})
    return d, second
