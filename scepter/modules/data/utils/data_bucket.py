# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import math
import random
from typing import List, NamedTuple

import numpy as np


def make_bucket_resolutions(max_reso,
                            min_size=256,
                            max_size=1024,
                            divisible=64):
    max_width, max_height = max_reso
    max_area = (max_width // divisible) * (max_height // divisible)

    resos = set()

    size = int(math.sqrt(max_area)) * divisible
    resos.add((size, size))

    size = min_size
    while size <= max_size:
        width = size
        height = min(max_size, (max_area // (width // divisible)) * divisible)
        resos.add((width, height))
        resos.add((height, width))

        # # make additional resos
        # if width >= height and width - divisible >= min_size:
        #   resos.add((width - divisible, height))
        #   resos.add((height, width - divisible))
        # if height >= width and height - divisible >= min_size:
        #   resos.add((width, height - divisible))
        #   resos.add((height - divisible, width))

        size += divisible

    resos = list(resos)
    resos.sort()
    return resos


class BucketBatchIndex(NamedTuple):
    bucket_index: int
    bucket_batch_size: int
    batch_index: int
    bucket_reso: List[int]


class BucketManager:
    def __init__(self,
                 max_reso,
                 min_size=256,
                 max_size=1024,
                 reso_steps=64,
                 no_upscale=False) -> None:
        self.no_upscale = no_upscale
        if max_reso is None:
            self.max_reso = None
            self.max_area = None
        else:
            self.max_reso = max_reso
            self.max_area = max_reso[0] * max_reso[1]
        self.min_size = min_size
        self.max_size = max_size
        self.reso_steps = reso_steps

        self.resos = []
        self.reso_to_id = {}
        self.buckets = []

    def add_image(self, reso, image):
        bucket_id = self.reso_to_id[reso]
        self.buckets[bucket_id].append(image)

    def shuffle(self):
        for bucket in self.buckets:
            random.shuffle(bucket)

    def sort(self):
        sorted_resos = self.resos.copy()
        sorted_resos.sort()

        sorted_buckets = []
        sorted_reso_to_id = {}
        for i, reso in enumerate(sorted_resos):
            bucket_id = self.reso_to_id[reso]
            sorted_buckets.append(self.buckets[bucket_id])
            sorted_reso_to_id[reso] = i

        self.resos = sorted_resos
        self.buckets = sorted_buckets
        self.reso_to_id = sorted_reso_to_id

    def make_buckets(self):
        resos = make_bucket_resolutions(self.max_reso, self.min_size,
                                        self.max_size, self.reso_steps)
        self.set_predefined_resos(resos)

    def set_predefined_resos(self, resos):
        self.predefined_resos = resos.copy()
        self.predefined_resos_set = set(resos)
        self.predefined_aspect_ratios = np.array([w / h for w, h in resos])

    def add_if_new_reso(self, reso):
        if reso not in self.reso_to_id:
            bucket_id = len(self.resos)
            self.reso_to_id[reso] = bucket_id
            self.resos.append(reso)
            self.buckets.append([])
            # print(reso, bucket_id, len(self.buckets))

    def round_to_steps(self, x):
        x = int(x + 0.5)
        return x - x % self.reso_steps

    def select_bucket(self, image_width, image_height):
        aspect_ratio = image_width / image_height
        if not self.no_upscale:
            reso = (image_width, image_height)
            if reso in self.predefined_resos_set:
                pass
            else:
                ar_errors = self.predefined_aspect_ratios - aspect_ratio
                predefined_bucket_id = np.abs(ar_errors).argmin()
                reso = self.predefined_resos[predefined_bucket_id]

            ar_reso = reso[0] / reso[1]
            if aspect_ratio > ar_reso:
                scale = reso[1] / image_height
            else:
                scale = reso[0] / image_width

            resized_size = (int(image_width * scale + 0.5),
                            int(image_height * scale + 0.5))
            # print("use predef", image_width, image_height, reso, resized_size)
        else:
            if image_width * image_height > self.max_area:
                resized_width = math.sqrt(self.max_area * aspect_ratio)
                resized_height = self.max_area / resized_width
                assert abs(resized_width / resized_height -
                           aspect_ratio) < 1e-2, 'aspect is illegal'

                b_width_rounded = self.round_to_steps(resized_width)
                b_height_in_wr = self.round_to_steps(b_width_rounded /
                                                     aspect_ratio)
                ar_width_rounded = b_width_rounded / b_height_in_wr

                b_height_rounded = self.round_to_steps(resized_height)
                b_width_in_hr = self.round_to_steps(b_height_rounded *
                                                    aspect_ratio)
                ar_height_rounded = b_width_in_hr / b_height_rounded

                # print(b_width_rounded, b_height_in_wr, ar_width_rounded)
                # print(b_width_in_hr, b_height_rounded, ar_height_rounded)

                if abs(ar_width_rounded -
                       aspect_ratio) < abs(ar_height_rounded - aspect_ratio):
                    resized_size = (b_width_rounded,
                                    int(b_width_rounded / aspect_ratio + 0.5))
                else:
                    resized_size = (int(b_height_rounded * aspect_ratio + 0.5),
                                    b_height_rounded)
                # print(resized_size)
            else:
                resized_size = (image_width, image_height)

            bucket_width = resized_size[0] - resized_size[0] % self.reso_steps
            bucket_height = resized_size[1] - resized_size[1] % self.reso_steps
            # print("use arbitrary", image_width, image_height, resized_size, bucket_width, bucket_height)

            reso = (bucket_width, bucket_height)
        self.add_if_new_reso(reso)

        ar_error = (reso[0] / reso[1]) - aspect_ratio
        return reso, resized_size, ar_error


if __name__ == '__main__':
    image_size_list = [(256, 256), (512, 378), (378, 512), (1024, 1024),
                       (768, 1024), (768, 768), (256, 1024), (512, 512)]
    image_path_list = [f'image_path_{i}' for i in range(len(image_size_list))]

    max_reso = (512, 1024)
    min_bucket_reso = 256
    max_bucket_reso = 1024
    bucket_reso_steps = 64
    bucket_no_upscale = False
    bucket_manager = BucketManager(max_reso=max_reso,
                                   min_size=min_bucket_reso,
                                   max_size=max_bucket_reso,
                                   reso_steps=bucket_reso_steps,
                                   no_upscale=bucket_no_upscale)
    if not bucket_no_upscale:
        bucket_manager.make_buckets()
    else:
        print(
            'min_bucket_reso and max_bucket_reso are ignored if bucket_no_upscale is set, '
            'because bucket reso is defined by image size automatically / bucket_no_upscale'
        )

    for i, (path, size) in enumerate(zip(image_path_list, image_size_list)):
        image_width, image_height = size
        bucket_reso, resized_size, ar_error = bucket_manager.select_bucket(
            image_width, image_height)
        print(i, size, bucket_reso, resized_size, ar_error)
        bucket_manager.add_image(reso=bucket_reso, image=path)

    for i, (reso, bucket) in enumerate(
            zip(bucket_manager.resos, bucket_manager.buckets)):
        count = len(bucket)
        if count > 0:
            print(
                f'bucket {i}: resolution {reso}, bucket {bucket}, count: {len(bucket)}'
            )

    batch_size = 2
    buckets_indices: List[BucketBatchIndex] = []
    for bucket_index, bucket in enumerate(bucket_manager.buckets):
        batch_count = int(math.ceil(len(bucket) / batch_size))
        for batch_index in range(batch_count):
            buckets_indices.append(
                BucketBatchIndex(bucket_index, batch_size, batch_index))

    def shuffle_buckets():
        random.shuffle(buckets_indices)
        bucket_manager.shuffle()

    shuffle_buckets()
