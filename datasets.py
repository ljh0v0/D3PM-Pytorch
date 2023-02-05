# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Datasets.

All data generated should be in [0, 255].
Data loaders should iterate through the data in the same order for all hosts,
and sharding across hosts is done here.
"""

import torch
import torchvision.transforms as T
import torchvision.datasets
import numpy as np
from torch.utils.data import Subset
TRAINSUBSET = 0

class ReshapeTransform:
    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, img):
        return torch.reshape(img, self.new_size)


class CropTransform:
    def __init__(self, bbox):
        self.bbox = bbox

    def __call__(self, img):
        return img.crop(self.bbox)


def get_train_data(conf):
    if conf.dataset.name == 'cifar10':
        transform = T.Compose(
            [
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                lambda x: x * 255
            ]
        )
        transform_test = T.Compose(
            [
                T.ToTensor(),
                lambda x: x * 255
            ]
        )

        train_set = torchvision.datasets.CIFAR10(conf.dataset.path,
                                                 train=True,
                                                 transform=transform,
                                                 download=True)
        eval_set = torchvision.datasets.CIFAR10(conf.dataset.path,
                                                  train=False,
                                                  transform=transform_test,
                                                  download=True)

        if TRAINSUBSET:
            # limit_size = list(range(min(len(train_set), conf.training.dataloader.batch_size*10+1000)))
            limit_size = list(range(128))
            train_set = Subset(train_set, limit_size)
            eval_set = Subset(eval_set, limit_size)

    return train_set, eval_set
