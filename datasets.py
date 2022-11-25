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
        valid_set = torchvision.datasets.CIFAR10(conf.dataset.path,
                                                  train=True,
                                                  transform=transform_test,
                                                  download=True)

        num_train  = len(train_set)
        indices    = torch.randperm(num_train).tolist()
        valid_size = int(np.floor(0.05 * num_train))

        train_idx, valid_idx = indices[valid_size:], indices[:valid_size]

        train_set = Subset(train_set, train_idx)
        valid_set = Subset(valid_set, valid_idx)

    elif conf.dataset.name == 'svhn':
        transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
            ]
        )
        transform_test = T.Compose(
            [
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
            ]
        )

        train_set = torchvision.datasets.SVHN(conf.dataset.path,
                                              split='train',
                                              transform=transform,
                                              download=True)
        valid_set = torchvision.datasets.SVHN(conf.dataset.path,
                                              split='train',
                                              transform=transform_test,
                                              download=True)

        num_train  = len(train_set)
        indices    = torch.randperm(num_train).tolist()
        valid_size = int(np.floor(0.05 * num_train))

        train_idx, valid_idx = indices[valid_size:], indices[:valid_size]

        train_set = Subset(train_set, train_idx)
        valid_set = Subset(valid_set, valid_idx)

    elif conf.dataset.name == 'celeba':
        transform = T.Compose(
            [
                CropTransform((25, 50, 25 + 128, 50 + 128)),
                T.Resize(128),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
            ]
        )
        transform_test = T.Compose(
            [
                CropTransform((25, 50, 25 + 128, 50 + 128)),
                T.Resize(128),
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
            ]
        )
        transform = T.Compose(
            [
                T.CenterCrop(148),
                T.Resize(64),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
            ]
        )
        transform_test = T.Compose(
            [
                T.CenterCrop(148),
                T.Resize(64),
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
            ]
        )

        train_set = torchvision.datasets.CelebA(conf.dataset.path,
                                                split='train',
                                                transform=transform,
                                                download=True)
        valid_set = torchvision.datasets.CelebA(conf.dataset.path,
                                                split='train',
                                                transform=transform_test,
                                                download=True)

        if conf.dataset.limit_dataset_size:
            # limit_size = list(range(min(len(train_set), conf.training.dataloader.batch_size*10+1000)))
            limit_size = list(range(100))
            train_set = Subset(train_set, limit_size)
            valid_set = Subset(valid_set, limit_size)

        num_train = len(train_set)
        indices    = torch.randperm(num_train).tolist()
        valid_size = int(np.floor(0.05 * num_train))

        train_idx, valid_idx = indices[valid_size:], indices[:valid_size]

        train_set = Subset(train_set, train_idx)
        valid_set = Subset(valid_set, valid_idx)

    else:
        raise FileNotFoundError

    return train_set, valid_set
