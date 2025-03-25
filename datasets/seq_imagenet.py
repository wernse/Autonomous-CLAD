# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import pickle
from argparse import Namespace
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import Tensor

from backbone.ResNet18 import resnet18
from PIL import Image
from torch.utils.data import Dataset

from datasets.transforms.denormalization import DeNormalize
from datasets.utils.continual_dataset import (ContinualDataset,
                                              store_masked_loaders)
from datasets.utils.validation import get_train_val
from torchvision import datasets, transforms


class Imagenet(Dataset):
    def __init__(self, root: str, train: bool,
                 transform: transforms.Compose):

        self.not_aug_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

        self.train = train


        if train:
            train_data = datasets.ImageFolder("../SSIL/dat/Imagenet/train", transform=transform)

            self.data = []
            self.targets = []


            for i in range(len(train_data.imgs)):
                path, target = train_data.imgs[i]
                if target > 100:
                    continue
                self.data.append(path)
                self.targets.append(target)

            self.data = np.stack(self.data, axis=0)
        else:
            self.data = []
            self.targets = []
            test_data = datasets.ImageFolder("../SSIL/dat/Imagenet/val", transform=transform)
            for i in range(len(test_data.imgs)):
                path, target = test_data.imgs[i]
                if target > 100:
                    continue
                self.data.append(path)
                self.targets.append(target)
            self.data = np.stack(self.data, axis=0)
        self.transform = transform

    @staticmethod
    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo)
            # endwith

        return dict
        # enddef

    def __getitem__(self, index):
        org_img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        try:
            img = Image.fromarray(org_img.astype('uint8'), 'RGB')
        except:
            img = Image.open(org_img, mode='r').convert('RGB')
            pass

        original_img = img.copy()

        if self.transform is not None:
            img = self.transform(img)

        if not self.train:
            return img, target

        not_aug_img = self.not_aug_transform(original_img)

        return img, target, not_aug_img

    def __len__(self) -> int:
        return len(self.targets)
        # enddef


class SequentialImagenet(ContinualDataset):
    """The Sequential Tiny Imagenet dataset.

    Args:
        NAME (str): name of the dataset.
        SETTING (str): setting of the dataset.
        N_CLASSES_PER_TASK (int): number of classes per task.
        N_TASKS (int): number of tasks.
        N_CLASSES (int): number of classes.
        SIZE (tuple): size of the images.
        MEAN (tuple): mean of the dataset.
        STD (tuple): standard deviation of the dataset.
        TRANSFORM (torchvision.transforms): transformations to apply to the dataset.
    """

    NAME = 'seq-img'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 10
    N_TASKS = 100
    N_CLASSES = 1000

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    SIZE = (224, 224)
    TRANSFORM = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

    def get_data_loaders(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        transform = self.TRANSFORM
        DIR_DATA = '/home/wernsen/wsn/data'

        test_transform =  transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ])

        train_dataset = Imagenet(root=os.path.join(DIR_DATA, 'imagenet'), train=True, transform=transform)
        test_dataset = Imagenet(root=os.path.join(DIR_DATA, 'imagenet'), train=False, transform=test_transform)

        train, test = store_masked_loaders(train_dataset, test_dataset, self)
        return train, test

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_setting():
        return Namespace(**{
            "batch_size":256,
            "minibatch_size":256,
            "replaybatch_size": 32,
            "scheduler":"simple",
            "scheduler_rate":0.1,
            "n_epochs":100,
            "pre_epochs":50,
            "opt_steps":[30, 60, 80, 90]})

    def get_transform(self):
        transform = transforms.Compose(
            [transforms.ToPILImage(), SequentialImagenet.TRANSFORM])
        return transform

    @staticmethod
    def get_epochs():
        return 50

    @staticmethod
    def get_batch_size():
        return 32


class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None


class iImageNet1000(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
        assert 0, "You should specify the folder of your dataset"
        train_dir = "[DATA-PATH]/train/"
        test_dir = "[DATA-PATH]/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iImageNet100(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
        assert 0, "You should specify the folder of your dataset"
        train_dir = "[DATA-PATH]/train/"
        test_dir = "[DATA-PATH]/val/"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)