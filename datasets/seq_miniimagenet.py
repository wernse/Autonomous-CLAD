import os
import pickle
from argparse import Namespace
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch import nn
from torch.utils.data import Dataset

from backbone.ResNet18 import resnet18
from datasets.transforms.denormalization import DeNormalize
from datasets.utils.continual_dataset import ContinualDataset, store_masked_loaders

def base_path() -> str:
    return './data/'

def smart_joint(*paths):
    return os.path.join(*paths).replace("\\", "/")

class MiniImagenet(Dataset):
    def __init__(self, root: str, train: bool = True, transform: Optional[nn.Module] = None,
                 target_transform: Optional[nn.Module] = None, download: bool = False) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.name = 'train' if train else 'test'
        root = os.path.join('../wsn/data/', 'miniimagenet')
        with open(os.path.join(root, '{}.pkl'.format(self.name)), 'rb') as f:
            data_dict = pickle.load(f)

        self.data = data_dict['images']
        self.targets = data_dict['labels']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(np.uint8(img))
        original_img = img.copy()

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        not_aug_img = self.not_aug_transform(original_img)
        return img, target, not_aug_img

class SequentialMiniImagenet(ContinualDataset):
    NAME = 'seq-miniimg'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 10  # Adjust if different
    N_TASKS = 10  # Adjust if different
    N_CLASSES = 100  # Total number of classes in MiniImagenet
    MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    SIZE = (84, 84)  # Image size for MiniImagenet

    TRANSFORM = transforms.Compose([
        transforms.RandomCrop(84, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

    def get_data_loaders(self):
        transform = self.TRANSFORM
        test_transform = transforms.Compose([
            transforms.ToTensor(), self.get_normalization_transform()
        ])

        train_dataset = MiniImagenet(base_path() + 'MINIIMG', train=True, download=True, transform=transform)
        test_dataset = MiniImagenet(base_path() + 'MINIIMG', train=False, download=True, transform=test_transform)

        train, test = store_masked_loaders(train_dataset, test_dataset, self)
        return train, test

    @staticmethod
    def get_setting():
        return Namespace(**{
            "batch_size": 64,
            "minibatch_size": 64,
            "replaybatch_size": 64,
            "scheduler": "simple",
            "scheduler_rate": 0.1,
            "n_epochs": 50,
            "pre_epochs": 50,
            "opt_steps": [35, 45]})

    def get_transform(self):
        transform = transforms.Compose(
            [transforms.ToPILImage(), self.TRANSFORM])
        return transform

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize(SequentialMiniImagenet.MEAN, SequentialMiniImagenet.STD)
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize(SequentialMiniImagenet.MEAN, SequentialMiniImagenet.STD)
        return transform


    @staticmethod
    def get_backbone():
        return resnet18(SequentialMiniImagenet.N_CLASSES_PER_TASK * SequentialMiniImagenet.N_TASKS)

    @staticmethod
    def get_loss():
        return F.cross_entropy
