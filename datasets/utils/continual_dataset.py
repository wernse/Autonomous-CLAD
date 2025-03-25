import copy
from abc import abstractmethod
from argparse import Namespace
from torch import nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from typing import Tuple
from torchvision import datasets
import numpy as np
import socket
import torch
import os
class ContinualDataset:
    """
    Continual learning evaluation setting.
    """
    NAME = None
    SETTING = None
    N_CLASSES_PER_TASK = None
    N_TASKS = None
    TRANSFORM = None

    def __init__(self, args: Namespace) -> None:
        """
        Initializes the train and test lists of dataloaders.
        :param args: the arguments which contains the hyperparameters
        """
        self.train_loader = None
        self.test_loaders = []
        self.i = 0
        self.args = args

    @staticmethod
    @abstractmethod
    def get_setting() -> Namespace:
        """
        Return common settings such as batch_size, number of epochs, minibatch_size, ...
        """
        pass

    @abstractmethod
    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Creates and returns the training and test loaders for the current task.
        The current training loader and all test loaders are stored in self.
        :return: the current training and test loaders
        """
        pass

    @staticmethod
    @abstractmethod
    def get_backbone() -> nn.Module:
        """
        Returns the backbone to be used for to the current dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_transform() -> transforms:
        """
        Returns the transform to be used for to the current dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_loss() -> nn.functional:
        """
        Returns the loss to be used for to the current dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_normalization_transform() -> transforms:
        """
        Returns the transform used for normalizing the current dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_denormalization_transform() -> transforms:
        """
        Returns the transform used for denormalizing the current dataset.
        """
        pass


def store_masked_loaders(train_dataset: datasets, test_dataset: datasets,
                    setting: ContinualDataset, class_order=None) -> Tuple[DataLoader, DataLoader]:
    """
    Divides the dataset into tasks.
    :param train_dataset: train dataset
    :param test_dataset: test dataset
    :param setting: continual learning setting
    :return: train and test loaders
    """
    dset_setting = setting.get_setting()

    if class_order is not None:
        train_dataset.targets = class_order[np.array(train_dataset.targets)]
        test_dataset.targets = class_order[np.array(test_dataset.targets)]

    if 'seq-ilsvrc' not in setting.NAME:
        train_mask = np.logical_and(np.array(train_dataset.targets) >= setting.i,
            np.array(train_dataset.targets) < setting.i + setting.N_CLASSES_PER_TASK)
        test_mask = np.logical_and(np.array(test_dataset.targets) >= setting.i,
            np.array(test_dataset.targets) < setting.i + setting.N_CLASSES_PER_TASK)

        if not isinstance(train_dataset.data, np.ndarray):
            train_dataset.data = np.array(train_dataset.data)
            test_dataset.data = np.array(test_dataset.data)

        train_dataset.data = train_dataset.data[train_mask]
        test_dataset.data = test_dataset.data[test_mask]

        train_dataset.targets = np.array(train_dataset.targets)[train_mask]
        test_dataset.targets = np.array(test_dataset.targets)[test_mask]

    if 'MAMMOTH_RANK' not in os.environ:
        train_loader = DataLoader(train_dataset,
                                batch_size=32, shuffle=True, drop_last=True)
    else:
        train_loader = DataLoader(train_dataset,
                                batch_size=32, drop_last=True, sampler=torch.utils.data.DistributedSampler(train_dataset, shuffle=True))
                                
    if not 'MAMMOTH_SLAVE' in os.environ:
        test_loader = DataLoader(test_dataset,
                                batch_size=32, shuffle=False)
    else:
        test_loader = None
    setting.test_loaders.append(test_loader)
    setting.train_loader = train_loader

    setting.i += setting.N_CLASSES_PER_TASK
    return train_loader, test_loader

def store_noisy_masked_loaders(train_dataset: datasets, test_dataset: datasets, setting: ContinualDataset, corrupt_prob=0) -> Tuple[DataLoader, DataLoader]:
    """
    Divides the dataset into tasks.
    :param train_dataset: train dataset
    :param test_dataset: test.sh dataset
    :param setting: continual learning setting
    :return: train and test.sh loaders
    """
    train_mask = np.logical_and(np.array(train_dataset.targets) >= setting.i,
        np.array(train_dataset.targets) < setting.i + setting.N_CLASSES_PER_TASK)

    test_mask = np.logical_and(np.array(test_dataset.targets) >= setting.i,
        np.array(test_dataset.targets) < setting.i + setting.N_CLASSES_PER_TASK)

    train_dataset.data = train_dataset.data[train_mask]
    test_dataset.data = test_dataset.data[test_mask]

    train_dataset.targets = np.array(train_dataset.targets)[train_mask]
    test_dataset.targets = np.array(test_dataset.targets)[test_mask]

    orig_labels = copy.deepcopy(train_dataset.targets)


    print(f'Applying label corruption with probability {corrupt_prob}')
    mask = np.random.rand(len(train_dataset.targets)) <= corrupt_prob
    rnd_labels = np.random.choice(list(range(setting.i, setting.i + setting.N_CLASSES_PER_TASK)), mask.sum())
    train_dataset.targets[mask] = rnd_labels
    train_dataset.targets = [int(x) for x in train_dataset.targets]

    train_dataset.is_noise = orig_labels != train_dataset.targets
    print('Actual Noise:', np.mean(train_dataset.is_noise))

    train_loader = DataLoader(train_dataset,
                              batch_size=32, shuffle=True, num_workers=0, drop_last=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=32, shuffle=False, num_workers=0)
    setting.test_loaders.append(test_loader)
    setting.train_loader = train_loader

    setting.i += setting.N_CLASSES_PER_TASK
    return train_loader, test_loader


def get_previous_train_loader(train_dataset: datasets, batch_size: int,
                              setting: ContinualDataset) -> DataLoader:
    """
    Creates a dataloader for the previous task.
    :param train_dataset: the entire training set
    :param batch_size: the desired batch size
    :param setting: the continual dataset at hand
    :return: a dataloader
    """
    train_mask = np.logical_and(np.array(train_dataset.targets) >=
        setting.i - setting.N_CLASSES_PER_TASK, np.array(train_dataset.targets)
        < setting.i - setting.N_CLASSES_PER_TASK + setting.N_CLASSES_PER_TASK)

    train_dataset.data = train_dataset.data[train_mask]
    train_dataset.targets = np.array(train_dataset.targets)[train_mask]

    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
