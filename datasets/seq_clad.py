import argparse
from argparse import Namespace

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms
from backbone.ResNet18 import resnet18
import torch.nn.functional as F
import numpy as np
from PIL import Image

from backbone.alexnet import STLAlexNet
from datasets.utils.validation import get_train_val
from datasets.utils.continual_dataset import ContinualDataset, store_masked_loaders
from typing import Tuple
from datasets.transforms.denormalization import DeNormalize

def base_path() -> str:
    """
    Returns the base bath where to log accuracies and tensorboard data.
    """
    return './data/'

class SequentialCLAD(ContinualDataset):
    NAME = 'seq-clad'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 6
    N_TASKS = 6
    N_CLASSES = 6
    MEAN, STD = (0.3252, 0.3283, 0.3407), (0.0265, 0.0241, 0.0252)
    # MEAN, STD = (0.4802, 0.4480, 0.3975), (0.2770, 0.2691, 0.2821)
    SIZE = (64, 64)
    TRANSFORM = transforms.Compose(
        [transforms.RandomCrop(64, padding=4),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize(MEAN, STD)])

    def get_data_loaders(self, beta=0, root='/raid/wernsen/clad/data'):
        transform = self.TRANSFORM
        test_transform = transforms.Compose(
            [transforms.ToTensor(), self.get_normalization_transform()])

        from clad.classification.cladc import get_cladc_train, get_cladc_val
        train_set = get_cladc_train(root, transform=transform)

        if beta == 10:
            from torch.utils.data import  ConcatDataset
            combined_train_set = ConcatDataset(train_set)
            train = [DataLoader(combined_train_set, batch_size=10, num_workers=0, shuffle=True)]
        else:
            train = [DataLoader(x, batch_size=10, num_workers=0, shuffle=False) for x in train_set]

        val_set = get_cladc_val(root, transform=test_transform)
        test = [DataLoader(x, batch_size=256, num_workers=0) for x in val_set]

        self.test = test

        return train, test

    @staticmethod
    def get_setting():
        return Namespace(**{
            "batch_size":10,
            "minibatch_size":10,
            "scheduler":"simple",
            "scheduler_rate":0.1,
            "n_epochs":50,
            "pre_epochs":50,
            "opt_steps":[35, 45]})

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), SequentialCLAD.TRANSFORM])
        return transform

    @staticmethod
    def get_backbone(hookme=False):
        return resnet18(SequentialCLAD.N_CLASSES, hookme=hookme)

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize(SequentialCLAD.MEAN, SequentialCLAD.STD)
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize(SequentialCLAD.MEAN, SequentialCLAD.STD)
        return transform


# Example usage
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Sequential PMNIST with GPM')
    parser.add_argument('--batch_size_train', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--batch_size_test', type=int, default=256, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--data_path', type=str, default="data/tiny-imagenet-200/", metavar='',
                        help="")
    parser.add_argument('--class_order', type=str, default="random", metavar='MODEL',
                        help="")
    parser.add_argument('--dataset', type=str, default="tinyimagenet", metavar='',
                        help="")
    parser.add_argument('--pc_valid', default=0.1, type=float,
                        help='fraction of training data used for validation')
    parser.add_argument('--loader', type=str,
                        default="class_incremental_loader",
                        metavar='MODEL',
                        help="Models to be incorporated for the experiment")
    # increment
    parser.add_argument('--increment', type=int, default=5, metavar='S',
                        help='(default: 5)')

    args = parser.parse_args()

    dataset = SequentialCLAD(args)
    train_loader, test_loader = dataset.get_data_loaders()

    for images, targets in train_loader:
        print(images.shape, targets.shape)
        break
