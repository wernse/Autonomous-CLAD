### This is a copy of La-MAML from https://github.com/montrealrobotics/La-MAML
### In order to ensure complete reproducability, we do not change the file and treat it as a baseline.
import argparse
import random

import numpy as np
import torch
from PIL import Image
from sklearn.utils import shuffle
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
from backbone.ResNet18 import resnet18
from datasets.transforms.denormalization import DeNormalize
import torch.nn.functional as F


class DummyDataset(torch.utils.data.Dataset):

    def __init__(self, x, y, trsf, pretrsf = None, imgnet_like = False, super_y = None):
        self.x, self.y = x, y
        self.super_y = super_y

        # transforms to be applied before and after conversion to imgarray
        self.trsf = trsf
        self.pretrsf = pretrsf

        # if not from imgnet, needs to be converted to imgarray first
        self.imgnet_like = imgnet_like
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])  # Add non-augmented transform


    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]
        if self.super_y is not None:
            super_y = self.super_y[idx]

        if(self.pretrsf is not None):
            x = self.pretrsf(x)

        if(not self.imgnet_like):
            x = Image.fromarray(x)

        original_x = self.not_aug_transform(x)  # Non-augmented image
        x = self.trsf(x)

        if self.super_y is not None:
            return x, y, original_x, super_y
        else:
            return x, y, original_x

class DummyArrayDataset(torch.utils.data.Dataset):

    def __init__(self, x, y):
        self.x, self.y = x, y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]

        return x, y

def _get_datasets(dataset_names):
    return [_get_dataset(dataset_name) for dataset_name in [dataset_names]]


def _get_dataset(dataset_name):
    dataset_name = dataset_name.lower().strip()

    if dataset_name == "cifar10":
        return iCIFAR10
    elif dataset_name == "cifar100":
        return iCIFAR100
    elif dataset_name == "seq-tinyimg":
        return iImgnet
    else:
        raise NotImplementedError("Unknown dataset {}.".format(dataset_name))

class DataHandler:
    base_dataset = None
    train_transforms = []
    common_transforms = [transforms.ToTensor()]
    class_order = None


class iImgnet(DataHandler):

    base_dataset = datasets.ImageFolder

    top_transforms = [
        lambda x: Image.open(x[0]).convert('RGB'),
    ]

    train_transforms = [
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip() #,
        #transforms.ColorJitter(brightness=63 / 255)
    ]

    common_transforms = [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ]

    class_order = [
        i for i in range(200)
    ]

import random


# --------
# Datasets CIFAR and TINYIMAGENET
# --------

class IncrementalLoader:
    NAME = 'seq-tinyimg'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 5
    N_TASKS = 40
    N_CLASSES = N_CLASSES_PER_TASK * N_TASKS
    MEAN, STD = (0.4802, 0.4480, 0.3975), (0.2770, 0.2691, 0.2821)
    SIZE = (64, 64)
    TRANSFORM = transforms.Compose(
        [transforms.RandomCrop(64, padding=4),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize(MEAN, STD)])


    @staticmethod
    def get_setting():
        return argparse.Namespace(**{
            "batch_size": 32,
            "minibatch_size": 32,
            "scheduler": "simple",
            "scheduler_rate": 0.1,
            "n_epochs": 50,
            "pre_epochs": 50,
            "opt_steps": [35, 45]})

    @staticmethod
    def get_backbone():
        return resnet18(IncrementalLoader.N_CLASSES_PER_TASK
                        * IncrementalLoader.N_TASKS)

    @staticmethod
    def get_loss():
        return F.cross_entropy

    def get_transform(self):
        transform = transforms.Compose(
            [transforms.ToPILImage(), self.TRANSFORM])
        return transform

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize(IncrementalLoader.MEAN, IncrementalLoader.STD)
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize(IncrementalLoader.MEAN, IncrementalLoader.STD)
        return transform

    @staticmethod
    def get_epochs():
        return 50

    @staticmethod
    def get_batch_size():
        return 32

    def __init__(
            self,
            opt,
            shuffle=True,
            seed=1,
    ):
        self._opt = opt
        dataset_name = opt.dataset
        validation_split = opt.pc_valid
        self.increment = opt.increment

        datasets = _get_datasets(dataset_name)
        self._setup_data(
            datasets,
            class_order_type=opt.class_order,
            seed=seed,
            increment=self.increment,
            validation_split=validation_split
        )
        self.validation_split = validation_split
        self.train_transforms = datasets[0].train_transforms
        self.common_transforms = datasets[0].common_transforms
        self.top_transforms = datasets[0].top_transforms

        self._current_task = 0

        self._batch_size = opt.batch_size_train
        self._test_batch_size = opt.batch_size_test
        self._workers = 0
        self._shuffle = shuffle

        self._setup_test_tasks(validation_split)

    @property
    def n_tasks(self):
        return len(self.increments)

    def set_new_task(self, task_id):
        self._current_task = int(task_id)

    def new_task(self, memory=None, buffer_percentage=None, seed=1):
        if self._current_task >= len(self.increments):
            raise Exception("No more tasks.")

        min_class = sum(self.increments[:self._current_task])
        max_class = sum(self.increments[:self._current_task + 1])
        x_train, y_train = self._select(
            self.data_train, self.targets_train, low_range=min_class, high_range=max_class
        )

        if buffer_percentage:
            x_train, y_train = shuffle(x_train, y_train, random_state=seed)
            max_size = int(len(x_train) * buffer_percentage)
            x_train = x_train[:max_size]
            y_train = y_train[:max_size]

        x_val, y_val = self._select(
            self.data_val, self.targets_val, low_range=min_class, high_range=max_class
        )
        x_test, y_test = self._select(self.data_test, self.targets_test, high_range=max_class)

        if memory is not None:
            data_memory, targets_memory = memory
            print("Set memory of size: {}.".format(data_memory.shape[0]))
            x_train = np.concatenate((x_train, data_memory))
            y_train = np.concatenate((y_train, targets_memory))

        train_loader = self._get_loader(x_train, y_train, mode="train")
        val_loader = self._get_loader(x_val, y_val, mode="train") if len(x_val) > 0 else None
        test_loader = self._get_loader(x_test, y_test, mode="test")

        task_info = {
            "min_class": min_class,
            "max_class": max_class,
            "increment": self.increments[self._current_task],
            "task": self._current_task,
            "max_task": len(self.increments),
            "n_train_data": x_train.shape[0],
            "n_test_data": x_test.shape[0]
        }

        self._current_task += 1

        return task_info, train_loader, val_loader, test_loader

    def _setup_test_tasks(self, validation_split):
        self.test_tasks = []
        self.val_tasks = []
        for i in range(len(self.increments)):
            min_class = sum(self.increments[:i])
            max_class = sum(self.increments[:i + 1])

            x_test, y_test = self._select(self.data_test, self.targets_test, low_range=min_class, high_range=max_class)
            self.test_tasks.append(self._get_loader(x_test, y_test, mode="test"))

            if validation_split > 0.0:
                x_val, y_val = self._select(self.data_val, self.targets_val, low_range=min_class, high_range=max_class)
                self.val_tasks.append(self._get_loader(x_val, y_val, mode="test"))

    def get_tasks(self, dataset_type='test'):
        if dataset_type == 'val':
            if self.validation_split > 0.0:
                return self.val_tasks
            else:
                return self.test_tasks
        elif dataset_type == 'test':
            return self.test_tasks
        else:
            raise NotImplementedError("Unknown mode {}.".format(dataset_type))

    def get_dataset_info(self):
        if (self._opt.dataset == 'seq-tinyimg'):
            n_inputs = 3 * 64 * 64
            input_size = [3, 64, 64]
        else:
            n_inputs = self.data_train.shape[3] * self.data_train.shape[1] * self.data_train.shape[2]
            input_size = [self.data_train.shape[3], self.data_train.shape[1], self.data_train.shape[2]]
        n_outputs = self._opt.increment * len(self.increments)
        n_task = len(self.increments)
        return n_inputs, n_outputs, n_task, input_size

    def _select(self, x, y, low_range=0, high_range=0):
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[idxes], y[idxes]

    def _get_loader(self, x, y, shuffle=True, mode="train"):
        if mode == "train":
            pretrsf = transforms.Compose([*self.top_transforms])
            trsf = transforms.Compose([*self.train_transforms, *self.common_transforms])
            batch_size = self._batch_size
        elif mode == "test":
            pretrsf = transforms.Compose([*self.top_transforms])
            trsf = transforms.Compose(self.common_transforms)
            batch_size = self._test_batch_size
        elif mode == "flip":
            trsf = transforms.Compose(
                [transforms.RandomHorizontalFlip(p=1.), *self.common_transforms]
            )
            batch_size = self._test_batch_size
        else:
            raise NotImplementedError("Unknown mode {}.".format(mode))

        return DataLoader(
            DummyDataset(x, y, trsf, pretrsf, self._opt.dataset == 'seq-tinyimg'),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self._workers
        )

    def _setup_data(self, datasets, class_order_type=False, seed=1, increment=10, validation_split=0.):
        # FIXME: handles online loading of images
        self.data_train, self.targets_train = [], []
        self.data_test, self.targets_test = [], []
        self.data_val, self.targets_val = [], []
        self.increments = []
        self.class_order = []

        current_class_idx = 0  # When using multiple datasets
        for dataset in datasets:

            if (self._opt.dataset == 'seq-tinyimg'):
                root_path = self._opt.data_path
                train_dataset = dataset.base_dataset(root_path + 'train/')
                test_dataset = dataset.base_dataset(root_path + 'val/')

                train_dataset.data = train_dataset.samples
                test_dataset.data = test_dataset.samples

                x_train, y_train = train_dataset.data, np.array(train_dataset.targets)
                x_val, y_val, x_train, y_train = self._list_split_per_class(
                    x_train, y_train, validation_split
                )
                x_test, y_test = test_dataset.data, np.array(test_dataset.targets)

                order = [i for i in range(len(np.unique(y_train)))]
                if class_order_type == 'random':
                    random.seed(seed)  # Ensure that following order is determined by seed:
                    random.shuffle(order)
                    print("Class order:", order)
                elif class_order_type == 'old' and dataset.class_order is not None:
                    order = dataset.class_order
                else:
                    print("Classes are presented in a chronological order")

            else:
                root_path = self._opt.data_path
                train_dataset = dataset.base_dataset(root_path, train=True, download=True)
                test_dataset = dataset.base_dataset(root_path, train=False, download=True)

                x_train, y_train = train_dataset.data, np.array(train_dataset.targets)
                x_val, y_val, x_train, y_train = self._split_per_class(
                    x_train, y_train, validation_split
                )
                x_test, y_test = test_dataset.data, np.array(test_dataset.targets)

                order = [i for i in range(len(np.unique(y_train)))]
                if class_order_type == 'random':
                    random.seed(seed)  # Ensure that following order is determined by seed:
                    random.shuffle(order)
                    print("Class order:", order)
                elif class_order_type == 'old' and dataset.class_order is not None:
                    order = dataset.class_order
                elif class_order_type == 'super' and dataset.class_order_super is not None:
                    order = dataset.class_order_super
                else:
                    print("Classes are presented in a chronological order")

            self.class_order.append(order)

            y_train = self._map_new_class_index(y_train, order)
            y_val = self._map_new_class_index(y_val, order)
            y_test = self._map_new_class_index(y_test, order)

            y_train += current_class_idx
            y_val += current_class_idx
            y_test += current_class_idx

            current_class_idx += len(order)
            if len(datasets) > 1:
                self.increments.append(len(order))
            else:
                self.increments = [increment for _ in range(len(order) // increment)]

            self.data_train.append(x_train)
            self.targets_train.append(y_train)
            self.data_val.append(x_val)
            self.targets_val.append(y_val)
            self.data_test.append(x_test)
            self.targets_test.append(y_test)

            self.data_train = np.concatenate(self.data_train)
            self.targets_train = np.concatenate(self.targets_train)
            self.data_val = np.concatenate(self.data_val)
            self.targets_val = np.concatenate(self.targets_val)
            self.data_test = np.concatenate(self.data_test)
            self.targets_test = np.concatenate(self.targets_test)

    @staticmethod
    def _map_new_class_index(y, order):
        """Transforms targets for new class order."""

        return np.array(list(map(lambda x: order.index(x), y)))

    @staticmethod
    def _split_per_class(x, y, validation_split=0.):
        """Splits train data for a subset of validation data.
        Split is done so that each class has a much data.
        """
        shuffled_indexes = np.random.permutation(x.shape[0])
        x = x[shuffled_indexes]
        y = y[shuffled_indexes]

        x_val, y_val = [], []
        x_train, y_train = [], []

        for class_id in np.unique(y):
            class_indexes = np.where(y == class_id)[0]
            nb_val_elts = int(class_indexes.shape[0] * validation_split)

            val_indexes = class_indexes[:nb_val_elts]
            train_indexes = class_indexes[nb_val_elts:]

            x_val.append(x[val_indexes])
            y_val.append(y[val_indexes])
            x_train.append(x[train_indexes])
            y_train.append(y[train_indexes])

        x_val, y_val = np.concatenate(x_val), np.concatenate(y_val)
        x_train, y_train = np.concatenate(x_train), np.concatenate(y_train)

        return x_val, y_val, x_train, y_train

    @staticmethod
    def _list_split_per_class(x, y, validation_split=0.):
        """Splits train data for a subset of validation data.
        Split is done so that each class has a much data.
        """
        c = list(zip(x, y))
        random.shuffle(c)
        x, y = zip(*c)

        x_val, y_val = [], []
        x_train, y_train = [], []

        for class_id in np.unique(y):
            class_indexes = np.where(y == class_id)[0]
            nb_val_elts = int(class_indexes.shape[0] * validation_split)

            val_indexes = class_indexes[:nb_val_elts]
            train_indexes = class_indexes[nb_val_elts:]

            x_val_i = [x[i] for i in val_indexes]
            y_val_i = [y[i] for i in val_indexes]

            x_train_i = [x[i] for i in train_indexes]
            y_train_i = [y[i] for i in train_indexes]

            x_val.append(x_val_i)
            y_val.append(y_val_i)

            x_train.append(x_train_i)
            y_train.append(y_train_i)

        x_val, y_val = np.concatenate(x_val), np.concatenate(y_val)
        x_train, y_train = np.concatenate(x_train), np.concatenate(y_train)

        return x_val, y_val, x_train, y_train

    def get_idx_data(self, idx, batch_size, mode="test", data_source="train"):
        """Returns a custom loader with specific idxs only.
        :param idx: A list of data indexes that we want.
        :param mode: Various mode for the transformations applied on it.
        :param data_source: Whether to fetch from the train, val, or test set.
        :return: The raw data and a loader.
        """
        if data_source == "train":
            x, y = self.data_train, self.targets_train
        elif data_source == "val":
            x, y = self.data_val, self.targets_val
        elif data_source == "test":
            x, y = self.data_test, self.targets_test
        else:
            raise ValueError("Unknown data source <{}>.".format(data_source))
        y, sorted_idx = y.sort()

        sampler = torch.utils.data.sampler.SubsetRandomSampler(idx)
        trsf = transforms.Compose(self.common_transforms)

        loader = DataLoader(
            DummyDataset(x[sorted_idx], y, trsf),
            sampler=sampler,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self._workers)

    def get_custom_loader(self, class_indexes, mode="test", data_source="train"):
        """Returns a custom loader.
        :param class_indexes: A list of class indexes that we want.
        :param mode: Various mode for the transformations applied on it.
        :param data_source: Whether to fetch from the train, val, or test set.
        :return: The raw data and a loader.
        """
        if not isinstance(class_indexes, list):  # TODO: deprecated, should always give a list
            class_indexes = [class_indexes]

        if data_source == "train":
            x, y = self.data_train, self.targets_train
        elif data_source == "val":
            x, y = self.data_val, self.targets_val
        elif data_source == "test":
            x, y = self.data_test, self.targets_test
        else:
            raise ValueError("Unknown data source <{}>.".format(data_source))

        data, targets = [], []
        for class_index in class_indexes:
            class_data, class_targets = self._select(
                x, y, low_range=class_index, high_range=class_index + 1
            )
            data.append(class_data)
            targets.append(class_targets)

        data = np.concatenate(data)
        targets = np.concatenate(targets)

        return data, self._get_loader(data, targets, shuffle=False, mode=mode)


if __name__ == '__main__':
    # data loader
    parser = argparse.ArgumentParser(description='Sequential PMNIST with GPM')
    parser.add_argument('--batch_size_train', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--batch_size_test', type=int, default=256, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--data_path', type=str, default="data/tiny-imagenet-200/", metavar='',
                        help="")
    parser.add_argument('--class_order', type=str, default="random", metavar='MODEL',
                        help="")
    parser.add_argument('--dataset', type=str, default="seq-tinyimg", metavar='',
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

    loader = IncrementalLoader(args, seed=0)
    n_inputs, n_outputs, n_tasks, input_size = loader.get_dataset_info()
    task_info, train_loader, _, _ = loader.new_task()
    for (k, (v_x, v_y, not_aug_v_x)) in enumerate(train_loader):
        data = v_x
        target = v_y
        not_aug_data = not_aug_v_x
        print(k)

    print(task_info)
