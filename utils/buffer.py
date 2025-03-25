import torch
import numpy as np
from typing import Tuple
from torch.functional import Tensor
from torchvision import transforms
from torch.utils.data import Dataset
from copy import deepcopy
from utils.no_bn import bn_track_stats


def icarl_replay(self, dataset, val_set_split=0):
    """
    Merge the replay buffer with the current task data.
    Optionally split the replay buffer into a validation set.
    :param self: the model instance
    :param dataset: the dataset
    :param val_set_split: the fraction of the replay buffer to be used as validation set
    """

    if self.current_task > 0:
        buff_val_mask = torch.rand(len(self.buffer)) < val_set_split
        val_train_mask = torch.zeros(len(dataset.train_loader.dataset.data)).bool()
        val_train_mask[torch.randperm(len(dataset.train_loader.dataset.data))[:buff_val_mask.sum()]] = True

        if val_set_split > 0:
            self.val_loader = deepcopy(dataset.train_loader)

        data_concatenate = torch.cat if isinstance(dataset.train_loader.dataset.data, torch.Tensor) else np.concatenate
        need_aug = hasattr(dataset.train_loader.dataset, 'not_aug_transform')
        if not need_aug:
            def refold_transform(x):
                return x.cpu()
        else:
            data_shape = len(dataset.train_loader.dataset.data[0].shape)
            if data_shape == 3:
                def refold_transform(x):
                    return (x.cpu() * 255).permute([0, 2, 3, 1]).numpy().astype(np.uint8)
            elif data_shape == 2:
                def refold_transform(x):
                    return (x.cpu() * 255).squeeze(1).type(torch.uint8)

        # REDUCE AND MERGE TRAINING SET
        dataset.train_loader.dataset.targets = np.concatenate([
            dataset.train_loader.dataset.targets[~val_train_mask],
            self.buffer.labels.cpu().numpy()[:len(self.buffer)][~buff_val_mask]
        ])
        dataset.train_loader.dataset.data = data_concatenate([
            dataset.train_loader.dataset.data[~val_train_mask],
            refold_transform((self.buffer.examples)[:len(self.buffer)][~buff_val_mask])
        ])

        if val_set_split > 0:
            # REDUCE AND MERGE VALIDATION SET
            self.val_loader.dataset.targets = np.concatenate([
                self.val_loader.dataset.targets[val_train_mask],
                self.buffer.labels.cpu().numpy()[:len(self.buffer)][buff_val_mask]
            ])
            self.val_loader.dataset.data = data_concatenate([
                self.val_loader.dataset.data[val_train_mask],
                refold_transform((self.buffer.examples)[:len(self.buffer)][buff_val_mask])
            ])


def reservoir(num_seen_examples: int, buffer_size: int) -> int:
    """
    Reservoir sampling algorithm.
    :param num_seen_examples: the number of seen examples
    :param buffer_size: the maximum buffer size
    :return: the target index if the current image is sampled, else -1
    """
    if num_seen_examples < buffer_size:
        return num_seen_examples

    rand = np.random.randint(0, num_seen_examples + 1)
    if rand < buffer_size:
        return rand
    else:
        return -1


def ring(num_seen_examples: int, buffer_portion_size: int, task: int) -> int:
    return num_seen_examples % buffer_portion_size + task * buffer_portion_size


class Buffer(Dataset):
    """
    The memory buffer of rehearsal method.
    """

    def __init__(self, buffer_size, device, n_tasks=None, mode='reservoir', gpu_id=None):
        assert mode in ['ring', 'reservoir']
        self.gpu_id = gpu_id
        self.buffer_size = buffer_size
        self.device = device
        self.num_seen_examples = 0
        self.functional_index = eval(mode)
        if mode == 'ring':
            assert n_tasks is not None
            self.task_number = n_tasks
            self.buffer_portion_size = buffer_size // n_tasks
        self.attributes = ['examples', 'labels', 'logits', 'task_labels', 'teacher_logits', 'timestamps', 'is_noise']

        self.attention_maps = [None] * buffer_size
        self.lip_values = [None] * buffer_size

        self.balanced_class_perm = None
        self.transform = None

    def class_stratified_add_data(self, dataset, num_classes=6):
        """
        Add task-specific data to the buffer while maintaining class balance across tasks.
        Handles class imbalance in the current task dataset.
        :param dataset: The dataset with the current task data.
        :param num_classes: Total number of classes (default is 6).
        """

        for data in dataset:
            inputs, labels, _ = data
            self.init_tensors(inputs, labels)
            break

        examples_per_class = self.buffer_size // num_classes
        remaining_slots = self.buffer_size - (examples_per_class * num_classes)

        # Initialize class slots based on available samples in the dataset
        class_counts = torch.zeros(num_classes, dtype=torch.int64)
        for _, labels, _ in dataset:
            for label in labels:
                class_counts[label.item()] += 1

        # Assign slots based on available counts and distribute leftovers
        class_slots = torch.min(class_counts, torch.tensor([examples_per_class] * num_classes))
        leftover_slots = self.buffer_size - class_slots.sum().item()
        for i in range(leftover_slots):
            # Distribute leftover slots to classes with remaining data
            for class_idx in range(num_classes):
                if class_slots[class_idx] < class_counts[class_idx]:
                    class_slots[class_idx] += 1
                    break

        # Reduce buffer to balance classes before adding new data
        if self.num_seen_examples > 0:
            self._reduce_buffer_to_balance(class_slots)

        # Add new data to the buffer
        for data in dataset:
            inputs, labels, _ = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            for idx, label in enumerate(labels):
                class_idx = label.item()
                if class_slots[class_idx] > 0:
                    # Add data to the buffer
                    self.examples[self.num_seen_examples] = inputs[idx]
                    self.labels[self.num_seen_examples] = labels[idx]
                    self.num_seen_examples += 1
                    class_slots[class_idx] -= 1

                if self.num_seen_examples >= self.buffer_size:
                    return  # Stop if buffer is full

        print(f"Added data. Current buffer size: {self.num_seen_examples}/{self.buffer_size}.")
        print(f"Class distribution in buffer: {[torch.sum(self.labels == i).item() for i in range(num_classes)]}")

    def _reduce_buffer_to_balance(self, class_slots):
        """
        Reduce the buffer content to ensure space for new task data while maintaining class balance.
        :param class_slots: The desired number of samples per class.
        """
        for class_idx in range(len(class_slots)):
            indices = torch.where(self.labels == class_idx)[0]
            if len(indices) > class_slots[class_idx]:
                excess = len(indices) - class_slots[class_idx]
                remove_indices = indices[:excess]  # Remove oldest samples first
                self._remove_indices(remove_indices)

    def _remove_indices(self, indices):
        """
        Remove data from the buffer at the specified indices.
        :param indices: Indices of the samples to be removed.
        """
        self.examples = torch.cat([self.examples[:indices[0]], self.examples[indices[-1] + 1:]])
        self.labels = torch.cat([self.labels[:indices[0]], self.labels[indices[-1] + 1:]])
        self.num_seen_examples -= len(indices)

    def generate_class_perm(self):
        self.balanced_class_perm = (self.labels.unique()[torch.randperm(len(self.labels.unique()))]).cpu()
        self.balanced_class_index = 0

    def to(self, device):
        self.device = device
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                setattr(self, attr_str, getattr(self, attr_str).to(device))

        return self

    def __len__(self):
        return min(self.num_seen_examples, self.buffer_size)

    def set_transform(self, transform):
        self.transform = transform

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        if self.transform is None:
            transform = lambda x: x
        else:
            transform = self.transform

        inp = self.examples[index]
        ret_tuple = (transform(inp).to(self.device), inp)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str).to(self.device)
                ret_tuple += (attr[index],)

        return ret_tuple

    def init_tensors(self, examples: torch.Tensor, labels: torch.Tensor,
                     logits=None, task_labels=None, teacher_logits=None, timestamps=None,
                     is_noise=None) -> None:
        """
        Initializes just the required tensors.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        """
        for attr_str in self.attributes:
            attr = eval(attr_str)

            try:
                if attr is not None and not hasattr(self, attr_str):
                    typ = torch.int64 if attr_str.endswith('els') else torch.float32
                    setattr(self, attr_str, torch.zeros((self.buffer_size,
                                                         *attr.shape[1:]), dtype=typ, device=self.device))
            except:
                pass

    def add_data(self, examples, labels=None, logits=None, teacher_logits=None, task_labels=None, attention_maps=None,
                 lip_values=None, timestamps=None, is_noise=None):
        """
        Adds the data to the memory buffer according to the reservoir strategy.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        :return:
        """
        if not hasattr(self, 'examples'):
            self.init_tensors(examples, labels, logits, task_labels, teacher_logits, timestamps, is_noise)

        for i in range(examples.shape[0]):
            index = reservoir(self.num_seen_examples, self.buffer_size)
            self.num_seen_examples += 1
            if index >= 0:
                if self.examples.device != self.device:
                    self.examples.to(self.device)
                self.examples[index] = examples[i].to(self.device)
                if labels is not None:
                    if self.labels.device != self.device:
                        self.labels.to(self.device)
                    self.labels[index] = labels[i].to(self.device)
                if logits is not None:
                    if self.logits.device != self.device:
                        self.logits.to(self.device)
                    self.logits[index] = logits[i].to(self.device)
                if teacher_logits is not None:
                    if self.teacher_logits.device != self.device:
                        self.teacher_logits.to(self.device)
                    self.teacher_logits[index] = teacher_logits[i].to(self.device)
                if task_labels is not None:
                    self.task_labels[index] = task_labels[i].to(self.device)
                if attention_maps is not None:
                    self.attention_maps[index] = [at[i].byte() for at in attention_maps]
                if lip_values is not None:
                    self.lip_values[index] = [val[i].data for val in lip_values]
                if timestamps is not None:
                    self.timestamps[index] = timestamps[i].to(self.device)
                if is_noise is not None:
                    self.is_noise[index] = is_noise[i].to(self.device)

    def get_data_old(self, size: int, transform: transforms = None, return_index=False, to_device=None,
                     task_id_nominal=None, return_non_aug=False) -> Tuple:
        """
        Random samples a batch of size items.
        :param size: the number of requested items
        :param transform: the transformation to be applied (data augmentation)
        :return:
        """
        if size > min(self.num_seen_examples, self.examples.shape[0]):
            size = min(self.num_seen_examples, self.examples.shape[0])

        target_device = self.device if to_device is None else to_device
        choice = np.random.choice(min(self.num_seen_examples, self.examples.shape[0]),
                                  size=size, replace=False)
        if transform is None: transform = lambda x: x
        ret_tuple = (torch.stack([transform(ee.cpu()) for ee in self.examples[choice]]).to(target_device),)

        if return_non_aug:
            ret_tuple += (torch.stack([ee.cpu() for ee in self.examples[choice]]).to(self.device),)

        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                if 'noise' in attr_str:
                    continue
                attr = getattr(self, attr_str).to(target_device)
                ret_tuple += (attr[choice],)

        if not return_index:
            return ret_tuple
        else:
            return (torch.tensor(choice).to(target_device),) + ret_tuple

    def get_data(self, size: int, transform=None, return_index=False, to_device=None, task_id_nominal=None,
                 return_non_aug=False, batch_size_buf=None) -> Tuple:
        """
        Random samples a batch of size items.
        :param size: the number of requested items
        :param transform: the transformation to be applied (data augmentation)
        :return:
        """
        effective_size = min(self.num_seen_examples, self.examples.shape[0])
        size = min(size, effective_size)

        target_device = self.device if to_device is None else to_device

        unique_labels = torch.unique(self.task_labels)
        choice = []

        if len(unique_labels) > 1:
            for label in unique_labels:
                if label == task_id_nominal:
                    continue
                indices = torch.where(self.task_labels == label)[0]
                tmp_batch_size_buf = batch_size_buf if batch_size_buf < len(indices) else len(indices)
                task_choice = np.random.choice(len(indices), size=tmp_batch_size_buf, replace=False)
                choice.extend(np.array(indices[task_choice].cpu()))
        else:
            choice = np.random.choice(min(self.num_seen_examples, self.examples.shape[0]),
                                      size=size, replace=False)
            # Optimize data handling
        selected_examples = self.examples[choice]
        if transform:
            ret_tuple = (torch.stack([transform(ee) for ee in selected_examples]).to(target_device),)
        else:
            ret_tuple = (selected_examples.to(target_device),)

        if return_non_aug:
            ret_tuple += (selected_examples.clone().to(self.device),)

        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                if attr_str == 'task_labels' or 'noise' in attr_str:
                    continue
                attr = getattr(self, attr_str).to(target_device)
                ret_tuple += (attr[choice],)

        if not return_index:
            return ret_tuple
        else:
            return (torch.tensor(choice).to(target_device),) + ret_tuple

    def get_data_by_index(self, indexes: Tensor, transform: transforms = None, to_device=None) -> Tuple:
        """
        Returns the data by the given index.
        :param index: the index of the item
        :param transform: the transformation to be applied (data augmentation)
        :return:
        """
        target_device = self.device if to_device is None else to_device
        if transform is None: transform = lambda x: x
        ret_tuple = (torch.stack([transform(ee.cpu())
                                  for ee in self.examples[indexes]]).to(target_device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str).to(target_device)
                ret_tuple += (attr[indexes],)
        return ret_tuple

    def get_data_balanced(self, n_classes: int, n_instances: int, transform: transforms = None,
                          return_index=False) -> Tuple:
        """
        Random samples a batch of size items.
        :param n_classes: the number of classes to sample
        :param n_instances: the number of instances to be sampled per class
        :param transform: the transformation to be applied (data augmentation)
        :return:
        """
        classes_to_sample = torch.tensor([])
        choice = torch.tensor([]).long()

        while len(classes_to_sample) < n_classes:
            if self.balanced_class_perm is None or \
                    self.balanced_class_index >= len(self.balanced_class_perm) or \
                    len(self.balanced_class_perm.unique()) != len(self.labels.unique()):
                self.generate_class_perm()

            classes_to_sample = torch.cat([
                classes_to_sample,
                self.balanced_class_perm[self.balanced_class_index:self.balanced_class_index + n_classes]
            ])
            self.balanced_class_index += n_classes

        for a_class in classes_to_sample:
            candidates = np.arange(len(self.labels))[self.labels.cpu() == a_class]
            candidates = candidates[candidates < self.num_seen_examples]
            choice = torch.cat([
                choice,
                torch.tensor(
                    np.random.choice(candidates,
                                     size=n_instances,
                                     replace=len(candidates) < n_instances
                                     )
                )
            ])

        if transform is None: transform = lambda x: x
        ret_tuple = (torch.stack([transform(ee.cpu())
                                  for ee in self.examples[choice]]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr[choice],)

        if not return_index:
            return ret_tuple
        else:
            return (choice.to(self.device),) + ret_tuple

    def is_empty(self) -> bool:
        """
        Returns true if the buffer is empty, false otherwise.
        """

        if self.num_seen_examples == 0 or len(self.labels) == 0:
            return True
        else:
            return False

    def get_all_data(self, transform: transforms = None) -> Tuple:
        """
        Return all the items in the memory buffer.
        :param transform: the transformation to be applied (data augmentation)
        :return: a tuple with all the items in the memory buffer
        """
        if transform is None: transform = lambda x: x
        ret_tuple = (torch.stack([transform(ee.cpu())
                                  for ee in self.examples]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr,)
        return ret_tuple

    def empty(self) -> None:
        """
        Set all the tensors to None.
        """
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                delattr(self, attr_str)
        self.num_seen_examples = 0

    def noisy_count(self) -> None:
        """
        Set all the tensors to None.
        """
        if hasattr(self, 'is_noise') is False:
            return None
        total_noise = self.is_noise.sum().item()
        total_count = len(self.labels)
        return (total_noise, total_count)

    def resize_buffer(self, new_size: int):
        """
        Resizes the buffer to contain only 'new_size' elements.
        If the buffer is larger than 'new_size', it randomly removes elements to match the new size.
        If the buffer is smaller, it does nothing.
        :param new_size: The new size of the buffer.
        """
        current_size = min(self.num_seen_examples, self.buffer_size)
        if new_size >= current_size:
            print("Buffer is already smaller than or equal to the requested size. No resizing needed.")
            return

        # Randomly sample indices to keep
        keep_indices = torch.randperm(current_size)[:new_size]

        # Update buffer contents
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                if attr is not None:
                    new_attr = attr[keep_indices].clone()
                    delattr(self, attr_str)  # Remove the old attribute
                    setattr(self, attr_str, new_attr)  # Assign the new attribute

        # Update the count of seen examples
        self.num_seen_examples = new_size
        self.buffer_size = new_size

        print(f"Buffer resized to {new_size} elements.")

