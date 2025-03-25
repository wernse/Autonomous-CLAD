from collections import defaultdict, Counter
from typing import Dict

from clad.utils.meta import SODA_CATEGORIES
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional
from torch.utils.data import DataLoader
import pandas as pd
import wandb

class AMCAtester:
    """
    Class that calculates the AMCA for the given testset. Every time the model is evaluated it will update
    the results. The AMCA, when calling summarize, is calculated as the average class accuracy over all evaluations
    points. Class accuracy is the mean accuracy of each class.
    """

    def __init__(self, test_loader: torch.utils.data.DataLoader, model: nn.Module, device: str = 'cuda', name="distill"):

        self.loader = test_loader
        self.model = model
        self.device = device
        self.name = name

        self.accs = defaultdict(list)
        self.accs_t1 = defaultdict(list)
        self.loss = defaultdict(list)

        for idx, loader in enumerate(self.loader):
            all_labels = []
            for data, target, non_aug in loader:
                data, target = data.to(self.device), target.to(self.device)
                all_labels.append(target.tolist())
            flattened_list = [item for sublist in all_labels for item in sublist]
            label_counts = Counter(flattened_list)
            print("Day" if idx == 0 else "Night")
            print("-----------Label counts:")
            for label, count in label_counts.items():
                print(f"Label {SODA_CATEGORIES[label]}: {count}")

    def evaluate(self):
        """
        Evaluate the model on the testset.
        """
        for idx, loader in enumerate(self.loader):
            accs, loss = test_cladc(self.model, loader, self.device, task_id=idx)
            wandb.log({f"{self.name}_accs": accs})
            wandb.log({f"{self.name}_loss": loss})
            print("ACC:", idx, accs)
            print("LOSS: ", idx, loss)
            for key, value in loss.items():
                self.loss[key].append(value)
            for key, value in accs.items():
                if idx == 0:
                    self.accs[key].append(value)
                else:
                    self.accs_t1[key].append(value)

    def summarize(self, print_results=True):
        """
        Returns the accuracy and loss dictionaries and AMCA.
        :param print_results: if True, print a formatted version of the results.
        """
        avg_accuracies = {k: np.mean(v) for k, v in self.accs.items()}
        avg_accuracies_t1 = {k: np.mean(v) for k, v in self.accs_t1.items()}
        avg_losses = {k: np.mean(v) for k, v in self.loss.items()}

        amca = np.mean(list(avg_accuracies.values()))
        amca_t1 = np.mean(list(avg_accuracies_t1.values()))
        if self.name == 'student':
            wandb.log({"CIL_acc_mean": amca*100})

        if print_results:
            print('Current class accuracies:')
            print("------DAY------")
            for k, v in sorted(self.accs.items()):
                print(f'{SODA_CATEGORIES[k]:20s}: {v[-1] * 100:.2f}%')
            print(f'AMCA after {len(avg_accuracies)} test points: \n'
                  f'{"AMCA":20s}: {amca * 100:.2f}% \n')
            df = pd.DataFrame(self.accs).T
            df.index = [SODA_CATEGORIES[i] for i in df.index]
            print(df)

            print("------NIGHT------")
            for k, v in sorted(self.accs_t1.items()):
                print(f'{SODA_CATEGORIES[k]:20s}: {v[-1] * 100:.2f}%')
            print(f'AMCA after {len(avg_accuracies_t1)} test points: \n'
                  f'{"AMCA":20s}: {amca_t1 * 100:.2f}% \n')
            df = pd.DataFrame(self.accs_t1).T
            df.index = [SODA_CATEGORIES[i] for i in df.index]
            print(df)

        return avg_accuracies, avg_losses, amca, df


def test_cladc(model: nn.Module, test_loader: DataLoader, device: str = 'cuda', task_id=None) -> [Dict, Dict]:
    """
    Tests a given model on a given dataloader, returns accuracies and losses per class.
    :param model: the model to test.
    :param test_loader: A DataLoader of the testset
    :param device: cuda/cpu device
    :return: Two dictionaries with the accuracies and losses for each class.
    """
    losses, length, correct = defaultdict(float), defaultdict(float), defaultdict(float)
    label_counts = Counter()

    for data, target, non_aug in test_loader:
        data, target = data.to(device), target.to(device)
        try:
            output = model(data)
        except:
            if model.per_task_masks.get(task_id) is None:
                break
            output = model(data, task_id=task_id, mask=model.per_task_masks[task_id])
        loss = torch.nn.functional.cross_entropy(output, target, reduction='none')
        pred = output.argmax(dim=1)

        for lo, pr, ta in zip(loss, pred, target):
            ta = ta.item()
            losses[ta] += lo.item()
            label_counts[ta] += 1  # Count label occurrences
            length[ta] += 1
            if pr.item() == ta:
                correct[ta] += 1

    return {label: correct[label] / length[label] for label in length}, \
           {label: losses[label] / length[label] for label in length}
#ffd0a6
#a9d1e8