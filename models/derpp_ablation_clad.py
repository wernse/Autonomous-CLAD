import copy
import math

from datasets import get_dataset
from utils.buffer import Buffer
from torch.nn import functional as F
from models.utils.continual_model import ContinualModel
from utils.args import *
import torch
import numpy as np
from copy import deepcopy

from utils.distillery import combine_teachers


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Dark Experience Replay++.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    add_aux_dataset_args(parser)
    add_gcil_args(parser)

    parser.add_argument('--alpha', type=float, required=True,
                        help='Penalty weight.')
    parser.add_argument('--beta', type=float, required=True,
                        help='Penalty weight.')
    parser.add_argument('--charlie', type=float, required=True,
                        help='Train teacher and impact on student.')
    parser.add_argument('--tkd', type=int, default=0,
                        help='Guide with help of student.')
    parser.add_argument('--plasticity', type=int, default=0,
                        help='Guide with help of student.')
    parser.add_argument('--stability', type=int, default=0,
                        help='Guide with help of student.')
    parser.add_argument('--agreement', type=int, default=0,
                        help='Guide with help of student.')
    return parser


class DerppABLATIONCLAD(ContinualModel):
    NAME = 'derpp_ablation_clad'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(DerppABLATIONCLAD, self).__init__(backbone, loss, args, transform)
        dataset = get_dataset(self.args)
        self.classes = dataset.N_CLASSES_PER_TASK
        self.buffers = {k: Buffer(round(self.args.buffer_size/self.classes), device=self.device) for k in range(0,self.classes)}
        self.current_task = 0
        self.cpt = get_dataset(args).N_CLASSES_PER_TASK
        self.buffer_backup = None
        self.labels = set()

    def begin_task(self, dataset):
        self.labels = set()
        if self.current_task == 0:
            self.load_initial_checkpoint()
            self.reset_classifier()

            if self.args.distributed == "post_bt":
                self.net = make_dp(self.net)
        # copy buffer
        if self.args.update_buffer_at_task_end:
            self.buffer_backup = deepcopy(self.buffer)
            print(
                f"At task {self.current_task} start after deep copy: buffer is {len(self.buffer)}, buffer_backup is {len(self.buffer_backup)}")
    # 35.83 [35.4 32.3 39.8  0.   0.   0.   0.   0.   0.   0. ] [52.1 47.5 55.9  0.   0.   0.   0.   0.   0.   0. ]
    # 7k, 1k per task
    # 5k base
    def end_task(self, dataset):
        self.current_task += 1
        print(self.labels)

    def get_cl_mask(self):
        t = self.current_task
        dataset = get_dataset(self.args)
        cur_classes = np.arange(t * dataset.N_CLASSES_PER_TASK, (t + 1) * dataset.N_CLASSES_PER_TASK)
        cl_mask = np.setdiff1d(np.arange(dataset.N_CLASSES_PER_TASK * dataset.N_TASKS), cur_classes)
        return cl_mask

    def mask_output(self, outputs):
        cl_mask = self.get_cl_mask()
        mask_add_on = torch.zeros_like(outputs)
        mask_add_on[:, cl_mask] = float('-inf')
        masked_outputs = mask_add_on + outputs
        return masked_outputs

    def observe(self, inputs: torch.Tensor, labels: torch.Tensor, not_aug_inputs: torch.Tensor, epoch=None, task_id_nominal=None, teacher=None, noise=None):
        self.opt.zero_grad()
        T=2
        seen_labels = set(labels.tolist())
        self.labels = self.labels.union(seen_labels)

        outputs = self.net(inputs)

        if self.args.tkd > 0:
            mask = torch.full(outputs.shape, -float('inf'), device=outputs.device)
            mask[:, list(self.labels)] = 0
            relevant_student = outputs + mask
            loss = self.loss(relevant_student, labels)
        else:
            loss = self.loss(outputs, labels)


        if self.args.charlie > 0:
            teacher_output = teacher(inputs)

            mask = torch.full(outputs.shape, -float('inf'), device=outputs.device)
            mask[:, list(teacher.labels)] = 0
            teacher_masked_logits = teacher_output + mask
            relevant_student = outputs + mask

            if self.args.plasticity > 0:
                output_student = F.log_softmax(relevant_student / T, dim=1)
                output_teacher = F.softmax(teacher_masked_logits / T, dim=1)
                loss += self.args.charlie * F.kl_div(output_student, output_teacher, reduction='batchmean') * (T ** 2)

        buf_inputs = []
        buf_labels = []
        buf_logits = []
        buf_teacher_logits = []
        for class_id in range(0, self.classes):
            if not self.buffers[class_id].is_empty():
                batch_size_buf = 8

                if self.args.charlie > 0:
                    buf_inputs_batch, buf_labels_batch, buf_logits_batch, buf_teacher_logits_batch = self.buffers[
                        class_id].get_data_old(batch_size_buf, transform=self.transform)
                    buf_teacher_logits.append(buf_teacher_logits_batch)
                else:
                    buf_inputs_batch, buf_labels_batch, buf_logits_batch = self.buffers[class_id].get_data_old(batch_size_buf, transform=self.transform)

                buf_inputs.append(buf_inputs_batch)
                buf_labels.append(buf_labels_batch)
                buf_logits.append(buf_logits_batch)

        if len(buf_inputs) > 0:
            buf_inputs = torch.cat(buf_inputs, dim=0)
            buf_labels = torch.cat(buf_labels, dim=0)
            buf_logits = torch.cat(buf_logits, dim=0)

            buf_outputs = self.net(buf_inputs).float()
            if self.args.charlie > 0 and self.args.stability > 0:
                buf_teacher_logits = torch.cat(buf_teacher_logits, dim=0)
                prev_score = buf_logits
                score = buf_teacher_logits
                valid_mask = buf_teacher_logits != float('-inf')

                avg_teacher_logits = torch.where(
                    valid_mask,
                    (prev_score + score) / 2,
                    prev_score
                )
                buf_logits = avg_teacher_logits

            loss += self.args.alpha * F.mse_loss(buf_outputs, buf_logits)

            loss += self.args.beta * self.loss(buf_outputs, buf_labels)

        loss.backward()

        self.opt.step()

        for class_id in range(0, self.classes):
            valid_mask = labels == class_id

            if self.args.charlie > 0:
                self.buffers[class_id].add_data(examples=not_aug_inputs[valid_mask],
                                     labels=labels[valid_mask],
                                     logits=outputs.data[valid_mask],
                                     teacher_logits=teacher_masked_logits.data[valid_mask])
            else:
                self.buffers[class_id].add_data(examples=not_aug_inputs[valid_mask],
                                     labels=labels[valid_mask],
                                     logits=outputs.data[valid_mask])

        return loss.item(), 0, 0, 0, 0

