from torch import nn

from datasets import get_dataset
from utils.buffer import Buffer
from torch.nn import functional as F
from models.utils.continual_model import ContinualModel
from utils.args import *
import torch
import numpy as np
from copy import deepcopy


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via  ER')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    add_aux_dataset_args(parser)
    add_gcil_args(parser)
    parser.add_argument('--alpha', type=float,
                        help='Penalty weight.', default=0)
    parser.add_argument('--beta', type=float,
                        help='Penalty weight.', default=0)
    parser.add_argument('--charlie', type=float,
                        help='Penalty weight.', default=0)


    return parser


class ERCLAD(ContinualModel):
    NAME = 'er_clad'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(ERCLAD, self).__init__(backbone, loss, args, transform)
        dataset = get_dataset(self.args)
        self.classes = dataset.N_CLASSES_PER_TASK
        self.buffers = {k: Buffer(round(self.args.buffer_size/self.classes), device=self.device) for k in range(0,self.classes)}

        self.current_task = 0
        self.cpt = get_dataset(args).N_CLASSES_PER_TASK
        self.buffer_backup = None

    def begin_task(self, dataset):
        if self.current_task == 0:
            self.load_initial_checkpoint()
            self.reset_classifier()

        # copy buffer
        if self.args.update_buffer_at_task_end:
            self.buffer_backup = deepcopy(self.buffer)
            print(f"At task {self.current_task} start after deep copy: buffer is {len(self.buffer)}, buffer_backup is {len(self.buffer_backup)}")
            
    def end_task(self, dataset):
        self.current_task += 1 
        # update buffer
        if self.args.update_buffer_at_task_end:
            print(f"At task {self.current_task} end before update: buffer is {len(self.buffer)}, buffer_backup is {len(self.buffer_backup)}")
            self.buffer = self.buffer_backup

    def get_cl_mask(self):
        t = self.current_task
        dataset = get_dataset(self.args)
        cur_classes = np.arange(t*dataset.N_CLASSES_PER_TASK, (t+1)*dataset.N_CLASSES_PER_TASK)
        cl_mask = np.setdiff1d(np.arange(dataset.N_CLASSES_PER_TASK*dataset.N_TASKS), cur_classes)
        return cl_mask

    def mask_output(self, outputs):
        cl_mask = self.get_cl_mask()
        mask_add_on = torch.zeros_like(outputs)
        mask_add_on[:, cl_mask] = float('-inf')
        masked_outputs = mask_add_on + outputs
        return masked_outputs

    def observe(self, inputs: torch.Tensor, labels: torch.Tensor, not_aug_inputs: torch.Tensor, epoch=None, task_id_nominal=None, teacher=None, noise=None):
        labels = labels.long()
        self.opt.zero_grad()
        outputs = self.net(inputs).float()

        # add cl mask to only the first batch, if specified
        if self.args.use_cl_mask:
            masked_outputs = self.mask_output(outputs)
            loss = self.loss(masked_outputs, labels)
        else:
            loss = self.loss(outputs, labels)


        buf_inputs = []
        buf_labels = []
        buf_logits = []
        for class_id in range(0, self.classes):
            if not self.buffers[class_id].is_empty():
                batch_size_buf = 8

                buf_inputs_batch, buf_labels_batch, buf_logits_batch = self.buffers[class_id].get_data_old(batch_size_buf, transform=self.transform)

                buf_inputs.append(buf_inputs_batch)
                buf_labels.append(buf_labels_batch)
                buf_logits.append(buf_logits_batch)

        if len(buf_inputs) > 0:
            buf_inputs = torch.cat(buf_inputs, dim=0)
            buf_labels = torch.cat(buf_labels, dim=0)
            buf_logits = torch.cat(buf_logits, dim=0)
            buf_outputs = self.net(buf_inputs).float()
            loss += self.loss(buf_outputs, buf_labels)

        loss.backward()
        self.opt.step()
        for class_id in range(0, self.classes):
            valid_mask = labels == class_id

            self.buffers[class_id].add_data(examples=not_aug_inputs[valid_mask],
                                            labels=labels[valid_mask],
                                            logits=outputs.data[valid_mask])

        return loss.item(), 0, 0, 0, 0