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

    parser.add_argument('--alpha', type=float,
                        help='Penalty weight.', default=0)
    parser.add_argument('--beta', type=float,
                        help='Penalty weight.', default=0)
    parser.add_argument('--charlie', type=float,
                        help='Penalty weight.', default=0)

    return parser


class WSN(ContinualModel):
    NAME = 'wsn'
    COMPATIBILITY = ['domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(WSN, self).__init__(backbone, loss, args, transform)

        if args.distributed != 'ddp':
            self.buffer = Buffer(self.args.buffer_size, self.device)
        else:
            import os
            partial_buf_size = self.args.buffer_size // int(os.environ['MAMMOTH_WORLD_SIZE'])
            print('using partial buf size', partial_buf_size)
            self.buffer = Buffer(partial_buf_size, self.device)
        self.classes = 6
        self.buffers = {
            0: {k: Buffer(round(self.args.buffer_size/self.classes), device=self.device) for k in range(0,self.classes)},
            1: {k: Buffer(round(self.args.buffer_size/self.classes), device=self.device) for k in range(0,self.classes)}
        }
        self.current_task = 0
        self.cpt = get_dataset(args).N_CLASSES_PER_TASK
        self.buffer_backup = None
        self.per_task_masks = {}
        self.consolidated_masks = {}
        self.args = args

    def begin_task(self, dataset):
        # copy buffer
        # self.buffers = {k: Buffer(round(self.args.buffer_size/self.classes), device=self.device) for k in range(0,self.classes)}
        if self.args.update_buffer_at_task_end:
            self.buffer_backup = deepcopy(self.buffer)
            print(f"At task {self.current_task} start after deep copy: buffer is {len(self.buffer)}, buffer_backup is {len(self.buffer_backup)}")
            
    def end_task(self, dataset):
        # Save the per-task-dependent masks
        tmp_mask = self.net.get_masks(self.current_task)
        self.per_task_masks[self.current_task] = tmp_mask

        self.consolidated_masks = deepcopy(self.per_task_masks[self.current_task])
        self.current_task = 1 if self.current_task == 0 else 0

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

    # def train(args, model, device, x, y, optimizer, criterion, task_id_nominal, consolidated_masks, weight_overlap):
    def observe(self, inputs, labels, not_aug_inputs, task_id_nominal, teacher=None, epoch=None, noise=None):
        task_id_nominal = task_id_nominal % 2
        labels = labels.long()
        self.opt.zero_grad()
        try:
            if self.args.alpha == 0:
                mask = self.per_task_masks.get(task_id_nominal)
            else:
                mask = None
        except:
            pass
        outputs = self.net(inputs, task_id_nominal, mask=mask).float()

        # add cl mask to only the first batch, if specified
        loss = self.loss(outputs, labels)

        buf_inputs = []
        buf_labels = []
        for class_id in range(0, self.classes):
            if not self.buffers[task_id_nominal][class_id].is_empty():
                batch_size_buf = 8
                buf_inputs_batch, buf_labels_batch, buf_logits_batch = self.buffers[task_id_nominal][class_id].get_data_old(batch_size_buf, transform=self.transform)
                buf_inputs.append(buf_inputs_batch)
                buf_labels.append(buf_labels_batch)
        if len(buf_inputs) > 0:
            buf_inputs = torch.cat(buf_inputs, dim=0)
            buf_labels = torch.cat(buf_labels, dim=0)

            buf_outputs = self.net(buf_inputs, task_id_nominal, mask=mask).float()
            loss += self.loss(buf_outputs, buf_labels)

        loss.backward()
        # Continual Subnet no backprop
        curr_head_keys = ["last.{}.weight".format(task_id_nominal), "last.{}.bias".format(task_id_nominal)]
        if self.consolidated_masks is not None and self.consolidated_masks != {}:  # Only do this for tasks 1 and beyond
            # if args.use_continual_masks:
            for key in self.consolidated_masks.keys():
                # Skip if not task head is not for curent task
                if 'last' in key:
                    if key not in curr_head_keys:
                        continue

                # Determine wheter it's an output head or not
                key_split = key.split('.')
                if 'last' in key_split or len(key_split) == 2:
                    if 'last' in key_split:
                        module_attr = key_split[-1]
                        task_num = int(key_split[-2])
                        module_name = '.'.join(key_split[:-2])

                    else:
                        module_attr = key_split[1]
                        module_name = key_split[0]

                    # Zero-out gradients
                    if (hasattr(getattr(self.net, module_name), module_attr)):
                        if (getattr(getattr(self.net, module_name), module_attr) is not None):
                            getattr(getattr(self.net, module_name), module_attr).grad[self.consolidated_masks[key] == 1] = 0

                else:
                    module_attr = key_split[-1]

                    # Zero-out gradients
                    curr_module = getattr(getattr(self.net, key_split[0])[int(key_split[1])], key_split[2])
                    if hasattr(curr_module, module_attr):
                        if getattr(curr_module, module_attr) is not None:
                            getattr(curr_module, module_attr).grad[self.consolidated_masks[key] == 1] = 0

        self.opt.step()
        for class_id in range(0, self.classes):
            valid_mask = labels == class_id
            self.buffers[task_id_nominal][class_id].add_data(examples=not_aug_inputs[valid_mask],
                                            labels=labels[valid_mask],
                                            logits=outputs.data[valid_mask])
        return loss.item(), 0, 0, 0, 0