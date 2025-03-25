#!/usr/bin/env python3
"""
Helper function to initialize a neural network
"""
from typing import Any


import torch.nn as nn
import torch
from backbone.ResNet18 import resnet18

class TeacherNetwork(nn.Module):
    def __init__(self, args, n_tasks, num_cls, dataset=None):
        super(TeacherNetwork, self).__init__()
        self.teachers = nn.ModuleList()
        for t in range(n_tasks):
            if args.teacher_backbone == 'convnet':
                self.teachers.append(fetch_net(args, n_tasks, num_cls, 0.2))
            elif args.teacher_backbone == 'resnet18':
                backbone = resnet18(dataset.N_CLASSES_PER_TASK * dataset.N_TASKS, nf=20)
                self.teachers.append(backbone)

    def forward(self, x, task_id):
        return self.teachers[task_id](x)

def fetch_net(args: Any,
      num_tasks: int,
      num_cls: int,
      dropout: float = 0.3):
    """
    Create a nearal network to train
    """
    if "mnist" in args.dataset:
        inp_chan = 1
        pool = 2
        l_size = 80
    elif "imgsm" in args.dataset:
        inp_chan = 3
        pool = 2
        l_size = 320
        # net = MedConv(num_cls=num_cls,channels=inp_chan, avg_pool=pool)
    elif "tinyimg" in args.dataset:
        inp_chan = 3
        pool = 2
        l_size = 6 * 6 * 80
    elif "miniimg" in args.dataset:
        inp_chan = 3
        pool = 2
        l_size = 8 * 8 * 80
    elif "cifar" in args.dataset:
        inp_chan = 3
        pool = 2
        l_size = 320
    else:
        raise NotImplementedError

    # if args.model == "wrn16_4":
    #     net = WideResNetMultiTask(depth=16, num_task=num_tasks,
    #                               num_cls=num_cls, widen_factor=4,
    #                               drop_rate=dropout, inp_channels=inp_chan)
    # elif args.model == "conv":
    net = SmallConv(num_task=num_tasks, num_cls=num_cls,
                    channels=inp_chan, avg_pool=pool,
                    lin_size=l_size)
    # else:
    #     raise ValueError("Invalid network")

    # if args.gpu_id:
    #     net.cuda()
    return net




class SmallConv(nn.Module):
    """
    Small convolution network with no residual connections
    """
    def __init__(self, num_task=1, num_cls=10, channels=3,
                 avg_pool=2, lin_size=320):
        super(SmallConv, self).__init__()
        self.conv1 = nn.Conv2d(channels, 80, kernel_size=3, bias=False)
        self.conv2 = nn.Conv2d(80, 80, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(80)
        self.conv3 = nn.Conv2d(80, 80, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(80)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(avg_pool)

        self.linsize = lin_size
        self.last = nn.Linear(self.linsize, num_cls)
        # lin_layers = []
        # for task in range(num_task):
        #     lin_layers.append(nn.Linear(self.linsize, num_cls))
        #
        # self.fc = nn.ModuleList(lin_layers)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):

        x = self.conv1(x)
        x = self.maxpool(self.relu(x))

        x = self.conv2(x)
        x = self.maxpool(self.relu(self.bn2(x)))

        x = self.conv3(x)
        x = self.maxpool(self.relu(self.bn3(x)))
        x = x.view(-1, self.linsize)

        logits = self.last(x)

        # for idx, lin in enumerate(self.fc):
        #     task_idx = torch.nonzero((idx == tasks), as_tuple=False).view(-1)
        #     if len(task_idx) == 0:
        #         continue
        #
        #     task_out = torch.index_select(x, dim=0, index=task_idx)
        #     task_logit = lin(task_out)
        #     logits.index_add_(0, task_idx, task_logit)

        return logits
