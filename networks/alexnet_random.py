import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from collections import OrderedDict
import torch.nn.functional as F


def compute_conv_output_size(Lin, kernel_size, stride=1, padding=0, dilation=1):
    return int(np.floor((Lin + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1))


def percentile(scores, sparsity):
    k = 1 + round(.01 * float(sparsity) * (scores.numel() - 1))
    return scores.view(-1).kthvalue(k).values.item()


def get_random_subnet(scores, zeros, ones, sparsity):
    k_val = percentile(scores, sparsity * 100)  # 4. calculate the percentile cutoff score based on sparsity.
    if k_val == 0:
        k_val = 0.00000001
    # TODO: FIGURE OUT HOW THE scores KEEP CHANGING! Seems to be based on the magnitude of the weights
    return torch.where(scores < k_val, zeros.to(scores.device), ones.to(scores.device))


class SubnetAlexNet_Random(nn.Module):
    def __init__(self, taskcla, sparsity=0.5):
        super(SubnetAlexNet_Random, self).__init__()

        self.use_track = False

        self.in_channel = []
        self.conv1 = SubnetConv2dRandom(3, 64, 4, sparsity=sparsity, bias=False)

        if self.use_track:
            self.bn1 = nn.BatchNorm2d(64, momentum=0.1)
        else:
            self.bn1 = nn.BatchNorm2d(64, track_running_stats=False, affine=False)
        s = compute_conv_output_size(32, 4)
        s = s // 2
        self.in_channel.append(3)
        self.conv2 = SubnetConv2dRandom(64, 128, 3, sparsity=sparsity, bias=False)
        if self.use_track:
            self.bn2 = nn.BatchNorm2d(128, momentum=0.1)
        else:
            self.bn2 = nn.BatchNorm2d(128, track_running_stats=False, affine=False)
        s = compute_conv_output_size(s, 3)
        s = s // 2
        self.in_channel.append(64)
        self.conv3 = SubnetConv2dRandom(128, 256, 2, sparsity=sparsity, bias=False)
        if self.use_track:
            self.bn3 = nn.BatchNorm2d(256, momentum=0.1)
        else:
            self.bn3 = nn.BatchNorm2d(256, track_running_stats=False, affine=False)
        s = compute_conv_output_size(s, 2)
        s = s // 2
        self.smid = s
        self.in_channel.append(128)
        self.maxpool = torch.nn.MaxPool2d(2)
        self.relu = torch.nn.ReLU()
        self.drop1 = torch.nn.Dropout(0.2)
        self.drop2 = torch.nn.Dropout(0.5)

        self.fc1 = SubnetLinearRandom(256 * self.smid * self.smid, 2048, sparsity=sparsity, bias=False)
        if self.use_track:
            self.bn4 = nn.BatchNorm1d(2048, momentum=0.1)
        else:
            self.bn4 = nn.BatchNorm1d(2048, track_running_stats=False, affine=False)
        self.fc2 = SubnetLinearRandom(2048, 2048, sparsity=sparsity, bias=False)

        if self.use_track:
            self.bn5 = nn.BatchNorm1d(2048, momentum=0.1)
        else:
            self.bn5 = nn.BatchNorm1d(2048, track_running_stats=False, affine=False)

        self.taskcla = taskcla
        self.last = nn.ModuleList()
        for t, n in self.taskcla:
            self.last.append(nn.Linear(2048, n, bias=False))

        # Constant none_masks
        self.none_masks = {}
        for name, module in self.named_modules():
            if isinstance(module, SubnetLinearRandom) or isinstance(module, SubnetConv2dRandom):
                self.none_masks[name + '.weight'] = None
                self.none_masks[name + '.bias'] = None

                # TODO: Test uses this function to mask the weights

    def forward(self, x, task_id, mask=None, mode="train", consolidated_masks={}, weight_overlap=None):
        if mask is None:
            mask = self.none_masks

        bsz = deepcopy(x.size(0))
        x = self.conv1(x, weight_mask=mask['conv1.weight'], bias_mask=mask['conv1.bias'], mode=mode,
                       consolidated_masks=consolidated_masks.get('conv1.weight'), weight_overlap=weight_overlap)

        x = self.maxpool(self.drop1(self.relu(self.bn1(x))))

        x = self.conv2(x, weight_mask=mask['conv2.weight'], bias_mask=mask['conv2.bias'], mode=mode,
                       consolidated_masks=consolidated_masks.get('conv2.weight'), weight_overlap=weight_overlap)
        x = self.maxpool(self.drop1(self.relu(self.bn2(x))))

        x = self.conv3(x, weight_mask=mask['conv3.weight'], bias_mask=mask['conv3.bias'], mode=mode,
                       consolidated_masks=consolidated_masks.get('conv3.weight'), weight_overlap=weight_overlap)
        x = self.maxpool(self.drop2(self.relu(self.bn3(x))))

        x = x.view(bsz, -1)
        x = self.fc1(x, weight_mask=mask['fc1.weight'], bias_mask=mask['fc1.bias'], mode=mode,
                     consolidated_masks=consolidated_masks.get('fc1.weight'), weight_overlap=weight_overlap)
        x = self.drop2(self.relu(self.bn4(x)))

        x = self.fc2(x, weight_mask=mask['fc2.weight'], bias_mask=mask['fc2.bias'], mode=mode,
                     consolidated_masks=consolidated_masks.get('fc2.weight'), weight_overlap=weight_overlap)
        x = self.drop2(self.relu(self.bn5(x)))
        y = self.last[task_id](x)
        return y

    def init_masks(self, task_id):
        for name, module in self.named_modules():
            # For the time being we only care about the current task outputhead
            if 'last' in name:
                if name != 'last.' + str(task_id):
                    continue

            if isinstance(module, SubnetLinearRandom) or isinstance(module, SubnetConv2dRandom):
                print("{}:reinitialized weight score".format(name))
                module.init_mask_parameters()

    def get_masks(self, task_id):
        task_mask = {}
        for name, module in self.named_modules():
            # For the time being we only care about the current task outputhead
            if 'last' in name:
                if name != 'last.' + str(task_id):
                    continue
            if 'adapt' in name:
                continue

            if isinstance(module, SubnetLinearRandom) or isinstance(module, SubnetConv2dRandom):

                task_mask[name + '.weight'] = (module.weight_mask.detach().clone() > 0).type(torch.uint8)

                if getattr(module, 'bias') is not None:
                    task_mask[name + '.bias'] = (module.bias_mask.detach().clone() > 0).type(torch.uint8)
                else:
                    task_mask[name + '.bias'] = None

        return task_mask

    def refresh_weights(self):
        for name, module in self.named_modules():
            # For the time being we only care about the current task outputhead
            if isinstance(module, SubnetLinearRandom) or isinstance(module, SubnetConv2dRandom):
                print("{}:reinitialized weight score".format(name))
                module.refresh_mask()


class SubnetLinearRandom(nn.Linear):
    def __init__(self, in_features, out_features, bias=False, sparsity=0.5, trainable=True):
        super(self.__class__, self).__init__(in_features=in_features, out_features=out_features, bias=bias)
        self.sparsity = sparsity
        self.trainable = trainable

        # Mask Parameters of Weights and Bias
        # self.w_m = torch.empty(out_features, in_features)
        self.w_m = torch.rand(out_features, in_features)
        self.zeros_weight, self.ones_weight = torch.zeros(self.w_m.shape), torch.ones(self.w_m.shape)
        self.register_parameter('bias', None)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.w_m = self.w_m.to(self.device)

    def forward(self, x, weight_mask=None, bias_mask=None, mode="train", consolidated_masks=None, weight_overlap=None):
        w_pruned, b_pruned = None, None
        # If training, Get the subnet by sorting the scores
        if mode == "train":
            if weight_mask is None:  # Prune based off the random initialised weights. Do we need to store self.w_m?
                w_m_clone = self.w_m.abs().clone()
                if consolidated_masks is not None:
                    consolidated_masks_bool = consolidated_masks.bool()
                    w_m_clone[consolidated_masks_bool] = w_m_clone[consolidated_masks_bool] * weight_overlap

                self.weight_mask = get_random_subnet(w_m_clone,
                                                     self.zeros_weight,
                                                     self.ones_weight,
                                                     self.sparsity)

            else:
                self.weight_mask = weight_mask

            w_pruned = self.weight_mask * self.weight
            b_pruned = None

        elif mode == "test":
            if weight_mask is None:
                w_pruned = self.weight
            else:
                w_pruned = weight_mask * self.weight
            b_pruned = None


        return F.linear(input=x, weight=w_pruned, bias=b_pruned)

    def refresh_mask(self):
        self.w_m = torch.rand(self.zeros_weight.shape[0], self.zeros_weight.shape[1])
        self.w_m = self.w_m.to(self.device)


class SubnetConv2dRandom(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, sparsity=0.5,
                 trainable=True, overlap=True):
        super(self.__class__, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
            bias=bias)
        self.stride = stride
        # self.padding = padding
        self.sparsity = sparsity
        self.trainable = trainable

        # Mask Parameters of Weight and Bias 1. define mask
        # self.w_m = torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        self.w_m = torch.rand(out_channels, in_channels, kernel_size, kernel_size)
        self.weight_mask = None
        self.zeros_weight, self.ones_weight = torch.zeros(self.w_m.shape), torch.ones(self.w_m.shape)
        self.register_parameter('bias', None)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.w_m = self.w_m.to(self.device)

    def forward(self, x, weight_mask=None, bias_mask=None, mode="train", epoch=1, consolidated_masks=None,
                weight_overlap=None):
        if mode == "train":
            if weight_mask is None:
                w_m_clone = self.w_m.abs().clone()
                if consolidated_masks is not None:
                    consolidated_masks_bool = consolidated_masks.bool()
                    # Only selects the most important weights
                    w_m_clone[consolidated_masks_bool] = w_m_clone[consolidated_masks_bool] * weight_overlap

                self.weight_mask = get_random_subnet(w_m_clone,  # scores
                                                         self.zeros_weight,  # zeroes
                                                         self.ones_weight,  # ones
                                                         self.sparsity)  # sparsity

            else:
                self.weight_mask = weight_mask
            self.weight_mask = self.weight_mask.to(self.device)
            w_pruned = self.weight_mask * self.weight
            b_pruned = None

        elif mode == "test":
            if weight_mask is None:
                w_pruned = self.weight
            else:
                w_pruned = weight_mask * self.weight
            # print(torch.sum(w_pruned))
            b_pruned = None
        return F.conv2d(input=x, weight=w_pruned, bias=b_pruned, stride=self.stride, padding=self.padding)

    def refresh_mask(self):
        self.w_m = torch.rand(self.zeros_weight.shape[0], self.zeros_weight.shape[1], self.zeros_weight.shape[2], self.zeros_weight.shape[3])
        self.w_m = self.w_m.to(self.device)