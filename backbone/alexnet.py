import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from collections import OrderedDict
from torch.nn.functional import relu, avg_pool2d
# Authorized by Haeyong Kang.

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from collections import OrderedDict
import math
import numpy as np

def percentile(scores, sparsity):
    k = 1 + round(.01 * float(sparsity) * (scores.numel() - 1))
    return scores.view(-1).kthvalue(k).values.item()
class GetSubnetFaster(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, zeros, ones, sparsity):
        k_val = percentile(scores, sparsity * 100)  # 4. calculate the percentile cutoff score based on sparsity.
        if k_val == 0:
            k_val = 0.00000001
        # TODO: FIGURE OUT HOW THE scores KEEP CHANGING! Seems to be based on the magnitude of the weights
        return torch.where(scores < k_val, zeros.to(scores.device),
                           ones.to(scores.device))  # 5. Set scores to 0 if they are less than the cutoff score

    @staticmethod
    def backward(ctx, g):
        return g, None, None, None

class SubnetLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False, sparsity=0.5, trainable=True):
        super(self.__class__, self).__init__(in_features=in_features, out_features=out_features, bias=bias)
        self.sparsity = sparsity
        self.trainable = trainable

        # Mask Parameters of Weights and Bias
        self.w_m = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_mask = None
        self.zeros_weight, self.ones_weight = torch.zeros(self.w_m.shape), torch.ones(self.w_m.shape)
        if bias:
            self.b_m = nn.Parameter(torch.empty(out_features))
            self.bias_mask = None
            self.zeros_bias, self.ones_bias = torch.zeros(self.b_m.shape), torch.ones(self.b_m.shape)
        else:
            self.register_parameter('bias', None)

            # Init Mask Parameters
        self.init_mask_parameters()

        self.Uf = None

        if trainable == False:
            raise Exception("Non-trainable version is not yet implemented")

    def get_gpm(self, x, weights):
        with torch.no_grad():
            # -- GPM ---
            bsz = x.size(0)
            b_idx = range(bsz)
            activation = torch.mm(x[b_idx,], weights.t()).t().cpu().numpy()
            U, S, Vh = np.linalg.svd(activation, full_matrices=False)

            # criteria (Eq-5)
            sval_total = (S ** 2).sum()
            sval_ratio = (S ** 2) / sval_total
            r = np.sum(np.cumsum(sval_ratio) < 0.999)
            feat = U[:, 0:r]
            self.Uf = torch.Tensor(np.dot(feat, feat.transpose())).to(weights.device)

    def infer_mask(self, x, weights, scores, sparsity):
        with torch.no_grad():
            # -- GPM ---
            bsz = x.size(0)
            b_idx = range(bsz)
            activation = torch.mm(x[b_idx,], weights.t()).t().cpu().numpy()
            U, S, Vh = np.linalg.svd(activation, full_matrices=False)

            # criteria (Eq-5)
            sval_total = (S ** 2).sum()
            sval_ratio = (S ** 2) / sval_total
            r = np.sum(np.cumsum(sval_ratio) < 0.999)
            feat = U[:, 0:r]
            Uf = torch.Tensor(np.dot(feat, feat.transpose())).to(weights.device)

            scores = torch.mm(scores.view(bsz, -1), Uf).view(scores.size())

        k_val = percentile(scores, sparsity * 100)
        return torch.where(scores < k_val, zeros.to(scores.device), ones.to(scores.device))

    def forward(self, x, weight_mask=None, bias_mask=None, mode="train", raw=False):
        w_pruned, b_pruned = None, None
        # If training, Get the subnet by sorting the scores
        if raw:
            return F.linear(input=x, weight=self.weight, bias=b_pruned)

        if mode == "train":
            if weight_mask is None:  # Prune based off the random initialised weights. Do we need to store self.w_m?
                w_m_clone = self.w_m.abs().clone()
                self.weight_mask = GetSubnetFaster.apply(w_m_clone,
                                                         self.zeros_weight,
                                                         self.ones_weight,
                                                         self.sparsity)

            else:
                self.weight_mask = weight_mask

            w_pruned = self.weight_mask * self.weight
            b_pruned = None
            if self.bias is not None:
                self.bias_mask = GetSubnetFaster.apply(self.b_m.abs(),
                                                       self.zeros_bias,
                                                       self.ones_bias,
                                                       self.sparsity)
                b_pruned = self.bias_mask * self.bias

        if mode == "joint":
            w_pruned = self.weight
            b_pruned = None

        if mode == "valid":
            if weight_mask is None:
                w_pruned = self.weight
            else:
                w_pruned = weight_mask * self.weight

            b_pruned = None
            if self.bias is not None:
                b_pruned = self.bias_mask * self.bias

                # If inference, no need to compute the subnetwork
        # The mask is filtered on the weights, based on the mask input
        elif mode == "test":
            if weight_mask is None:
                w_pruned = self.weight
            else:
                w_pruned = weight_mask * self.weight
            b_pruned = None
            if self.bias is not None:
                b_pruned = bias_mask * self.bias


        return F.linear(input=x, weight=w_pruned, bias=b_pruned)

    def init_mask_parameters(self):
        nn.init.kaiming_uniform_(self.w_m, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.w_m)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.b_m, -bound, bound)

    def get_taylor(self):
        gammas_pre_norm = self.weight.abs().clone().detach()
        grad_weight = self.weight.grad.abs().clone().detach().data
        taylor = grad_weight * gammas_pre_norm
        return taylor

class SubnetConv2d(nn.Conv2d):
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
        self.w_m = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        self.weight_mask = None
        self.zeros_weight, self.ones_weight = torch.zeros(self.w_m.shape), torch.ones(self.w_m.shape)

        if bias:
            self.b_m = nn.Parameter(torch.empty(out_channels))
            self.bias_mask = None
            self.zeros_bias, self.ones_bias = torch.zeros(self.b_m.shape), torch.ones(self.b_m.shape)
        else:
            self.register_parameter('bias', None)

            # 2. Init Mask Parameters based on vanishing gradients
        self.init_mask_parameters()

        if trainable == False:
            raise Exception("Non-trainable version is not yet implemented")

        self.Uf = None

    def get_gpm(self, x, weights, stride, padding):
        with torch.no_grad():
            # -- GPM ---
            activation = F.conv2d(input=x, weight=weights, bias=None, stride=stride, padding=padding).cpu().numpy()
            # --------------------------
            out_ch, in_ch, ksz, ksz = weights.size()
            bsz, out_ch, sz, sz = activation.shape

            p1d = (1, 1, 1, 1)
            k = 0
            # sf = compute_conv_output_size(activation.shape, ksz, stride, padding)
            b_idx = range(bsz)
            mat = np.zeros((ksz * ksz * in_ch, sz * sz * len(b_idx)))
            act = F.pad(x, p1d, "constant", 0).detach().cpu().numpy()
            for kk in b_idx:
                for ii in range(sz):
                    for jj in range(sz):
                        mat[:, k] = act[kk, :, stride * ii:ksz + stride * ii, stride * jj:ksz + stride * jj].reshape(-1)
                        k += 1
                        # activation
            U, S, Vh = np.linalg.svd(mat, full_matrices=False)
            # criteria (Eq-5)
            sval_total = (S ** 2).sum()
            sval_ratio = (S ** 2) / sval_total
            r = np.sum(np.cumsum(sval_ratio) < 0.945)
            feat = U[:, 0:r]
            self.Uf = torch.Tensor(np.dot(feat, feat.transpose())).to(weights.device)

    def forward(self, x, weight_mask=None, bias_mask=None, mode="train", raw=False):
        w_pruned, b_pruned = None, None
        if raw:
            return F.conv2d(input=x, weight=self.weight, bias=b_pruned, stride=self.stride, padding=self.padding)

        # 3. Get the trimmed weight mask with the w_m scores   If training, Get the subnet by sorting the scores
        if mode == "train":
            if weight_mask is None:
                w_m_clone = self.w_m.abs().clone()
                self.weight_mask = GetSubnetFaster.apply(w_m_clone,  # scores
                                                         self.zeros_weight,  # zeroes
                                                         self.ones_weight,  # ones
                                                         self.sparsity)  # sparsity

            else:
                self.weight_mask = weight_mask

            w_pruned = self.weight_mask * self.weight
            b_pruned = None

            if self.bias is not None:
                self.bias_mask = GetSubnetFaster.apply(self.b_m.abs(), self.zeros_bias, self.ones_bias, self.sparsity)
                b_pruned = self.bias_mask * self.bias

                # If inference/valid, use the last compute masks/subnetworks
        elif mode == "valid":
            if weight_mask is not None:
                self.weight_mask = weight_mask

            w_pruned = self.weight_mask * self.weight
            b_pruned = None
            if self.bias is not None:
                b_pruned = self.bias_mask * self.bias

                # If inference/test, no need to compute the subnetwork
        elif mode == "test":
            if weight_mask is None:
                w_pruned = self.weight
            else:
                w_pruned = weight_mask * self.weight
            # print(torch.sum(w_pruned))
            b_pruned = None
            if self.bias is not None:
                b_pruned = bias_mask * self.bias

        else:
            raise Exception("[ERROR] The mode " + str(mode) + " is not supported!")

        return F.conv2d(input=x, weight=w_pruned, bias=b_pruned, stride=self.stride, padding=self.padding)

    def init_mask_parameters(self):
        print("Initialising Mask Parameters")
        nn.init.kaiming_uniform_(self.w_m, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.w_m)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.b_m, -bound, bound)

    def get_taylor(self):
        gammas_pre_norm = self.weight.abs().clone().detach()
        grad_weight = self.weight.grad.abs().clone().detach().data
        taylor = grad_weight * gammas_pre_norm
        return taylor

## Define AlexNet model
def compute_conv_output_size(Lin, kernel_size, stride=1, padding=0, dilation=1):
    return int(np.floor((Lin + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1))


class STLAlexNet(nn.Module):
    def __init__(self, taskcla):
        super(STLAlexNet, self).__init__()
        self.num_classes = taskcla
        self.in_channel = []
        self.conv1 = nn.Conv2d(3, 64, 4, bias=False)
        s = compute_conv_output_size(32, 4)
        s = s // 2
        self.in_channel.append(3)
        self.conv2 = nn.Conv2d(64, 128, 3, bias=False)
        s = compute_conv_output_size(s, 3)
        s = s // 2
        self.in_channel.append(64)
        self.conv3 = nn.Conv2d(128, 256, 2, bias=False)
        s = compute_conv_output_size(s, 2)
        s = s // 2
        self.smid = s
        self.in_channel.append(128)
        self.maxpool = torch.nn.MaxPool2d(2)
        self.relu = torch.nn.ReLU()
        self.drop1 = torch.nn.Dropout(0.2)
        self.drop2 = torch.nn.Dropout(0.5)

        self.fc1 = nn.Linear(256 * self.smid * self.smid, 2048, bias=False)
        self.fc2 = nn.Linear(2048, 2048, bias=False)

        self.taskcla = taskcla
        self.classifier = torch.nn.Linear(2048, taskcla, bias=False)

    def forward(self, x):
        bsz = deepcopy(x.size(0))
        x = self.conv1(x)
        x = self.maxpool(self.drop1(self.relu(x)))

        x = self.conv2(x)
        x = self.maxpool(self.drop1(self.relu(x)))

        x = self.conv3(x)
        x = self.maxpool(self.drop2(self.relu(x)))

        x = x.reshape(bsz, -1)
        x = self.fc1(x)
        x = self.drop2(self.relu(x))

        x = self.fc2(x)
        x = self.drop2(self.relu(x))

        y = self.classifier(x)
        return y


class SubnetAlexNet(nn.Module):
    def __init__(self, taskcla, sparsity=0.5):
        super(SubnetAlexNet, self).__init__()
        self.in_channel = []
        self.conv1 = SubnetConv2d(3, 64, 4, sparsity=sparsity, bias=False)
        s = compute_conv_output_size(32, 4)
        s = s // 2
        self.in_channel.append(3)
        self.conv2 = SubnetConv2d(64, 128, 3, sparsity=sparsity, bias=False)
        s = compute_conv_output_size(s, 3)
        s = s // 2
        self.in_channel.append(64)
        self.conv3 = SubnetConv2d(128, 256, 2, sparsity=sparsity, bias=False)
        s = compute_conv_output_size(s, 2)
        s = s // 2
        self.smid = s
        self.in_channel.append(128)
        self.maxpool = torch.nn.MaxPool2d(2)
        self.relu = torch.nn.ReLU()
        self.drop1 = torch.nn.Dropout(0.2)
        self.drop2 = torch.nn.Dropout(0.5)

        self.fc1 = SubnetLinear(256 * self.smid * self.smid, 2048, sparsity=sparsity, bias=False)
        self.fc2 = SubnetLinear(2048, 2048, sparsity=sparsity, bias=False)

        self.taskcla = taskcla
        self.last = nn.ModuleList()
        for t, n in self.taskcla:
            self.last.append(nn.Linear(2048, n, bias=False))

            # Constant none_masks
        self.none_masks = {}
        for name, module in self.named_modules():
            if isinstance(module, SubnetLinear) or isinstance(module, SubnetConv2d):
                self.none_masks[name + '.weight'] = None
                self.none_masks[name + '.bias'] = None

    def forward(self, x, task_id, mask, mode="train"):
        if mask is None:
            mask = self.none_masks

        bsz = deepcopy(x.size(0))
        x = self.conv1(x, weight_mask=mask['conv1.weight'], bias_mask=mask['conv1.bias'], mode=mode)
        x = self.maxpool(self.drop1(self.relu(x)))

        x = self.conv2(x, weight_mask=mask['conv2.weight'], bias_mask=mask['conv2.bias'], mode=mode)
        x = self.maxpool(self.drop1(self.relu(x)))

        x = self.conv3(x, weight_mask=mask['conv3.weight'], bias_mask=mask['conv3.bias'], mode=mode)
        x = self.maxpool(self.drop2(self.relu(x)))

        x = x.view(bsz, -1)
        x = self.fc1(x, weight_mask=mask['fc1.weight'], bias_mask=mask['fc1.bias'], mode=mode)
        x = self.drop2(self.relu(x))

        x = self.fc2(x, weight_mask=mask['fc2.weight'], bias_mask=mask['fc2.bias'], mode=mode)
        x = self.drop2(self.relu(x))

        h_keys = ["last.{}.weight".format(task_id), "last.{}.bias".format(task_id)]
        # y = self.last[task_id](x, mask[h_keys[0]], mask[h_keys[1]], mode=mode)
        y = self.last[task_id](x)
        return y

    def init_masks(self):
        for name, module in self.named_modules():
            # For the time being we only care about the current task outputhead
            if 'last' in name:
                if name != 'last.' + str(task_id):
                    continue

            if isinstance(module, SubnetLinear) or isinstance(module, SubnetConv2d):
                print("{}:reinitialized weight score".format(name))
                module.init_mask_parameters()

    def get_masks(self, task_id):
        task_mask = {}
        for name, module in self.named_modules():
            # For the time being we only care about the current task outputhead
            if 'last' in name:
                if name != 'last.' + str(task_id):
                    continue

            if isinstance(module, SubnetLinear) or isinstance(module, SubnetConv2d):

                task_mask[name + '.weight'] = (module.weight_mask.detach().clone() > 0).type(torch.uint8)

                if getattr(module, 'bias') is not None:
                    task_mask[name + '.bias'] = (module.bias_mask.detach().clone() > 0).type(torch.uint8)
                else:
                    task_mask[name + '.bias'] = None
        return task_mask



class SubnetAlexNet_NN_overlap(nn.Module):
    def __init__(self, taskcla, sparsity=0.5):
        super(SubnetAlexNet_NN_overlap, self).__init__()

        self.use_track = False

        self.in_channel = []
        self.conv1 = SubnetConv2d(3, 64, 4, sparsity=sparsity, bias=False)

        if self.use_track:
            self.bn1 = nn.BatchNorm2d(64, momentum=0.1)
        else:
            self.bn1 = nn.BatchNorm2d(64, track_running_stats=False, affine=False)
        s = compute_conv_output_size(32, 4)
        s = s // 2
        self.in_channel.append(3)
        self.conv2 = SubnetConv2d(64, 128, 3, sparsity=sparsity, bias=False)
        if self.use_track:
            self.bn2 = nn.BatchNorm2d(128, momentum=0.1)
        else:
            self.bn2 = nn.BatchNorm2d(128, track_running_stats=False, affine=False)
        s = compute_conv_output_size(s, 3)
        s = s // 2
        self.in_channel.append(64)
        self.conv3 = SubnetConv2d(128, 256, 2, sparsity=sparsity, bias=False)
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

        self.fc1 = SubnetLinear(256 * self.smid * self.smid, 2048, sparsity=sparsity, bias=False)
        if self.use_track:
            self.bn4 = nn.BatchNorm1d(2048, momentum=0.1)
        else:
            self.bn4 = nn.BatchNorm1d(2048, track_running_stats=False, affine=False)
        self.fc2 = SubnetLinear(2048, 2048, sparsity=sparsity, bias=False)

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
            if isinstance(module, SubnetLinear) or isinstance(module, SubnetConv2d):
                self.none_masks[name + '.weight'] = None
                self.none_masks[name + '.bias'] = None

                # TODO: Test uses this function to mask the weights

    def forward(self, x, task_id, mask=None, mode="train", training_masks=None, joint=False, consolidated_masks={}, weight_overlap=None, raw=False):
        # Set for second mas
        if mask is None:
            mask = self.none_masks

        bsz = deepcopy(x.size(0))
        x = self.conv1(x, weight_mask=mask['conv1.weight'], bias_mask=mask['conv1.bias'], mode=mode)

        x = self.maxpool(self.drop1(self.relu(self.bn1(x))))

        x = self.conv2(x, weight_mask=mask['conv2.weight'], bias_mask=mask['conv2.bias'], mode=mode)
        x = self.maxpool(self.drop1(self.relu(self.bn2(x))))

        x = self.conv3(x, weight_mask=mask['conv3.weight'], bias_mask=mask['conv3.bias'], mode=mode)
        x = self.maxpool(self.drop2(self.relu(self.bn3(x))))

        x = x.view(bsz, -1)
        x = self.fc1(x, weight_mask=mask['fc1.weight'], bias_mask=mask['fc1.bias'], mode=mode)
        x = self.drop2(self.relu(self.bn4(x)))

        x = self.fc2(x, weight_mask=mask['fc2.weight'], bias_mask=mask['fc2.bias'], mode=mode)
        x = self.drop2(self.relu(self.bn5(x)))
        y = self.last[task_id](x)

        if raw:
            return x

        return y

    def init_masks(self, task_id):
        for name, module in self.named_modules():
            # For the time being we only care about the current task outputhead
            if 'last' in name:
                if name != 'last.' + str(task_id):
                    continue

            if isinstance(module, SubnetLinear) or isinstance(module, SubnetConv2d):
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

            if isinstance(module, SubnetLinear) or isinstance(module, SubnetConv2d):

                task_mask[name + '.weight'] = (module.weight_mask.detach().clone() > 0).type(torch.uint8)

                if getattr(module, 'bias') is not None:
                    task_mask[name + '.bias'] = (module.bias_mask.detach().clone() > 0).type(torch.uint8)
                else:
                    task_mask[name + '.bias'] = None

        return task_mask


