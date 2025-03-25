# Authorized by Haeyong Kang.

import torch
import torch.nn as nn
import numpy as np
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


class STEMult(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, m):
        ctx.save_for_backward(w)
        return w * m

    @staticmethod
    def backward(ctx, g):
        return g, g * ctx.saved_tensors[0].clone()


def get_none_masks(model):
    none_masks = {}
    for name, module in model.named_modules():
        if isinstance(module, SubnetLinear) or isinstance(module, SubnetConv2d):
            none_masks[name + '.weight'] = None
            none_masks[name + '.bias'] = None


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
#

#######################################################################################
#      GPM ResNet18
#######################################################################################

## Define ResNet18 model
def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
def conv7x7(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=7, stride=stride,
                     padding=1, bias=False)

class GPMBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(GPMBasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, track_running_stats=False)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, track_running_stats=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes, track_running_stats=False)
            )
        self.act = OrderedDict()
        self.count = 0

    def forward(self, x):
        self.count = self.count % 2 
        self.act['conv_{}'.format(self.count)] = x
        self.count +=1
        out = relu(self.bn1(self.conv1(x)))
        self.count = self.count % 2 
        self.act['conv_{}'.format(self.count)] = out
        self.count +=1
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = relu(out)
        return out

class GPMResNet(nn.Module):
    def __init__(self, block, num_blocks, taskcla, nf):
        super(GPMResNet, self).__init__()
        self.in_planes = nf
        self.conv1 = conv3x3(3, nf * 1, 1)
        self.bn1 = nn.BatchNorm2d(nf * 1, track_running_stats=False)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)

        self.taskcla = taskcla
        self.linear=torch.nn.ModuleList()
        for t, n in self.taskcla:
            self.linear.append(nn.Linear(nf * 8 * block.expansion * 4, n, bias=False))
        self.act = OrderedDict()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        bsz = x.size(0)
        self.act['conv_in'] = x.view(bsz, 3, 32, 32)
        out = relu(self.bn1(self.conv1(x.view(bsz, 3, 32, 32)))) 
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        y=[]
        for t,i in self.taskcla:
            y.append(self.linear[t](out))
        return y

def GPMResNet18(taskcla, nf=32):
    return GPMResNet(GPMBasicBlock, [2, 2, 2, 2], taskcla, nf)


#######################################################################################
#       CSNB ResNet18
#######################################################################################

# Multiple Input Sequential
class mySequential(nn.Sequential):
    def forward(self, *inputs):
        mask = inputs[1]
        mode = inputs[2]
        inputs = inputs[0]
        for module in self._modules.values():
            if isinstance(module, SubnetBasicBlock):
                inputs = module(inputs, mask, mode)
            else:
                inputs = module(inputs)

        return inputs

## Define ResNet18 model
def subnet_conv3x3(in_planes, out_planes, stride=1, sparsity=0.5):
    return SubnetConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, sparsity=sparsity)

def subnet_conv7x7(in_planes, out_planes, stride=1, sparsity=0.5):
    return SubnetConv2d(in_planes, out_planes, kernel_size=7, stride=stride,
                     padding=1, bias=False, sparsity=sparsity)

class SubnetBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, sparsity=0.5, name=""):
        super(SubnetBasicBlock, self).__init__()
        self.name = name
        self.affine = True
        self.conv1 = subnet_conv3x3(in_planes, planes, stride, sparsity=sparsity)
        if self.affine:
            self.bn1 = nn.BatchNorm2d(planes, track_running_stats=False, affine=False)
        else:
            self.bn1 = nn.BatchNorm2d(planes, track_running_stats=False)
        self.conv2 = subnet_conv3x3(planes, planes, sparsity=sparsity)
        if self.affine:
            self.bn2 = nn.BatchNorm2d(planes, track_running_stats=False, affine=False)
        else:
            self.bn2 = nn.BatchNorm2d(planes, track_running_stats=False)

        # Shortcut
        self.shortcut = None
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = 1
            self.conv3 = SubnetConv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False, sparsity=sparsity)
            if self.affine:
                self.bn3 = nn.BatchNorm2d(self.expansion * planes, track_running_stats=False, affine=False)
            else:
                self.bn3 = nn.BatchNorm2d(self.expansion * planes, track_running_stats=False)
        self.count = 0

    def forward(self, x, mask, mode='train'):
        name = self.name + ".conv1"
        out = relu(self.bn1(self.conv1(x, weight_mask=mask[name+'.weight'], bias_mask=mask[name+'.bias'], mode=mode)))
        name = self.name + ".conv2"
        out = self.bn2(self.conv2(out, weight_mask=mask[name+'.weight'], bias_mask=mask[name+'.bias'], mode=mode))
        if self.shortcut is not None:
            name = self.name + ".conv3"
            out += self.bn3(self.conv3(x, weight_mask=mask[name+'.weight'], bias_mask=mask[name+'.bias'], mode=mode))
        else:
            out += x
        out = relu(out)
        return out

class SubnetResNet(nn.Module):
    def __init__(self, block, num_blocks, taskcla, nf, sparsity, size=None):
        super(SubnetResNet, self).__init__()
        self.in_planes = nf
        self.size = size
        self.conv1 = subnet_conv3x3(3, nf * 1, 1, sparsity=sparsity)
        self.bn1 = nn.BatchNorm2d(nf * 1, track_running_stats=False, affine=False)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1, sparsity=sparsity, name="layer1")
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2, sparsity=sparsity, name="layer2")
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2, sparsity=sparsity, name="layer3")
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2, sparsity=sparsity, name="layer4")

        self.taskcla = taskcla
        self.last=torch.nn.ModuleList()
        for t, n in self.taskcla:
            self.last.append(nn.Linear(nf * 8 * block.expansion * 4, n, bias=False))
        self.act = OrderedDict()

        # Constant none_masks
        self.none_masks = {}
        for name, module in self.named_modules():
            if isinstance(module, SubnetLinear) or isinstance(module, SubnetConv2d):
                self.none_masks[name + '.weight'] = None
                self.none_masks[name + '.bias'] = None

    def _make_layer(self, block, planes, num_blocks, stride, sparsity, name):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        name_count = 0
        for stride in strides:
            new_name = name + "." + str(name_count)
            layers.append(block(self.in_planes, planes, stride, sparsity, new_name))
            self.in_planes = planes * block.expansion
            name_count += 1
        # return nn.Sequential(*layers)
        return mySequential(*layers)

    def forward(self, x, task_id, mask, mode="train", epoch=1, early_exit=False, joint=False):
        if mask is None:
            mask = self.none_masks

        bsz = x.size(0)
        x = x.reshape(bsz, 3, 64, 64)
        try:
            out = relu(self.bn1(self.conv1(x, weight_mask=mask['conv1.weight'], bias_mask=mask['conv1.bias'], mode=mode)))
        except:
            pass
        out = self.layer1(out, mask, mode, epoch)
        out = self.layer2(out, mask, mode, epoch)
        out = self.layer3(out, mask, mode, epoch)
        out = self.layer4(out, mask, mode, epoch)
        out = avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        if early_exit:
            return out
        y = self.last[task_id](out)
        return y

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

    def init_masks(self, task_id):
        for name, module in self.named_modules():
            # For the time being we only care about the current task outputhead
            if 'last' in name:
                if name != 'last.' + str(task_id):
                    continue

            if isinstance(module, SubnetLinear) or isinstance(module, SubnetConv2d):
                print("{}:reinitialized weight score".format(name))
                module.init_mask_parameters()


def SubnetResNet18(taskcla, nf=32, sparsity=0.5, size=None):
    return SubnetResNet(SubnetBasicBlock, [2, 2, 2, 2], taskcla, nf, sparsity=sparsity, size=size)


#######################################################################################
#      STL ResNet18
#######################################################################################

## Define ResNet18 model
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
def conv7x7(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=7, stride=stride,
                     padding=1, bias=False)

class STLBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(STLBasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, track_running_stats=False)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, track_running_stats=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes, track_running_stats=False)
            )
        self.count = 0

    def forward(self, x):
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = relu(out)
        return out

class STLResNet(nn.Module):
    def __init__(self, block, num_blocks, taskcla, nf, ncla):
        super(STLResNet, self).__init__()
        self.in_planes = nf
        self.conv1 = conv3x3(3, nf * 1, 1)
        self.bn1 = nn.BatchNorm2d(nf * 1, track_running_stats=False)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        
        self.taskcla = taskcla
        self.last = nn.Linear(nf * 8 * block.expansion * 4, ncla, bias=False)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        bsz = x.size(0)
        out = relu(self.bn1(self.conv1(x.view(bsz, 3, 32, 32)))) 
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        y = self.last(out)

        return y

def STLResNet18(taskcla, ncla, nf=32):
    return STLResNet(STLBasicBlock, [2, 2, 2, 2], taskcla, nf, ncla)
