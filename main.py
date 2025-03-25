import argparse
import numpy as np
import importlib
import os
import sys
import socket
import setproctitle
from backbone.ResNet18 import resnet18
from backbone.alexnet import STLAlexNet
from backbone.convNet import fetch_net, TeacherNetwork
conf_path = os.getcwd()
sys.path.append(conf_path)
from models import get_all_models
from argparse import ArgumentParser
from utils.args import add_management_args, add_gcil_args
from datasets import get_dataset
from models import get_model
from utils.conf import set_random_seed, get_device
from utils import create_if_not_exists
import torch
import uuid
import datetime


def lecun_fix():
    # Yann moved his website to CloudFlare. You need this now
    from six.moves import urllib
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)

def parse_args():
    parser = ArgumentParser(description='mammoth', allow_abbrev=False)
    parser.add_argument('--model', type=str, required=True,
                        help='Model name.', choices=get_all_models())
    parser.add_argument('--epochs', type=int, default=50)

    # torch.set_num_threads(4)
    add_management_args(parser)

    # increment
    parser.add_argument('--increment', type=int, default=5, metavar='S',
                        help='(default: 5)')
    args = parser.parse_known_args()[0]
    mod = importlib.import_module('models.' + args.model)

    get_parser = getattr(mod, 'get_parser')
    parser = get_parser()
    args = parser.parse_args()

    if args.seed is not None:
        set_random_seed(args.seed)

    if args.model == 'mer': setattr(args, 'batch_size', 1)
    return args


def main(args=None):
    if args is None:
        args = parse_args()

    # job number
    args.conf_jobnum = str(uuid.uuid4())
    args.conf_timestamp = str(datetime.datetime.now())
    args.conf_host = socket.gethostname()

    dataset = get_dataset(args)
    if args.dataset == 'seq-clad':
        args.n_tasks = 6

    total_class = dataset.N_CLASSES
    dataset.N_TASKS = args.n_tasks
    dataset.N_CLASSES_PER_TASK = int(total_class/args.n_tasks)

    if args.backbone == 'resnet18_sp':
        backbone = resnet18(dataset.N_CLASSES_PER_TASK * dataset.N_TASKS, nf=68)
    if args.backbone == 'resnet18_lg':
        import torchvision.models as models
        import torch.nn as nn
        # Modify the final fully connected layer for 100 classes
        # num_classes = 100
        # backbone.fc = nn.Linear(backbone.fc.in_features, dataset.N_CLASSES_PER_TASK * dataset.N_TASKS)
        # pretrained_state_dict = pretrained_resnet18.state_dict()
        backbone = resnet18(dataset.N_CLASSES_PER_TASK * dataset.N_TASKS, nf=64)
    if args.backbone == 'resnet18':
        backbone = resnet18(dataset.N_CLASSES_PER_TASK * dataset.N_TASKS, nf=20)
    if args.backbone == 'resnet18_sm':
        backbone = resnet18(dataset.N_CLASSES_PER_TASK * dataset.N_TASKS, nf=10)

    distill_model = None
    if args.model == 'teachers':
        backbone = TeacherNetwork(args, dataset.N_TASKS, dataset.N_TASKS * dataset.N_CLASSES_PER_TASK, dataset)

    if args.charlie > 0:
        if args.teacher_backbone == 'convnet_real':
            distill_backbone = fetch_net(args, dataset.N_TASKS, dataset.N_TASKS * dataset.N_CLASSES_PER_TASK, 0.2)
        elif args.teacher_backbone == 'resnet18_sm':
            distill_backbone = resnet18(dataset.N_CLASSES_PER_TASK * dataset.N_TASKS, nf=10)
        elif args.teacher_backbone == 'resnet18':
            distill_backbone = resnet18(dataset.N_CLASSES_PER_TASK * dataset.N_TASKS, nf=20)
        elif args.teacher_backbone == 'resnet18_lg':
            distill_backbone = resnet18(dataset.N_CLASSES_PER_TASK * dataset.N_TASKS, nf=64)
        distill_loss = dataset.get_loss()
        distill_model = get_model(args, distill_backbone, distill_loss, dataset.get_transform(), "teachers")

    if args.model == 'wsn':
        if args.backbone == 'resnet18_lg':
            from backbone.resnet18_wsn import SubnetResNet18 as ResNet18
            taskcla = [(idx, dataset.N_CLASSES) for idx in range(0, int(dataset.N_TASKS/3))]
            backbone = ResNet18(taskcla, nf=64, sparsity=0.7, size=dataset.SIZE[0])
        if args.backbone == 'resnet18':
            from backbone.resnet18_wsn import SubnetResNet18 as ResNet18
            taskcla = [(idx, dataset.N_CLASSES) for idx in range(0, int(dataset.N_TASKS/3))]
            backbone = ResNet18(taskcla, nf=20, sparsity=0.7, size=dataset.SIZE[0])
        if args.backbone == 'alexnet':
            from backbone.alexnet import SubnetAlexNet as AlexNet
            taskcla = [(idx, dataset.N_CLASSES_PER_TASK * dataset.N_CLASSES_PER_TASK) for idx in
                       range(0, dataset.N_TASKS)]
            backbone = AlexNet(taskcla, sparsity=0.7, size=dataset.SIZE[0])

    loss = dataset.get_loss()
    model = get_model(args, backbone, loss, dataset.get_transform())
    args.distributed = None
    import wandb
    os.environ["WANDB_API_KEY"] = ""
    wandb.init(project=f"{args.name}_{args.dataset}_{args.model}_{args.backbone}_t{args.n_tasks}", config=args)
    # setproctitle.setproctitle(f"{args.dataset[:-4]}_{args.model[:4]}_{args.backbone[-2:]}_c{args.charlie}_{str(args.buffer_size)[:2]}_{args.teacher_backbone[-4:]}{args.seed}")

    from utils.training import train
    train(model, dataset, args, distill_model=distill_model)

if __name__ == '__main__':
    main()

#     ```
#     python main.py --dataset gcil-cifar100 --weight_dist unif --model esmer --buffer_size 200 --load_best_args
#
#     python main.py --dataset gcil-cifar100 --weight_dist longtail --model esmer --buffer_size 200 --load_best_args
#     ```