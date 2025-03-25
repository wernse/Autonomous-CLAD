# from datasets.seq_cifar100 import SequentialCIFAR100
from datasets.noisy_seq_cifar100 import NoisySequentialCIFAR100
from datasets.noisy_seq_tinyimagenet import NoisySequentialTinyImagenet
from datasets.seq_clad import SequentialCLAD
from datasets.seq_imagenet import SequentialImagenet
from datasets.seq_miniimagenet import SequentialMiniImagenet
from datasets.seq_tinyimagenet import SequentialTinyImagenet
from datasets.seq_cifar100 import SequentialCIFAR100
from datasets.utils.continual_dataset import ContinualDataset
from argparse import Namespace
from datasets.gcil_cifar100 import GCILCIFAR100

NAMES = {
    SequentialCIFAR100.NAME: SequentialCIFAR100,
    NoisySequentialCIFAR100.NAME: NoisySequentialCIFAR100,
    NoisySequentialTinyImagenet.NAME: NoisySequentialTinyImagenet,
    SequentialTinyImagenet.NAME: SequentialTinyImagenet,
    SequentialMiniImagenet.NAME: SequentialMiniImagenet,
    SequentialImagenet.NAME: SequentialImagenet,
    GCILCIFAR100.NAME: GCILCIFAR100,
    SequentialCLAD.NAME: SequentialCLAD
}


def get_dataset(args: Namespace) -> ContinualDataset:
    """
    Creates and returns a continual dataset.
    :param args: the arguments which contains the hyperparameters
    :return: the continual dataset
    """
    assert args.dataset in NAMES.keys()
    return NAMES[args.dataset](args)
