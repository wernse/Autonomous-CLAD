
from argparse import ArgumentParser
from datasets import NAMES as DATASET_NAMES
from models import get_all_models

def add_gcil_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments required for GCIL-CIFAR100 Dataset.
    :param parser: the parser instance
    """
    # arguments for GCIL-CIFAR100
    parser.add_argument('--gil_seed', type=int, default=1993, help='Seed value for GIL-CIFAR task sampling')
    parser.add_argument('--pretrain', action='store_true', default=False, help='whether to use pretrain')
    parser.add_argument('--phase_class_upper', default=50, type=int, help='the maximum number of classes')
    parser.add_argument('--epoch_size', default=1000, type=int, help='Number of samples in one epoch')
    parser.add_argument('--pretrain_class_nb', default=0, type=int, help='the number of classes in first group')
    parser.add_argument('--weight_dist', default='unif', type=str, help='what type of weight distribution assigned to classes to sample (unif or longtail)')


def add_experiment_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the models.
    :param parser: the parser instance
    """
    parser.add_argument('--dataset', type=str, required=True,
                        choices=DATASET_NAMES,
                        help='Which dataset to perform experiments on.')
    parser.add_argument('--model', type=str, required=True,
                        help='Model name.', choices=get_all_models())
    parser.add_argument('--epochs', type=int, default=50,
                        help='optimizer nesterov momentum.')
    parser.add_argument('--n_tasks', type=int, default=10,
                        help='optimizer nesterov momentum.')
    parser.add_argument('--backbone', type=str, default="resnet18_lg",
                        help='optimizer nesterov momentum.')
    parser.add_argument('--teacher_backbone', type=str, default="convnet",
                        help='optimizer nesterov momentum.')
    parser.add_argument('--finetune', type=int, default=0,
                        help='optimizer nesterov momentum.')
    parser.add_argument('--name', type=str, default="",
                        help='optimizer nesterov momentum.')
    parser.add_argument('--pre_load', type=int, default=0,
                        help='optimizer nesterov momentum.')
    parser.add_argument('--save_teachers', type=int, default=0,
                        help='optimizer nesterov momentum.')
    parser.add_argument('--corrupt_perc', type=float, default=0.2,
                        help='optimizer nesterov momentum.')

    parser.add_argument('--lr', type=float, required=True,
                        help='Learning rate.')
    parser.add_argument('--optim', type=str, default='sgd', choices=['sgd', 'adam'],
                        help='optimizer.')
    parser.add_argument('--optim_wd', type=float, default=0.,
                        help='optimizer weight decay.')
    parser.add_argument('--optim_mom', type=float, default=0.,
                        help='optimizer momentum.')
    parser.add_argument('--optim_nesterov', type=int, default=0,
                        help='optimizer nesterov momentum.')
    parser.add_argument('--quantile', type=float, default=0,
                        help='optimizer nesterov momentum.')
    parser.add_argument('--ancl', type=int, default=0,
                        help='optimizer nesterov momentum.')


    # CL mask
    parser.add_argument('--use_cl_mask', action='store_true', default=False, help='use CL mask or not')
    # use Siam to calculate interact loss
    parser.add_argument('--use_siam', action='store_true', default=False, help='use Siam to calculate interact loss or not')

def add_management_args(parser: ArgumentParser) -> None:
    parser.add_argument('--job_number', type=int, default=None,
                        help='The job ID in Slurm.')
    parser.add_argument('--seed', type=int, default=None,
                        help='The random seed.')
    parser.add_argument('--notes', type=str, default=None,
                        help='Notes for this run.')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID.')

    parser.add_argument('--distill', type=int, default=0)

    parser.add_argument('--non_verbose', action='store_true')

    parser.add_argument('--distributed', default=None,
                        choices=[None, 'dp', 'ddp', 'no', 'post_bt'])

    parser.add_argument('--ignore_other_metrics', type=int, choices=[0, 1], default=0,
                        help='disable additional metrics')
    parser.add_argument('--debug_mode', type=int, default=0, help="If set, run program with partial epochs and no wandb log.")

    parser.add_argument('--disable_log', action='store_true',
                        help='Disable results logging.')
    parser.add_argument('--tensorboard', action='store_true',
                        help='Enable tensorboard logging')
    parser.add_argument('--validation', action='store_true',
                        help='Test on the validation set')
    parser.add_argument('--savecheck', action='store_true',
                        help='Save checkpoint?')
    parser.add_argument('--start_from', type=int, default=None, help="Task to start from")
    parser.add_argument('--stop_after', type=int, default=None, help="Task limit")
    parser.add_argument("--log-filename", default=None, type=str, help='log filename, will override self naming')

def add_rehearsal_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the rehearsal-based methods
    :param parser: the parser instance
    """
    parser.add_argument('--buffer_size', type=int, required=True,
                        help='The size of the memory buffer.')
    parser.add_argument('--update_buffer_at_task_end', action='store_true', default=False,
                        help='If update the buffer only at the end of task.')


def add_aux_dataset_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used to load initial (pretrain) checkpoint
    :param parser: the parser instance
    """
    parser.add_argument('--pre_epochs', type=int, required=False,
                        help='pretrain_epochs.')
    parser.add_argument('--datasetS', type=str, required=False,
                        choices=['cifar100', 'tinyimgR', 'imagenet'])
    parser.add_argument('--load_cp', type=str, default=None)
    parser.add_argument('--stop_after_prep', action='store_true')
