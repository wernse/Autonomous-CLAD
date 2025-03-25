
from utils.args import *
from models.utils.continual_model import ContinualModel


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Sequential training (usually a Lower Bound)')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    add_aux_dataset_args(parser)

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


class Sgd(ContinualModel):
    NAME = 'sgd'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(Sgd, self).__init__(backbone, loss, args, transform)
        self.current_task = 0

    def end_task(self, dataset):
        self.current_task += 1

    def begin_task(self, dataset):
        if self.current_task == 0:
            self.load_initial_checkpoint()
            self.reset_classifier()

    def observe(self, inputs, labels, not_aug_inputs, epoch=None, task_id_nominal=None, teacher=None, noise=None):
        labels = labels.long()
        self.opt.zero_grad()
        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()
        self.opt.step()

        return loss.item(),0,0,0,0