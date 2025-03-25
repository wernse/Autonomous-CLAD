import torch

from utils.buffer import Buffer
from utils.status import ProgressBar, create_stash, update_status, update_accs
from utils.tb_logger import *
from utils.loggers import *
from utils.loggers import CsvLogger, LossLogger, ExampleLogger, ExampleFullLogger, DictxtLogger
from argparse import Namespace
from models.utils.continual_model import ContinualModel
from datasets.utils.continual_dataset import ContinualDataset
from typing import Tuple
from datasets import get_dataset
from tqdm import tqdm
from datetime import datetime
import sys
import pickle
import math
from copy import deepcopy
import wandb
import torch.optim
def mask_classes(outputs: torch.Tensor, dataset: ContinualDataset, k: int) -> None:
    """
    Given the output tensor, the dataset at hand and the current task,
    masks the former by setting the responses for the other tasks at -inf.
    It is used to obtain the results for the task-il setting.
    :param outputs: the output tensor
    :param dataset: the continual dataset
    :param k: the task index
    """
    outputs[:, 0:k * dataset.N_CLASSES_PER_TASK] = -float('inf')
    outputs[:, (k + 1) * dataset.N_CLASSES_PER_TASK:
               dataset.N_TASKS * dataset.N_CLASSES_PER_TASK] = -float('inf')

def clad_evaluate(tester):
    tester.evaluate()
    tester.summarize(print_results=True)

def evaluate(model: ContinualModel, dataset: ContinualDataset, last=False, example_logger=None, verbose=False) -> Tuple[
    list, list]:
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    status = model.net.training
    model.net.eval()
    accs = np.zeros((dataset.N_TASKS,))
    accs_mask_classes = np.zeros((dataset.N_TASKS,))
    iterator = enumerate(dataset.test_loaders)
    for k, test_loader in iterator:
        if last and k < len(dataset.test_loaders) - 1:
            continue
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        for idx, data in enumerate(test_loader):
            with torch.no_grad():
                if len(data) == 3:
                    inputs, labels, _ = data
                else:
                    inputs, labels = data
                inputs, labels = inputs.to(
                    model.device), labels.to(model.device)
                if 'wsn' in model.NAME:
                    outputs = model(inputs, task_id=k, mask=model.per_task_masks.get(k))
                elif 'class-il' not in model.COMPATIBILITY:
                    outputs = model(inputs, task_id=k)
                else:
                    outputs = model(inputs)  # [:,0:100]

            _, pred = torch.max(outputs.data, 1)
            matches = pred == labels
            correct += torch.sum(matches).item()

            total += labels.shape[0]

            if dataset.SETTING == 'class-il':
                mask_classes(outputs, dataset, k)
                _, pred = torch.max(outputs.data, 1)
                masked_matches = pred == labels
                correct_mask_classes += torch.sum(masked_matches).item()
                if example_logger and type(example_logger) == ExampleLogger:
                    example_logger.log_batch(
                        k, idx, masked_matches.cpu().numpy().tolist(), masked_classes=True)

        # accs.append(correct / total * 100
        #             if 'class-il' in model.COMPATIBILITY else 0)
        # accs_mask_classes.append(correct_mask_classes / total * 100)
        # accs.append(correct / total * 100)
        accs[k] = round(correct / total * 100, 2)
        # accs_mask_classes.append(correct_mask_classes / total * 100)
        accs_mask_classes[k] = round(correct_mask_classes / total * 100, 2)

    model.net.train(status)
    # print(f"Task {idx}, Average loss {test_loss:.4f}, Class inc Accuracy {acc:.3f}, Task inc Accuracy {til_acc:.3f}")
    # print(f"Task {idx}, Class inc Accuracy {accs:.3f}")

    return accs, accs_mask_classes


def count_grads_gt_zero(model):
    layers_with_non_zero_gradients = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            count = torch.sum(param.grad > 0).item()  # count gradients > 0
            if count > 0:
                layers_with_non_zero_gradients[name] = count
    return layers_with_non_zero_gradients


def get_cl_mask(current_task, args):
    # t = self.current_task
    t = current_task
    # dataset = get_dataset(self.args)
    dataset = get_dataset(args)
    cur_classes = np.arange(t * dataset.N_CLASSES_PER_TASK, (t + 1) * dataset.N_CLASSES_PER_TASK)
    cl_mask = np.setdiff1d(np.arange(dataset.N_CLASSES_PER_TASK * dataset.N_TASKS), cur_classes)
    return cl_mask


def mask_output(outputs, current_task, args):
    cl_mask = get_cl_mask(current_task, args)
    mask_add_on = torch.zeros_like(outputs)
    mask_add_on[:, cl_mask] = float('-inf')
    masked_outputs = mask_add_on + outputs
    return masked_outputs


def compute_average_logit(model: ContinualModel, dataset: ContinualDataset, subsample: float):
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    status = model.net.training
    model.net.eval()
    prio = torch.zeros(dataset.N_CLASSES_PER_TASK *
                       dataset.N_TASKS).to(model.device)
    c = 0
    for k, test_loader in enumerate(dataset.test_loaders):
        for idx, data in enumerate(test_loader):
            if idx / len(test_loader) > subsample:
                break
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(
                    model.device), labels.to(model.device)
                if 'class-il' not in model.COMPATIBILITY:
                    outputs = model(inputs, k)
                else:
                    outputs = model(inputs)
                prio += outputs.sum(0)
                c += len(outputs)
    model.net.train(status)
    return prio.cpu() / c


def noise_acc(model, train_loader):
    overall_correct, overall_noise = 0,0
    total, total_noise = 0, 0
    correct, correct_noise = 0, 0
    for i, data in enumerate(train_loader):
        inputs, labels, not_aug_inputs, is_noise = data
        inputs, labels = inputs.to(model.device), labels.to(model.device)
        outputs = model(inputs)
        _, pred = torch.max(outputs.data, 1)
        matches = pred == labels
        overall_correct += torch.sum(matches)

        noisy_matches = matches[is_noise]
        correct_noise += torch.sum(noisy_matches).item()

        non_noisy_matches = matches[~is_noise]
        correct += torch.sum(non_noisy_matches).item()

        total += non_noisy_matches.shape[0]
        total_noise += noisy_matches.shape[0]

    acc = correct / total * 100
    acc_noise = correct_noise / total_noise * 100

    return acc, acc_noise


def train(model: ContinualModel, dataset: ContinualDataset, args: Namespace, distill_model=None) -> None:
    # global sig_pause
    """
    The training process, including evaluations and loggers.
    :param model: the module to be trained
    :param dataset: the continual dataset at hand
    :param args: the arguments of the current execution
    """

    print(args)

    dataset_setting = dataset.get_setting()
    model.net.to(model.device)
    if distill_model:
        distill_model.net.to(distill_model.device)
    results, results_mask_classes = [], []
    total_til = []
    total_cil = []
    total_noise_acc, total_teacher_noise_acc = [], []
    total_non_noise_acc, total_teacher_non_noise_acc = [], []
    if not args.disable_log:
        logger = DictxtLogger(dataset.SETTING, dataset.NAME, model.NAME)

    # independent log file
    log_filename = 'data/logfile/test.txt'
    print(log_filename)

    log_filename_dir_str = log_filename.split('/')
    log_filename_dir = "/".join(log_filename_dir_str[:-1])
    if not os.path.exists(log_filename_dir):
        os.system('mkdir -p ' + log_filename_dir)
        print("New folder {} created...".format(log_filename_dir))

    # log args first
    with open(log_filename, 'a') as f:
        for arg in sorted(vars(args)):
            f.write("{}:".format(arg))
            f.write("{}".format(getattr(args, arg)))
            f.write("\n")

    progress_bar = ProgressBar(verbose=not args.non_verbose)

    model_stash = create_stash(model, args, dataset)

    if args.tensorboard:
        tb_logger = TensorboardLogger(args, dataset.SETTING, model_stash)
        model_stash['tensorboard_name'] = tb_logger.get_name()

    print(file=sys.stderr)

    acc_matrix = np.zeros((dataset.N_TASKS, dataset.N_TASKS))
    acc_matrix_new = np.zeros((dataset.N_TASKS, dataset.N_TASKS))

    max_acc_at_last_task = 0.0
    max_log_line = ''
    is_noise = None
    print(f'Teacher Memory Utilization: {torch.cuda.max_memory_allocated(device=f"cuda:{args.gpu_id}") / (1024 ** 2)} MB')

    if args.dataset == 'seq-clad':
        train_loaders, test_loader = dataset.get_data_loaders(args.beta)
        from clad.utils.test_cladc import AMCAtester
        distill_tester = AMCAtester(dataset.test, distill_model, model.device, name="teacher")
        tester = AMCAtester(dataset.test, model, model.device, name="student")


    for t in range(0 if args.start_from is None else args.start_from,
                   dataset.N_TASKS if args.stop_after is None else args.stop_after):
        if args.dataset == 'seq-clad':
            train_loader = train_loaders[t]
        else:
            train_loader, test_loader = dataset.get_data_loaders()


        if args.charlie > 0:

            if args.pre_load == 1:
                if 'noisy' in args.dataset:
                    checkpoint = torch.load(
                        f'distill_checkpoints/s{args.seed}_t{t}_{args.dataset}_{args.teacher_backbone}_{args.corrupt_perc}',
                        map_location='cpu'
                    )
                else:
                    checkpoint = torch.load(
                        f'distill_checkpoints/s{args.seed}_t{t}_{args.dataset}_{args.teacher_backbone}',
                        map_location='cpu'
                    )
                model_state_dict = checkpoint.get('model_state_dict')
                distill_model.net.load_state_dict(model_state_dict)
                accs, til_accs = evaluate(distill_model, dataset)
                print(f"Preloaded TEACHER for t{t} Î-> accs {accs[t]} | til_accs {round(til_accs[t], 2)}")
            else:
                print("-----------BEGIN Teacher process-----------")
                distill_model.net.train()
                if hasattr(distill_model, 'begin_task'):
                    distill_model.begin_task(dataset)

                distill_model.evaluator = lambda: evaluate(distill_model, dataset)
                distill_model.evaluate = lambda dataset: evaluate(distill_model, dataset)
                distill_model.opt = torch.optim.SGD(distill_model.net.parameters(
                ), lr=args.lr, weight_decay=args.optim_wd, momentum=args.optim_mom)
                if dataset_setting.scheduler_rate is not None:
                    scheduler = torch.optim.lr_scheduler.MultiStepLR(
                        distill_model.opt, dataset_setting.opt_steps, gamma=dataset_setting.scheduler_rate, verbose=True)
                else:
                    scheduler = None

                for epoch in range(args.epochs):
                    for i, data in enumerate(train_loader):
                        if len(data) == 3:
                            inputs, labels, not_aug_inputs = data
                        else:
                            inputs, labels, not_aug_inputs, is_noise = data
                        inputs, labels = inputs.to(distill_model.device), labels.to(distill_model.device)
                        not_aug_inputs = not_aug_inputs.to(distill_model.device)
                        loss, loss_stream, loss_buff, loss_streamu, loss_buffu = distill_model.observe(inputs, labels,
                                                                                                       not_aug_inputs,
                                                                                                       task_id_nominal=t)
                        # if args.dataset == 'seq-img':
                        #     not_aug_inputs, labels = data
                        #     inputs = [SequentialImagenet(args).TRANSFORM(x) for x in not_aug_inputs]
                        assert not math.isnan(loss)

                        progress_bar.prog(i, len(train_loader), epoch, t, loss, loss_stream, animated_bar=False)

                        if i % 100 == 0:
                            update_status(i, len(train_loader), epoch, t,
                                          loss, job_number=args.job_number)

                    if scheduler is not None:
                        scheduler.step()

                    accs, til_accs = evaluate(distill_model, dataset)
                    wandb.log({"CIL_distill_acc": accs, "til_distill_acc": til_accs})
                    print(accs)

                    if is_noise is not None:
                        acc_teacher, acc_noise_teacher = noise_acc(distill_model, train_loader)
                        print(round(acc_teacher, 2), round(acc_noise_teacher,2))
                        total_teacher_non_noise_acc.append(acc_teacher)
                        total_teacher_noise_acc.append(acc_noise_teacher)
                        wandb.log({
                            'total_teacher_non_noise_acc': total_teacher_non_noise_acc,
                            'total_teacher_noise_acc': total_teacher_noise_acc
                        })
                    if args.dataset == 'seq-clad':
                        clad_evaluate(distill_tester)

                if hasattr(distill_model, 'end_task'):
                    distill_model.end_task(dataset)

            if args.save_teachers > 0:
                if 'noisy' in args.dataset:
                    torch.save({
                        'model_state_dict': distill_model.net.state_dict(),
                    }, f'distill_checkpoints/s{args.seed}_t{t}_{args.dataset}_{args.teacher_backbone}')
                else:
                    torch.save({
                        'model_state_dict': distill_model.net.state_dict(),
                    }, f'distill_checkpoints/s{args.seed}_t{t}_{args.dataset}_{args.teacher_backbone}_{args.corrupt_perc}')
                continue
        print("-----------BEGIN TRAINING process-----------")
        print(f'Teacher Memory Utilization: {torch.cuda.max_memory_allocated(device=f"cuda:{args.gpu_id}") / (1024 ** 2)} MB')
        model.net.train()
        if hasattr(model, 'begin_task'):
            model.begin_task(dataset)

        model.evaluator = lambda: evaluate(model, dataset)
        model.evaluate = lambda dataset: evaluate(model, dataset)

        model.opt = torch.optim.SGD(model.net.parameters(
        ), lr=args.lr, weight_decay=args.optim_wd, momentum=args.optim_mom)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            model.opt, dataset_setting.opt_steps, gamma=dataset_setting.scheduler_rate, verbose=True)

        # Train WSN first, then train Distil
        for epoch in range(args.epochs):
            mask_count = 0
            noise_count = 0
            included_noise_count = 0
            total = 0
            is_noise = None
            for i, data in enumerate(train_loader):
                if len(data) == 3:
                    inputs, labels, not_aug_inputs = data
                else:
                    inputs, labels, not_aug_inputs, is_noise = data
                total += len(labels)
                inputs, labels = inputs.to(model.device), labels.to(model.device)
                not_aug_inputs = not_aug_inputs.to(model.device)
                loss, aux_loss, loss_buff, loss_streamu, loss_buffu = model.observe(inputs, labels, not_aug_inputs, task_id_nominal=t, teacher=distill_model, epoch=epoch, noise=is_noise)
                if loss_buff != 0 and len(loss_buff) > 0:
                    mask_count += loss_buff.get('mask')
                    noise_count += loss_buff.get('noise')
                    included_noise_count += loss_buff.get('i_noise')
                assert not math.isnan(loss)

                progress_bar.prog(i, len(train_loader), epoch, t, loss, aux_loss, animated_bar=False)

                if i % 100 == 0 and not 'MAMMOTH_SLAVE' in os.environ:
                    update_status(i, len(train_loader), epoch, t,
                                  loss, job_number=args.job_number, aux_loss=aux_loss)

            if scheduler is not None:
                scheduler.step()


            print(f'Teacher Memory Utilization: {torch.cuda.max_memory_allocated(device=f"cuda:{args.gpu_id}") / (1024 ** 2)} MB')
            accs, til_accs = evaluate(model, dataset,
                                      verbose=not model.args.non_verbose)
            acc_mean = round(np.mean(accs[:t + 1]), 2)
            wandb.log({"CIL_acc": accs, "til_acc": til_accs, "CIL_acc_mean": acc_mean})
            total_til.append(list(til_accs))
            total_cil.append(list(accs))
            wandb.log({"total_til": total_til})
            wandb.log({"total_cil": total_cil})
            print(acc_mean, accs, til_accs)
            if args.charlie > 0:
                print("total", total, "mask_count", mask_count, "noise_count", noise_count, "included_noise_count", included_noise_count)


            if is_noise is not None and hasattr(model, 'buffer') and hasattr(model.buffer, 'is_noise'):
                acc, acc_noise = noise_acc(model, train_loader)
                print(round(acc, 2), round(acc_noise, 2))
                total_non_noise_acc.append(acc)
                total_noise_acc.append(acc_noise)
                wandb.log({
                    'total_noise_acc': total_noise_acc,
                    'total_non_noise_acc': total_non_noise_acc
                })
            if hasattr(model, 'end_epoch'):
                model.end_epoch(epoch + 1, dataset)


        if 'ancl' in args.model:
            print("-----------BEGIN ANCL NORMAL process-----------")
            model.reset_ancl()
            model.net.train()
            model.opt = torch.optim.SGD(model.net.parameters(
            ), lr=args.lr, weight_decay=args.optim_wd, momentum=args.optim_mom)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                model.opt, dataset_setting.opt_steps, gamma=dataset_setting.scheduler_rate, verbose=True)
            for epoch in range(args.epochs):
                total = 0
                is_noise = None
                for i, data in enumerate(train_loader):
                    inputs, labels, not_aug_inputs = data
                    total += len(labels)
                    inputs, labels = inputs.to(model.device), labels.to(model.device)
                    not_aug_inputs = not_aug_inputs.to(model.device)
                    loss, aux_loss, loss_buff, loss_streamu, loss_buffu = model.observe(inputs, labels, not_aug_inputs,
                                                                                        task_id_nominal=t,
                                                                                        teacher=model.ancl_model,
                                                                                        epoch=epoch, noise=is_noise)
                    assert not math.isnan(loss)

                    progress_bar.prog(i, len(train_loader), epoch, t, loss, aux_loss, animated_bar=False)

                    if i % 100 == 0 and not 'MAMMOTH_SLAVE' in os.environ:
                        update_status(i, len(train_loader), epoch, t,
                                      loss, job_number=args.job_number, aux_loss=aux_loss)

                scheduler.step()

                accs, til_accs = evaluate(model, dataset,
                                          verbose=not model.args.non_verbose)


                acc_mean = round(np.mean(accs[:t + 1]), 2)
                wandb.log({"CIL_acc": accs, "til_acc": til_accs, "CIL_acc_mean": acc_mean})
                print(acc_mean, accs, til_accs)

                if hasattr(model, 'end_epoch'):
                    print(f'Teacher Memory Utilization: {torch.cuda.max_memory_allocated(device=f"cuda:{args.gpu_id}") / (1024 ** 2)} MB')
                    model.end_epoch(epoch + 1, dataset)


        if hasattr(model, 'end_task'):
            # print(f'Training Memory Utilization: {torch.cuda.max_memory_allocated(device=None) / (1024 ** 2)} MB')
            model.end_task(dataset)

        if args.dataset == 'seq-clad':
            clad_evaluate(tester)
        model_stash['task_idx'] = t + 1
        model_stash['epoch_idx'] = 0
        # torch.save({
        #     'model_state_dict': model.net.state_dict(),
        # }, f'final_checkpoints/s{args.seed}_t{t}_c{args.charlie}_{args.dataset}_{args.teacher_backbone}_{args.model}')
        # possible checkpoint saving
        if t == dataset.N_TASKS - 1:
            if args.dataset == 'seq-clad':
                if 'wsn' in args.model:
                    torch.save({
                        'model_state_dict': model.net.state_dict(),
                        'per_task_masks': model.per_task_masks,
                    }, f'clad_checkpoints/s{args.seed}_{args.dataset}_{args.model}_a{args.alpha}_c{args.charlie}')
                else:
                    torch.save({
                        'model_state_dict': model.net.state_dict(),
                    }, f'clad_checkpoints/s{args.seed}_{args.dataset}_{args.model}_a{args.alpha}_c{args.charlie}')


            # checkpoint = torch.load(f'clad_checkpoints/s{args.seed}_{args.dataset}_{args.model}_a{args.alpha}_c{args.charlie}',
            #                         map_location='cpu')
            # model_state_dict = checkpoint.get('model_state_dict')
            # model.net.load_state_dict(model_state_dict)

            accs = evaluate(model, dataset,
                            verbose=not model.args.non_verbose)
            print(accs)

            acc_list, til_acc_list = accs[0], accs[1]
            prec1 = sum(acc_list) / (t + 1)
            til_prec1 = sum(til_acc_list) / (t + 1)
            acc_matrix[t] = acc_list
            acc_matrix_new[t] = til_acc_list
            # wandb.log({'acc_matrix': acc_matrix})
            # wandb.log({'acc_list': acc_list})
            # wandb.log({'til_acc_list': til_acc_list})
            forgetting = np.mean((np.max(acc_matrix, axis=0) - acc_list)[:t]) if t > 0 else 0.0
            forgetting_new = np.mean((np.max(acc_matrix_new, axis=0) - acc_list)[:t]) if t > 0 else 0.0
            learning_acc = np.mean(np.diag(acc_matrix)[:t + 1])

            log_line = 'Training on examples\n'
            log_line += f"Task: {t + 1}, Epoch:{dataset_setting.n_epochs - 1}, Average Acc:[{prec1:.3f}], Task Inc Acc:[{til_prec1:.3f}], Learning Acc:[{learning_acc:.3f}], Forgetting:[{forgetting:.3f}] Forgetting NEW:[{forgetting_new:.3f}]\n"
            log_line += "\t"
            for i in range(t + 1):
                log_line += f"Acc@T{i}: {acc_list[i]:.3f}\t"
            log_line += "\n"
            log_line += "\t"
            for i in range(t + 1):
                log_line += f"Til-Acc@T{i}: {til_acc_list[i]:.3f}\t"
            log_line += "\n"
            print(log_line)
            with open(log_filename, 'a') as f:
                f.write(log_line)
                f.write("\n")

            # update and log the max_log_line
            if t == dataset.N_TASKS - 1:
                if prec1 > max_acc_at_last_task:
                    max_acc_at_last_task = prec1
                    max_log_line = log_line

                max_log_line = "Epoch with max average acc:\n" + max_log_line
                print(max_log_line)
                with open(log_filename, 'a') as f:
                    f.write(max_log_line)
                    f.write("\n")

            results.append(accs[0])
            results_mask_classes.append(accs[1])

            mean_acc = np.array([prec1, til_prec1], dtype=np.float64)
            update_accs(mean_acc, dataset.SETTING, args.job_number)
            print_mean_accuracy(mean_acc, t + 1, dataset.SETTING)
            model_stash['mean_accs'].append(mean_acc)
            # possible checkpoint saving

        if distill_model:
            accs = evaluate(distill_model, dataset,
                            verbose=not distill_model.args.non_verbose)
            print(accs)

            acc_list, til_acc_list = accs[0], accs[1]
            prec1 = sum(acc_list) / (t + 1)
            til_prec1 = sum(til_acc_list) / (t + 1)
            acc_matrix[t] = acc_list
            # wandb.log({'acc_matrix': acc_matrix})
            # wandb.log({'acc_list': acc_list})
            # wandb.log({'til_acc_list': til_acc_list})
            forgetting_base = np.mean((np.max(acc_matrix, axis=0) - acc_list)[:t]) if t > 0 else 0.0
            forgetting = np.mean((np.max(acc_matrix, axis=0) - acc_list)[:t]) if t > 0 else 0.0
            learning_acc = np.mean(np.diag(acc_matrix)[:t + 1])

            log_line = 'Training on examples\n'
            log_line += f"Task: {t + 1}, Epoch:{dataset_setting.n_epochs - 1}, Average Acc:[{prec1:.3f}], Task Inc Acc:[{til_prec1:.3f}], Learning Acc:[{learning_acc:.3f}], Forgetting:[{forgetting:.3f}]\n"
            log_line += "\t"
            for i in range(t + 1):
                log_line += f"Acc@T{i}: {acc_list[i]:.3f}\t"
            log_line += "\n"
            log_line += "\t"
            for i in range(t + 1):
                log_line += f"Til-Acc@T{i}: {til_acc_list[i]:.3f}\t"
            log_line += "\n"
            print(log_line)
            with open(log_filename, 'a') as f:
                f.write(log_line)
                f.write("\n")

            # update and log the max_log_line
            if t == dataset.N_TASKS - 1:
                if prec1 > max_acc_at_last_task:
                    max_acc_at_last_task = prec1
                    max_log_line = log_line

                max_log_line = "Epoch with max average acc:\n" + max_log_line
                print(max_log_line)
                with open(log_filename, 'a') as f:
                    f.write(max_log_line)
                    f.write("\n")

            results.append(accs[0])
            results_mask_classes.append(accs[1])

            mean_acc = np.array([prec1, til_prec1], dtype=np.float64)
            update_accs(mean_acc, dataset.SETTING, args.job_number)
            print_mean_accuracy(mean_acc, t + 1, dataset.SETTING)
            model_stash['mean_accs'].append(mean_acc)

# --gpu_id 0 --model esmer_ablation --corrupt_perc 0.5 --tkd 1 --agreement 1 --plasticity 1 --stability 1 --charlie 0.1 --teacher_backbone resnet18 --lr 0.03 --buffer_size 200 --dataset noisy-seq-cifar100 --reg_weight 0.15 --ema_model_alpha 0.999 --ema_model_update_freq 0.07 --loss_margin 1.0 --loss_alpha 0.99 --task_warmup 1 --std_margin 1 --seed 0 --backbone resnet18_lg --epochs 50 --n_tasks 10 --name buf_tkd