import math
import sys
from typing import Iterable, Optional
from timm.utils.model import unwrap_model
import torch

from timm.scheduler import create_scheduler
from timm.optim import create_optimizer

from timm.data import Mixup
from timm.utils import accuracy, ModelEma
from lib import utils
import random
import time

import math

def manual_lr_schedule(epoch, args):
    # 고정 시작 learning rate
    start_lr = 1e-4
    min_lr = args.min_lr         # 예: 1e-5
    total_epochs = args.epochs   # 예: 100
    half = total_epochs // 2     # 예: 50

    # 두 구간 중 어떤 절반인지 판단
    if epoch < half:
        # 첫 번째 절반: 0 ~ 49
        epoch_in_half = epoch
    else:
        # 두 번째 절반: 50 ~ 99 → 상대 epoch: 0 ~ 49
        epoch_in_half = epoch - half

    # 해당 절반 내에서 cosine decay 적용
    decay_progress = epoch_in_half / (half - 1)
    cosine_decay = 0.5 * (1 + math.cos(math.pi * decay_progress))
    current_lr = min_lr + (start_lr - min_lr) * cosine_decay

    return current_lr


# def manual_lr_schedule(epoch, args):
#     warmup_epochs = 2
#     warmup_start_lr = 1e-5            # 워밍업 시작 learning rate
#     start_lr = 3e-4                    # 워밍업 이후 cosine decay 시작 learning rate
#     min_lr = args.min_lr              # 예: 1e-5
#     total_epochs = args.epochs        # 예: 100
#     half = total_epochs // 2          # 예: 50

#     # 두 구간 중 어떤 절반인지 판단
#     if epoch < half:
#         # 첫 번째 절반: 0 ~ 49
#         epoch_in_half = epoch
#     else:
#         # 두 번째 절반: 50 ~ 99 → 상대 epoch: 0 ~ 49
#         epoch_in_half = epoch - half

#     if epoch_in_half < warmup_epochs:
#         # 선형 워밍업
#         progress = epoch_in_half / warmup_epochs
#         current_lr = warmup_start_lr + (start_lr - warmup_start_lr) * progress
#     else:
#         # cosine decay
#         decay_progress = (epoch_in_half - warmup_epochs) / (half - warmup_epochs)
#         cosine_decay = 0.5 * (1 + math.cos(math.pi * decay_progress))
#         current_lr = min_lr + (start_lr - min_lr) * cosine_decay

#     return current_lr



def sample_configs(choices):

    config = {}
    dimensions = ['mlp_ratio', 'num_heads']
    depth = random.choice(choices['depth'])
    for dimension in dimensions:
        config[dimension] = [random.choice(choices[dimension]) for _ in range(depth)]

    config['embed_dim'] = [random.choice(choices['embed_dim'])]*depth

    config['layer_num'] = depth
    return config


def sample_configs_curriculum(choices, epoch=None, curriculum_epoch=None):
    config = {}
    dimensions = ['mlp_ratio', 'num_heads']
    depth = random.choice(choices['depth'])
    # depth = 12

    if epoch is None:
        config['embed_dim'] = [192] * depth
        config['mlp_ratio'] = [3.5] * depth
        config['num_heads'] = [3] * depth

    elif epoch < curriculum_epoch[1]:
        config['embed_dim'] = [192] * depth
        config['mlp_ratio'] = [3.5] * depth
        config['num_heads'] = [3] * depth
        # config['embed_dim'] = [192] * depth
        # config['mlp_ratio'] = [random.choices([3.5, 4.0], weights=[1, 1])[0] for _ in range(depth)]
        # config['num_heads'] = [random.choices([3, 4], weights=[1, 1])[0] for _ in range(depth)]

    elif epoch >= curriculum_epoch[1] and epoch < curriculum_epoch[2]:
        config['embed_dim'] = [216] * depth
        config['mlp_ratio'] = [random.choices([3.5, 4.0], weights=[1, 1])[0] for _ in range(depth)]
        config['num_heads'] = [random.choices([3, 4], weights=[1, 1])[0] for _ in range(depth)]

    else:
        config['embed_dim'] = [240] * depth
        config['mlp_ratio'] = [random.choices([3.5, 4.0], weights=[1, 1])[0] for _ in range(depth)]
        config['num_heads'] = [random.choices([3, 4], weights=[1, 1])[0] for _ in range(depth)]

    config['layer_num'] = depth
    return config

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    amp: bool = True, teacher_model: torch.nn.Module = None,
                    teach_loss: torch.nn.Module = None, choices=None, mode='super', retrain_config=None, args=None):
    model.train()
    criterion.train()

    # set random seed
    random.seed(epoch)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    if mode == 'retrain':
        config = retrain_config
        model_module = unwrap_model(model)
        print(config)
        model_module.set_sample_config(config=config)
        print(model_module.get_sampled_params_numel(config))

    temp_optimizer = create_optimizer(args, [p for p in model.parameters() if p.requires_grad])
    lr_scheduler, _ = create_scheduler(args, temp_optimizer)   

    curriculum_epoch = [-1, 0, 10]
    case_num = None

    if epoch in curriculum_epoch:
        case_num = curriculum_epoch.index(epoch) + 1
        # # case_num = None
        # 매 iteration마다 학습 가능한 파라미터만 포함하도록 새 optimizer를 생성
        optimizer = create_optimizer(
            args, 
            [p for p in model.parameters() if p.requires_grad]
        )
        lr_scheduler.optimizer = optimizer

    print("case_num : ", case_num)


    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # sample random config
        if mode == 'super':
            config = sample_configs_curriculum(choices=choices, epoch=epoch, curriculum_epoch=curriculum_epoch)    
            model_module = unwrap_model(model)
            model_module.set_sample_config(config=config, case_num=case_num)
            # config = sample_configs(choices=choices)
            # model_module = unwrap_model(model)
            # model_module.set_sample_config(config=config)

            current_lr = manual_lr_schedule(epoch, args)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

        elif mode == 'retrain':
            config = retrain_config
            model_module = unwrap_model(model)
            model_module.set_sample_config(config=config)
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        if amp:
            with torch.cuda.amp.autocast():
                if teacher_model:
                    with torch.no_grad():
                        teach_output = teacher_model(samples)
                    _, teacher_label = teach_output.topk(1, 1, True, True)
                    outputs = model(samples)
                    loss = 1/2 * criterion(outputs, targets) + 1/2 * teach_loss(outputs, teacher_label.squeeze())
                else:
                    outputs = model(samples)
                    loss = criterion(outputs, targets)
        else:
            outputs = model(samples)
            if teacher_model:
                with torch.no_grad():
                    teach_output = teacher_model(samples)
                _, teacher_label = teach_output.topk(1, 1, True, True)
                loss = 1 / 2 * criterion(outputs, targets) + 1 / 2 * teach_loss(outputs, teacher_label.squeeze())
            else:
                loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        if amp:
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)
        else:
            loss.backward()
            optimizer.step()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, optimizer

@torch.no_grad()
def evaluate(data_loader, model, device, amp=True, choices=None, mode='super', retrain_config=None, epoch=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    curriculum_epoch = [-1, 0, 10]

    # switch to evaluation mode
    model.eval()
    if mode == 'super':
        # config = sample_configs(choices=choices)
        config = sample_configs_curriculum(choices=choices, epoch=epoch, curriculum_epoch=curriculum_epoch) 
        model_module = unwrap_model(model)
        model_module.set_sample_config(config=config)
    else:
        config = retrain_config
        model_module = unwrap_model(model)
        model_module.set_sample_config(config=config)


    print("sampled model config: {}".format(config))
    parameters = model_module.get_sampled_params_numel(config)
    print("sampled model parameters: {}".format(parameters))

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        # compute output
        if amp:
            with torch.cuda.amp.autocast():
                output = model(images)
                loss = criterion(output, target)
        else:
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}