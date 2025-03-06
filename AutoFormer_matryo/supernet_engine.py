import math
import sys
from typing import Iterable, Optional
from timm.utils.model import unwrap_model
import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma
from lib import utils
import random
import time

def get_previous_config(config, choices):
    """
    한 단계 작은 subnet을 찾는 함수
    """
    prev_config = config.copy()
    
    for key in ['embed_dim']:
        current_val = config[key][0]
        choice_list = choices[key]
        idx = choice_list.index(current_val)
        if idx > 0:
            prev_config[key] = [choice_list[idx - 1]] * config['layer_num']
    
    for key in ['mlp_ratio', 'num_heads']:
        prev_config[key] = []
        for i in range(config['layer_num']):
            current_val = config[key][i]
            choice_list = choices[key]
            idx = choice_list.index(current_val)
            if idx > 0:
                prev_config[key].append(choice_list[idx - 1])
            else:
                prev_config[key].append(current_val)
    
    return prev_config

def freeze_weights(model, current_config, prev_config):
    """
    이전 subnet에서 학습된 부분을 freeze하는 함수
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            parts = name.split('.')
            
            if 'embed' in name:
                prev_embed_dim = prev_config['embed_dim'][0]
                cur_embed_dim = current_config['embed_dim'][0]
                if param.shape[0] > prev_embed_dim:
                    param[:prev_embed_dim].requires_grad = False
            
            elif 'attn' in name and 'qkv' in name:
                if 'blocks' in parts:
                    layer_idx = int(parts[parts.index('blocks') + 1])  # "blocks.X"에서 X 추출
                    prev_heads = prev_config['num_heads'][layer_idx]
                    cur_heads = current_config['num_heads'][layer_idx]
                    head_dim = param.shape[0] // cur_heads
                    freeze_heads = prev_heads * head_dim
                    param[:freeze_heads].requires_grad = False
            
            elif 'fc1' in name:
                if 'blocks' in parts:
                    layer_idx = int(parts[parts.index('blocks') + 1])  # "blocks.X"에서 X 추출
                    prev_mlp_ratio = prev_config['mlp_ratio'][layer_idx]
                    cur_mlp_ratio = current_config['mlp_ratio'][layer_idx]
                    prev_dim = int(param.shape[0] * (prev_mlp_ratio / cur_mlp_ratio))
                    param[:prev_dim].requires_grad = False
            
            elif 'fc2' in name:
                if 'blocks' in parts:
                    layer_idx = int(parts[parts.index('blocks') + 1])  # "blocks.X"에서 X 추출
                    prev_mlp_ratio = prev_config['mlp_ratio'][layer_idx]
                    cur_mlp_ratio = current_config['mlp_ratio'][layer_idx]
                    
                    # 텐서 차원 확인 후 처리 (1D인지 2D인지 확인)
                    if len(param.shape) == 2:
                        prev_dim = int(param.shape[1] * (prev_mlp_ratio / cur_mlp_ratio))
                        param[:, :prev_dim].requires_grad = False
                    elif len(param.shape) == 1:
                        prev_dim = int(param.shape[0] * (prev_mlp_ratio / cur_mlp_ratio))
                        param[:prev_dim].requires_grad = False

def sample_configs(choices):

    config = {}
    dimensions = ['mlp_ratio', 'num_heads']
    depth = random.choice(choices['depth'])
    for dimension in dimensions:
        config[dimension] = [random.choice(choices[dimension]) for _ in range(depth)]

    config['embed_dim'] = [random.choice(choices['embed_dim'])]*depth

    config['layer_num'] = depth
    return config

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    amp: bool = True, teacher_model: torch.nn.Module = None,
                    teach_loss: torch.nn.Module = None, choices=None, mode='super', retrain_config=None):
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

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # sample random config
        if mode == 'super':
            config = sample_configs(choices=choices)
            model_module = unwrap_model(model)
            model_module.set_sample_config(config=config)
            prev_config = get_previous_config(config=config, choices=choices)
            freeze_weights(model, config, prev_config)
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
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device, amp=True, choices=None, mode='super', retrain_config=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    if mode == 'super':
        config = sample_configs(choices=choices)
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