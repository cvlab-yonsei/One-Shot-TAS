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

@torch.no_grad()
def set_arc(model, config):
    metric_logger = utils.MetricLogger(delimiter="  ")

    # switch to evaluation mode
    model.eval()
    model_module = unwrap_model(model)
    model_module.set_sample_config(config=config)

    return

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
        else:
            prev_config[key] = config[key]  # 가장 작은 경우에는 freeze 없음
    
    for key in ['mlp_ratio', 'num_heads']:
        prev_config[key] = []
        for i in range(config['layer_num']):
            current_val = config[key][i]
            choice_list = choices[key]
            idx = choice_list.index(current_val)
            if idx > 0:
                prev_config[key].append(choice_list[idx - 1])
            else:
                prev_config[key].append(current_val)  # 가장 작은 경우에는 freeze 없음
    
    return prev_config

def get_locked_masks(model, current_config, prev_config, choices):
    """
    Gradient를 0으로 설정할 마스크 생성
    """
    locked_masks = {}
    for name, param in model.named_parameters():
        parts = name.split('.')
        
        if 'embed' in name:
            if current_config['embed_dim'][0] == min(choices['embed_dim']):
                continue  # 가장 작은 embed_dim이면 freeze 없음
            prev_embed_dim = prev_config['embed_dim'][0]
            if param.shape[0] > prev_embed_dim:
                mask = torch.zeros_like(param, dtype=torch.bool)
                mask[:prev_embed_dim] = True
                locked_masks[name] = mask
        
        elif 'attn' in name and 'qkv' in name:
            if 'blocks' in parts:
                layer_idx = int(parts[parts.index('blocks') + 1])
                if layer_idx >= len(current_config['num_heads']):
                    continue  # layer_idx 초과 시 continue
                if current_config['num_heads'][layer_idx] == min(choices['num_heads']):
                    continue  # 가장 작은 num_heads이면 freeze 없음
                if layer_idx < len(prev_config['num_heads']):
                    prev_heads = prev_config['num_heads'][layer_idx]
                    head_dim = param.shape[0] // prev_config['num_heads'][layer_idx]
                    freeze_heads = prev_heads * head_dim
                    mask = torch.zeros_like(param, dtype=torch.bool)
                    mask[:freeze_heads] = True
                    locked_masks[name] = mask
        
        elif 'fc1' in name or 'fc2' in name:
            if 'blocks' in parts:
                layer_idx = int(parts[parts.index('blocks') + 1])
                if layer_idx >= len(current_config['mlp_ratio']):
                    continue  # layer_idx 초과 시 continue
                if current_config['mlp_ratio'][layer_idx] == min(choices['mlp_ratio']):
                    continue  # 가장 작은 mlp_ratio이면 freeze 없음
                if layer_idx < len(prev_config['mlp_ratio']):
                    prev_mlp_ratio = prev_config['mlp_ratio'][layer_idx]
                    cur_mlp_ratio = current_config['mlp_ratio'][layer_idx]
                    
                    if len(param.shape) == 2:
                        prev_dim = int(param.shape[1] * (prev_mlp_ratio / cur_mlp_ratio))
                        mask = torch.zeros_like(param, dtype=torch.bool)
                        mask[:, :prev_dim] = True
                    elif len(param.shape) == 1:
                        prev_dim = int(param.shape[0] * (prev_mlp_ratio / cur_mlp_ratio))
                        mask = torch.zeros_like(param, dtype=torch.bool)
                        mask[:prev_dim] = True
                    else:
                        continue  # 예외 처리: shape이 예상과 다를 경우 skip
                    
                    locked_masks[name] = mask
    return locked_masks

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
                    layer_idx = int(parts[parts.index('blocks') + 1])
                    if layer_idx >= len(prev_config['num_heads']):
                        continue  # 처리하지 않고 다음 루프로 넘어감
                    prev_heads = prev_config['num_heads'][layer_idx]
                    cur_heads = current_config['num_heads'][layer_idx]
                    head_dim = param.shape[0] // cur_heads
                    freeze_heads = prev_heads * head_dim
                    param[:freeze_heads].requires_grad = False
            
            elif 'fc1' in name:
                if 'blocks' in parts:
                    layer_idx = int(parts[parts.index('blocks') + 1])
                    if layer_idx >= len(prev_config['mlp_ratio']):
                        continue  # 처리하지 않고 다음 루프로 넘어감
                    prev_mlp_ratio = prev_config['mlp_ratio'][layer_idx]
                    cur_mlp_ratio = current_config['mlp_ratio'][layer_idx]
                    prev_dim = int(param.shape[0] * (prev_mlp_ratio / cur_mlp_ratio))
                    param[:prev_dim].requires_grad = False
            
            elif 'fc2' in name:
                if 'blocks' in parts:
                    layer_idx = int(parts[parts.index('blocks') + 1])
                    if layer_idx >= len(prev_config['mlp_ratio']):
                        continue  # 처리하지 않고 다음 루프로 넘어감
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

    locked_masks = {}

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

        locked_masks = {}

        # sample random config
        if mode == 'super':
            config = sample_configs(choices=choices)
            model_module = unwrap_model(model)
            model_module.set_sample_config(config=config)
            prev_config = get_previous_config(config=config, choices=choices)
            locked_masks = get_locked_masks(model, config, prev_config, choices=choices)

            # for name, param in model_module.named_parameters():
            #     print(name, param.shape)

            # print("config : ", config)
            # for layer in model.modules():
            #     layer_name = layer._get_name()
            #     if hasattr(layer, 'samples') and 'weight' in layer.samples:
            #         param_shape = layer.samples['weight'].shape
            #         print(f"Layer: {layer_name}, Weight Shape: {param_shape}")
            #     else:
            #         print(f"Layer: {layer_name}, No weight parameter")
            # print("============================================")

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

        # AMP 사용 여부 확인
        if amp:
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order

            print(help(loss_scaler))


            # ✅ loss_scaler가 optimizer.step()까지 수행하지 않도록 need_update=False 설정
            loss_scaler(loss, optimizer, clip_grad=max_norm,
                        parameters=model.parameters(), create_graph=is_second_order, need_update=False)

            # 🔥 backward()는 수행된 상태 → 여기서 gradient 0으로 설정
            for name, param in model.named_parameters():
                if param.grad is not None and name in locked_masks:
                    param.grad[locked_masks[name]] = 0

            # ✅ optimizer step을 loss_scaler 내부에서 수행하지 않고, 여기서 직접 호출
            loss_scaler._scaler.step(optimizer)
            loss_scaler._scaler.update()

        else:
            loss.backward()

            # 🔥 AMP 미사용 시, backward() 이후 gradient 0으로 설정
            for name, param in model.named_parameters():
                if param.grad is not None and name in locked_masks:
                    param.grad[locked_masks[name]] = 0

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