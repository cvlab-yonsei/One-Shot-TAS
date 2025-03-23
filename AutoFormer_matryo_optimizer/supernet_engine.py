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

# def get_previous_config(config, choices):
#     """
#     한 단계 작은 subnet을 찾는 함수
#     """
#     prev_config = config.copy()
    
#     for key in ['embed_dim']:
#         current_val = config[key][0]
#         choice_list = choices[key]
#         idx = choice_list.index(current_val)
#         if idx > 0:
#             prev_config[key] = [choice_list[idx - 1]] * config['layer_num']
#         else:
#             prev_config[key] = config[key]  # 가장 작은 경우에는 freeze 없음
    
#     for key in ['mlp_ratio', 'num_heads']:
#         prev_config[key] = []
#         for i in range(config['layer_num']):
#             current_val = config[key][i]
#             choice_list = choices[key]
#             idx = choice_list.index(current_val)
#             if idx > 0:
#                 prev_config[key].append(choice_list[idx - 1])
#             else:
#                 prev_config[key].append(current_val)  # 가장 작은 경우에는 freeze 없음. current랑 똑같으면 mask 안주게 다른 함수에서 처리할거.
    
#     return prev_config

def get_previous_config(config, choices):
    """
    한 단계 작은 subnet을 찾는 함수.
    만약 현재 config의 인자값이 이미 choices에서 가장 작은 값이면, 그 인자는 None으로 처리함.
    """
    prev_config = config.copy()
    
    # embed_dim은 모든 layer에서 동일하다고 가정
    for key in ['embed_dim']:
        current_val = config[key][0]
        choice_list = choices[key]
        idx = choice_list.index(current_val)
        if idx > 0:
            prev_config[key] = [choice_list[idx - 1]] * config['layer_num']
        else:
            # 이미 가장 작은 값이면, 모든 layer에 대해 None 할당
            prev_config[key] = [None] * config['layer_num']
    
    # mlp_ratio와 num_heads는 각 layer마다 다를 수 있음
    for key in ['mlp_ratio', 'num_heads']:
        prev_config[key] = []
        for i in range(config['layer_num']):
            current_val = config[key][i]
            choice_list = choices[key]
            idx = choice_list.index(current_val)
            if idx > 0:
                prev_config[key].append(choice_list[idx - 1])
            else:
                # 가장 작은 값이면 None 할당
                prev_config[key].append(None)
    
    return prev_config

def get_locked_masks(model, current_config, prev_config, choices):
    """
    Gradient를 0으로 설정할 마스크 생성 (weight와 bias만 처리)
    """
    locked_masks = {}
    for name, param in model.named_parameters():
        if 'weight' not in name and 'bias' not in name:
            continue  # weight, bias가 아닌 경우 제외
        
        parts = name.split('.')
        
        if 'patch_embed_super.proj.weight' in name:
            prev_embed_dim = prev_config['embed_dim'][0]
            cur_embed_dim = current_config['embed_dim'][0]
            if prev_embed_dim == cur_embed_dim:
                continue  # 값이 같으면 freeze 없음
            if param.shape[0] > prev_embed_dim:
                mask = torch.zeros_like(param, dtype=torch.bool)
                mask[:prev_embed_dim, :, :, :] = True  # 앞부분만 freeze
                locked_masks[name] = mask
        
        elif 'patch_embed_super.proj.bias' in name:
            prev_embed_dim = prev_config['embed_dim'][0]
            cur_embed_dim = current_config['embed_dim'][0]
            if prev_embed_dim == cur_embed_dim:
                continue  # 값이 같으면 freeze 없음
            if param.shape[0] > prev_embed_dim:
                mask = torch.zeros_like(param, dtype=torch.bool)
                mask[:prev_embed_dim] = True
                locked_masks[name] = mask
        
        elif 'attn.qkv.weight' in name:
            layer_idx = int(parts[parts.index('blocks') + 1])
            if layer_idx >= len(prev_config['num_heads']):
                continue
            prev_heads = prev_config['num_heads'][layer_idx]
            cur_heads = current_config['num_heads'][layer_idx]
            prev_embed_dim = prev_config['embed_dim'][layer_idx]
            cur_embed_dim = current_config['embed_dim'][layer_idx]
            if prev_heads == cur_heads and prev_embed_dim == cur_embed_dim:
                continue
            # head_dim = 192 // prev_heads  # 192 반영
            # freeze_dim = head_dim * prev_heads
            freeze_dim = 192 * prev_heads
            mask = torch.zeros_like(param, dtype=torch.bool)
            mask[:freeze_dim, :prev_embed_dim] = True
            locked_masks[name] = mask
        
        elif 'attn.qkv.bias' in name:
            layer_idx = int(parts[parts.index('blocks') + 1])
            if layer_idx >= len(prev_config['num_heads']):
                continue
            prev_heads = prev_config['num_heads'][layer_idx]
            cur_heads = current_config['num_heads'][layer_idx]
            if prev_heads == cur_heads:
                continue
            freeze_dim = 192 * prev_heads  # 192 반영
            mask = torch.zeros_like(param, dtype=torch.bool)
            mask[:freeze_dim] = True
            locked_masks[name] = mask
        
        elif 'attn.proj.weight' in name:
            layer_idx = int(parts[parts.index('blocks') + 1])
            if layer_idx >= len(prev_config['num_heads']) or layer_idx >= len(prev_config['embed_dim']):
                continue
            prev_embed_dim = prev_config['embed_dim'][layer_idx]
            cur_embed_dim = current_config['embed_dim'][layer_idx]
            prev_heads = prev_config['num_heads'][layer_idx]
            cur_heads = current_config['num_heads'][layer_idx]
            if prev_embed_dim == cur_embed_dim and prev_heads == cur_heads:
                continue
            head_dim = 64  # 64 반영
            freeze_dim = prev_heads * head_dim
            mask = torch.zeros_like(param, dtype=torch.bool)
            mask[:prev_embed_dim, :freeze_dim] = True
            locked_masks[name] = mask
        
        elif 'fc1.weight' in name:
            layer_idx = int(parts[parts.index('blocks') + 1])
            if layer_idx >= len(prev_config['mlp_ratio']):
                continue
            prev_mlp_ratio = prev_config['mlp_ratio'][layer_idx]
            cur_mlp_ratio = current_config['mlp_ratio'][layer_idx]
            prev_embed_dim = prev_config['embed_dim'][layer_idx]
            cur_embed_dim = current_config['embed_dim'][layer_idx]
            if prev_mlp_ratio == cur_mlp_ratio and prev_embed_dim == cur_embed_dim:
                continue
            prev_dim = int(prev_mlp_ratio * prev_embed_dim)
            mask = torch.zeros_like(param, dtype=torch.bool)
            mask[:prev_dim, :prev_embed_dim] = True
            locked_masks[name] = mask
        
        elif 'fc2.weight' in name:
            layer_idx = int(parts[parts.index('blocks') + 1])
            if layer_idx >= len(prev_config['mlp_ratio']):
                continue
            prev_mlp_ratio = prev_config['mlp_ratio'][layer_idx]
            cur_mlp_ratio = current_config['mlp_ratio'][layer_idx]
            prev_embed_dim = prev_config['embed_dim'][layer_idx]
            cur_embed_dim = current_config['embed_dim'][layer_idx]
            if prev_mlp_ratio == cur_mlp_ratio and prev_embed_dim == cur_embed_dim:
                continue
            prev_dim = int(prev_mlp_ratio * prev_embed_dim)
            mask = torch.zeros_like(param, dtype=torch.bool)
            mask[:prev_embed_dim, :prev_dim] = True
            locked_masks[name] = mask

        elif 'attn.proj.bias' in name or 'attn_layer_norm.weight' in name or 'attn_layer_norm.bias' in name or \
        'ffn_layer_norm.weight' in name or 'ffn_layer_norm.bias' in name or 'fc2.bias' in name:
            prev_embed_dim = prev_config['embed_dim'][0]
            cur_embed_dim = current_config['embed_dim'][0]
            if prev_embed_dim == cur_embed_dim:
                continue  # 값이 같으면 freeze 없음
            mask = torch.zeros_like(param, dtype=torch.bool)
            mask[:prev_embed_dim] = True
            locked_masks[name] = mask

        elif 'fc1.bias' in name:
            layer_idx = int(name.split('.')[2])  # 블록 인덱스 추출
            if layer_idx >= len(prev_config['mlp_ratio']):
                continue
            prev_mlp_ratio = prev_config['mlp_ratio'][layer_idx]
            cur_mlp_ratio = current_config['mlp_ratio'][layer_idx]
            prev_embed_dim = prev_config['embed_dim'][layer_idx]
            cur_embed_dim = current_config['embed_dim'][layer_idx]
            if prev_mlp_ratio == cur_mlp_ratio and prev_embed_dim == cur_embed_dim:
                continue
            prev_dim = int(prev_mlp_ratio * prev_embed_dim)
            mask = torch.zeros_like(param, dtype=torch.bool)
            mask[:prev_dim] = True
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

def sample_configs_curriculum(choices, epoch):
    config = {}
    dimensions = ['mlp_ratio', 'num_heads']
    # depth = random.choice(choices['depth'])
    depth = 14

    if epoch <= 1000:
        config['embed_dim'] = [192] * depth
        config['mlp_ratio'] = [3.5] * depth
        config['num_heads'] = [3] * depth

    # elif 301 <= epoch <= 400:
    #     config['embed_dim'] = [random.choices([192, 216, 240], weights=[1, 4, 1])[0]] * depth
    #     config['mlp_ratio'] = [random.choices([3.5, 4.0], weights=[1, 3])[0] for _ in range(depth)]
    #     config['num_heads'] = [random.choices([3, 4], weights=[1, 3])[0] for _ in range(depth)]

    # elif 401 <= epoch <= 500:
    #     config['embed_dim'] = [random.choices([192, 216, 240], weights=[1, 1, 4])[0]] * depth
    #     config['mlp_ratio'] = [random.choices([3.5, 4.0], weights=[1, 5])[0] for _ in range(depth)]
    #     config['num_heads'] = [random.choices([3, 4], weights=[1, 5])[0] for _ in range(depth)]

    config['layer_num'] = depth
    return config

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    amp: bool = True, teacher_model: torch.nn.Module = None,
                    teach_loss: torch.nn.Module = None, choices=None, mode='super', retrain_config=None, args = None):
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

    temp_optimizer = create_optimizer(args, [p for p in model.parameters() if p.requires_grad])
    lr_scheduler, _ = create_scheduler(args, temp_optimizer)   

    global_iter = epoch * len(data_loader) 

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        locked_masks = {}

        # sample random config
        if mode == 'super':
            # config = sample_configs(choices=choices)
            config = sample_configs_curriculum(choices=choices, epoch=epoch)
            prev_config = get_previous_config(config=config, choices=choices) # None 처리 잘되는거 확인
            model_module = unwrap_model(model)
            model_module.set_sample_config(config=config, config_prev=prev_config)
            # model_module.set_sample_config(config=config)

            # 매 iteration마다 학습 가능한 파라미터만 포함하도록 새 optimizer를 생성
            optimizer = create_optimizer(
                args, 
                [p for p in model.parameters() if p.requires_grad]
            )
            lr_scheduler.optimizer = optimizer
            
            # 이후 for 루프 내에서 사용
            # current_lr = lr_scheduler.get_last_lr()[0]

            # current_lr = 0.001 - (0.001-0.00001)*(epoch/500)
            if epoch < 20:
                # 워밍업: 0~20 epoch에서 1e-6에서 1e-3로 증가
                current_lr = 1e-6 + (1e-3 - 1e-6) * (epoch / 20)
            else:
                # 디케이: 20~500 epoch에서 1e-3에서 1e-5로 감소
                current_lr = 1e-3 - (1e-3 - 1e-5) * ((epoch - 20) / (500 - 20))

            # 새 optimizer의 모든 파라미터 그룹에 현재 lr을 설정합니다.
            for group in optimizer.param_groups:
                group['lr'] = current_lr

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

        # optimizer.zero_grad()

        # # AMP 사용 여부 확인
        # if amp:
        #     is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order

        #     print(help(loss_scaler))


        #     # ✅ loss_scaler가 optimizer.step()까지 수행하지 않도록 need_update=False 설정
        #     loss_scaler(loss, optimizer, clip_grad=max_norm,
        #                 parameters=model.parameters(), create_graph=is_second_order, need_update=False)

        #     # 🔥 backward()는 수행된 상태 → 여기서 gradient 0으로 설정
        #     for name, param in model.named_parameters():
        #         if param.grad is not None and name in locked_masks:
        #             param.grad[locked_masks[name]] = 0

        #     # ✅ optimizer step을 loss_scaler 내부에서 수행하지 않고, 여기서 직접 호출
        #     loss_scaler._scaler.step(optimizer)
        #     loss_scaler._scaler.update()

        # else:
        #     loss.backward()

        #     # 🔥 AMP 미사용 시, backward() 이후 gradient 0으로 설정
        #     for name, param in model.named_parameters():
        #         if param.grad is not None and name in locked_masks:
        #             param.grad[locked_masks[name]] = 0

        #     optimizer.step()

        # torch.cuda.synchronize()
        # if model_ema is not None:
        #     model_ema.update(model)

        # metric_logger.update(loss=loss_value)
        # metric_logger.update(lr=optimizer.param_groups[0]["lr"])

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