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

# def manual_lr_schedule(epoch, args):
#     # 워밍업 설정
#     warmup_epochs = args.warmup_epochs       # ex. 20
#     warmup_start_lr = args.warmup_lr         # ex. 1e-6
#     base_lr = args.lr                        # ex. 5e-4
#     min_lr = args.min_lr                     # ex. 1e-5
#     total_epochs = args.epochs               # ex. 500

#     # if epoch < warmup_epochs:
#     #     # 선형 워밍업
#     #     current_lr = warmup_start_lr + (base_lr - warmup_start_lr) * (epoch / warmup_epochs)
#     # else:
#     #     # 선형 디케이
#     #     decay_progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
#     #     current_lr = base_lr - (base_lr - min_lr) * decay_progress

#     # cosine decay + warmup
#     if epoch < warmup_epochs:
#         current_lr = warmup_start_lr + (base_lr - warmup_start_lr) * (epoch / warmup_epochs)
#     else:
#         decay_progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
#         cosine_decay = 0.5 * (1 + math.cos(math.pi * decay_progress))
#         current_lr = min_lr + (base_lr - min_lr) * cosine_decay
    
#     return current_lr


import math

import math

# def manual_lr_schedule(epoch, args):
#     # config index에 따라 각 구간 할당 (399~499)
#     config_epochs = [10, 10] + [9] * 9  # 총 11개: [10, 10, 9, ..., 9]
#     config_start = 399

#     # 어떤 config에 해당하는지 찾기
#     offset = epoch - config_start
#     cum_epochs = 0
#     config_idx = -1
#     for i, e in enumerate(config_epochs):
#         if offset < cum_epochs + e:
#             config_idx = i
#             break
#         cum_epochs += e

#     if config_idx == -1:
#         return args.min_lr  # 예외 처리: 범위 벗어난 경우

#     # 해당 config 내에서 몇 번째 epoch인지
#     local_epoch = offset - cum_epochs

#     # warm-up 설정 (3 epochs)
#     warmup_lrs = [1e-4, 3e-4, 7e-4]
#     if local_epoch < 3:
#         current_lr = warmup_lrs[local_epoch]
#     else:
#         # cosine decay: 시작 lr 0.001 → 끝 lr 0.0005
#         decay_epoch = local_epoch - 3
#         decay_total = config_epochs[config_idx] - 3
#         base_lr = 1e-3
#         min_lr = 5e-4

#         decay_progress = decay_epoch / max(decay_total, 1)  # 0~1 사이
#         cosine_decay = 0.5 * (1 + math.cos(math.pi * decay_progress))
#         current_lr = min_lr + (base_lr - min_lr) * cosine_decay

#     return current_lr

def manual_lr_schedule(epoch, args):
    base_lr = 1e-4
    min_lr = 1e-5
    start_epoch = 399
    end_epoch = 499
    total_decay_epochs = end_epoch - start_epoch

    # 시작 전에는 base_lr 유지
    if epoch < start_epoch:
        return base_lr
    # 종료 후에도 min_lr 고정
    elif epoch > end_epoch:
        return min_lr

    decay_progress = (epoch - start_epoch) / total_decay_epochs
    cosine_decay = 0.5 * (1 + math.cos(math.pi * decay_progress))
    current_lr = min_lr + (base_lr - min_lr) * cosine_decay

    return current_lr





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


def sample_configs_curriculum(choices, epoch=None):
    import random

    config = {}
    # depth = random.choice(choices['depth'])
    depth = 14

    config_list = [
        (192, 3, 4.0),
        (192, 4, 3.5),
        (192, 4, 4.0),
        (216, 3, 3.5),
        (216, 3, 4.0),
        (216, 4, 3.5),
        (216, 4, 4.0),
        (240, 3, 3.5),
        (240, 3, 4.0),
        (240, 4, 3.5),
        (240, 4, 4.0),
    ]

    # ---------------------------
    # 1. Pre-curriculum 단계
    # ---------------------------
    if epoch is None or epoch < 399:
        config['embed_dim'] = [192] * depth
        config['num_heads'] = [3] * depth
        config['mlp_ratio'] = [3.5] * depth

    # ---------------------------
    # 2. 299~500: curriculum 적용
    # ---------------------------
    elif 399 <= epoch <= 500:
        curriculum_epochs = 101  # 399 ~ 499 inclusive
        num_configs = len(config_list)
        base_epochs = curriculum_epochs // num_configs  # 9
        extras = curriculum_epochs % num_configs        # 2

        # config별 시작-끝 구간 계산
        schedule = []
        start = 0
        for i in range(num_configs):
            duration = base_epochs + 1 if i < extras else base_epochs
            end = start + duration
            schedule.append((start, end))  # (inclusive, exclusive)
            start = end

        # 현재 config index 찾기
        offset = epoch - 399
        for config_idx, (s, e) in enumerate(schedule):
            if s <= offset < e:
                embed_dim, preferred_head, preferred_mlp = config_list[config_idx]
                break

        # 확률 기반 샘플링
        head_choices = [preferred_head, 4 if preferred_head == 3 else 3]
        head_weights = [1, 0] # 2, 1
        mlp_choices = [preferred_mlp, 3.5 if preferred_mlp == 4.0 else 4.0]
        mlp_weights = [1, 0] # 2, 1

        config['embed_dim'] = [embed_dim] * depth
        config['num_heads'] = [
            random.choices(head_choices, weights=head_weights)[0] for _ in range(depth)
        ]
        config['mlp_ratio'] = [
            random.choices(mlp_choices, weights=mlp_weights)[0] for _ in range(depth)
        ]

    # ---------------------------
    # 3. 이후 epoch: 마지막 config 반복
    # ---------------------------
    else:
        embed_dim, preferred_head, preferred_mlp = config_list[-1]
        head_choices = [preferred_head, 4 if preferred_head == 3 else 3]
        head_weights = [1, 0] # 2, 1
        mlp_choices = [preferred_mlp, 3.5 if preferred_mlp == 4.0 else 4.0]
        mlp_weights = [1, 0] # 2, 1

        config['embed_dim'] = [embed_dim] * depth
        config['num_heads'] = [
            random.choices(head_choices, weights=head_weights)[0] for _ in range(depth)
        ]
        config['mlp_ratio'] = [
            random.choices(mlp_choices, weights=mlp_weights)[0] for _ in range(depth)
        ]

    config['layer_num'] = depth
    return config

# def sample_configs_curriculum(choices, epoch=None):
#     config = {}
#     dimensions = ['mlp_ratio', 'num_heads']
#     # depth = random.choice(choices['depth'])
#     depth = 14
#     if epoch is None:
#             config['embed_dim'] = [192] * depth
#             config['mlp_ratio'] = [3.5] * depth
#             config['num_heads'] = [3] * depth
#     else:
#         if epoch <= 1000:
#             config['embed_dim'] = [192] * depth
#             config['mlp_ratio'] = [3.5] * depth
#             config['num_heads'] = [3] * depth

#         elif 301 <= epoch <= 400:
#             config['embed_dim'] = [random.choices([192, 216, 240], weights=[1, 4, 1])[0]] * depth
#             config['mlp_ratio'] = [random.choices([3.5, 4.0], weights=[1, 3])[0] for _ in range(depth)]
#             config['num_heads'] = [random.choices([3, 4], weights=[1, 3])[0] for _ in range(depth)]

#         elif 401 <= epoch <= 500:
#             config['embed_dim'] = [random.choices([192, 216, 240], weights=[1, 1, 4])[0]] * depth
#             config['mlp_ratio'] = [random.choices([3.5, 4.0], weights=[1, 5])[0] for _ in range(depth)]
#             config['num_heads'] = [random.choices([3, 4], weights=[1, 5])[0] for _ in range(depth)]


#         # elif 301 <= epoch <= 400:
#         #     config['embed_dim'] = [random.choices([192, 216, 240], weights=[1, 4, 1])[0]] * depth
#         #     config['mlp_ratio'] = [random.choices([3.5, 4.0], weights=[1, 3])[0] for _ in range(depth)]
#         #     config['num_heads'] = [random.choices([3, 4], weights=[1, 3])[0] for _ in range(depth)]

#         # elif 401 <= epoch <= 500:
#         #     config['embed_dim'] = [random.choices([192, 216, 240], weights=[1, 1, 4])[0]] * depth
#         #     config['mlp_ratio'] = [random.choices([3.5, 4.0], weights=[1, 5])[0] for _ in range(depth)]
#         #     config['num_heads'] = [random.choices([3, 4], weights=[1, 5])[0] for _ in range(depth)]

#     config['layer_num'] = depth
#     return config

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    amp: bool = True, teacher_model: torch.nn.Module = None,
                    teach_loss: torch.nn.Module = None, choices=None, mode='super', retrain_config=None, args = None, prev_step_config = None):
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
    config = sample_configs_curriculum(choices=choices, epoch=epoch)
    prev_config = get_previous_config(config=config, choices=choices) # None 처리 잘되는거 확인

    init_done = False


    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        locked_masks = {}

        # sample random config
        if mode == 'super':
            # config = sample_configs(choices=choices)
            # config = sample_configs_curriculum(choices=choices, epoch=epoch)
            # prev_config = get_previous_config(config=config, choices=choices) # None 처리 잘되는거 확인
            model_module = unwrap_model(model)
            # pretrained = prev_step_config is None and init_done == False
            pretrained = False
            # print("supernet_engine : ", pretrained)
            model_module.set_sample_config(config=config, config_prev=prev_config)
            init_done = True
            # model_module.set_sample_config(config=config)

            if config != prev_step_config:
                print("config : ", config)
                print("prev_step_config : ", prev_step_config)
                prev_step_config = config
                print("config change!")
                # 매 iteration마다 학습 가능한 파라미터만 포함하도록 새 optimizer를 생성
                optimizer = create_optimizer(
                    args, 
                    [p for p in model.parameters() if p.requires_grad]
                )
                lr_scheduler.optimizer = optimizer
                
            # if epoch < 20:
            #     # 워밍업: 0~20 epoch에서 1e-6에서 1e-3로 증가
            #     current_lr = 1e-6 + (1e-3 - 1e-6) * (epoch / 20)
            # else:
            #     # 디케이: 20~500 epoch에서 1e-3에서 1e-5로 감소
            #     current_lr = 1e-3 - (1e-3 - 1e-5) * ((epoch - 20) / (500 - 20))

            # # 새 optimizer의 모든 파라미터 그룹에 현재 lr을 설정합니다.
            # for group in optimizer.param_groups:
            #     group['lr'] = current_lr

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
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, prev_step_config

@torch.no_grad()
def evaluate(data_loader, model, device, amp=True, choices=None, mode='super', retrain_config=None, epoch=None, prev_step_config=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    if mode == 'super':
        # config = sample_configs(choices=choices)
        # config = sample_configs_curriculum(choices=choices, epoch=epoch)
        config = prev_step_config
        if prev_step_config is None:
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