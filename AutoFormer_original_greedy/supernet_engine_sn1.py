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
import torch.nn as nn
from typing import List, Dict

def sample_config_from_topk(model: torch.nn.Module, choices: Dict, m: int, k: int, device: torch.device, 
                            candidate_pool: List = None, pool_sampling_prob: float = 0.0) -> List:
    model.eval()
    model_module = unwrap_model(model)

    # DSS 점수를 계산하기 위한 기본 설정
    sampled_config = {
        'layer_num': 14,
        'mlp_ratio': [4.0] * 14,
        'num_heads': [4] * 14,
        'embed_dim': [256] * 14
    }
    set_arc(model, sampled_config)

    def linearize(net):
        signs = {}
        for name, param in net.state_dict().items():
            signs[name] = torch.sign(param)
            param.abs_()
        return signs

    def nonlinearize(net, signs):
        for name, param in net.state_dict().items():
            if 'weight_mask' not in name:
                param.mul_(signs[name])

    # DSS 점수 계산
    signs = linearize(model_module)
    supernet_indicators = {}
    total = 0
    for layer in model.modules():
        if isinstance(layer, torch.nn.Linear) and layer.samples:
            layer_metric = spectral_norm(layer, device)
            supernet_indicators[layer._get_name() + '_' + str(total)] = layer_metric
        total += 1
    nonlinearize(model_module, signs)

    sampled_paths = []
    groups = {i: [] for i in range(5)}

    # Sample m paths
    with torch.no_grad():
        while len(sampled_paths) < m:
            if candidate_pool and random.random() <= pool_sampling_prob:
                config = random.choice(candidate_pool)
            else:
                config = sample_configs(choices)
            param_count = model_module.get_sampled_params_numel(config)
            group = get_group(param_count)
            # if config not in sampled_paths:
                # sampled_paths.append(config)
                # groups[group].append(config)
            sampled_paths.append(config)
            groups[group].append(config)

        losses = []
        for config in sampled_paths:
            set_arc(model, config)
            signs = linearize(model_module)
            metric_array = []
            total = 0
            attn_layer_total = 0
            for layer in model.modules():
                norm_val = supernet_indicators.get(layer._get_name() + '_' + str(total), 0)
                if isinstance(layer, torch.nn.Linear) and layer.samples and norm_val != 0 and attn_layer_total < config['layer_num']:
                    metric_array.append(spectral_norm(layer, device) / norm_val)
                    attn_layer_total += 1
                total += 1
            sum_dss_score = sum_arr(metric_array)
            param_count = model_module.get_sampled_params_numel(config)
            group = get_group(param_count)
            losses.append((sum_dss_score, config, param_count, group))
            nonlinearize(model_module, signs)
            if len(losses) % 100 == 0:
                print(f"[{len(losses)} / {m}] Loss: {sum_dss_score}, Config: {config}, Params: {param_count}, Group: {group}")

    # 그룹별 상위 50% 선택
    top_k_paths = []
    remaining_items = []
    for group_id in range(5):
        group_losses = [loss for loss in losses if loss[3] == group_id]
        group_losses.sort(key=lambda x: x[0], reverse=True)
        num_to_select = max(1, len(group_losses) // 2)
        top_k_paths.extend(group_losses[:num_to_select])
        remaining_items.extend(group_losses[num_to_select:])
    if len(top_k_paths) < k:
        remaining_items.sort(key=lambda x: x[0], reverse=True)
        top_k_paths.extend(remaining_items[:k - len(top_k_paths)])
    random.shuffle(top_k_paths)

    # top_k_paths에서 config만 반환
    return [config for _, config, _, _ in top_k_paths]


@torch.no_grad()
def set_arc(model, config):
    metric_logger = utils.MetricLogger(delimiter="  ")

    # switch to evaluation mode
    model.eval()
    model_module = unwrap_model(model)
    model_module.set_sample_config(config=config)

    return

def sum_arr(arr):
    sum = 0.0  # float으로 초기화
    # print("arr : ", arr)
    for i in range(len(arr)):
        sum += torch.sum(arr[i]).item()  # 각 요소의 item을 더함
    return sum  # 최종 결과는 float으로 반환

def get_group(param_count):
    # Parameter 수를 기준으로 그룹을 반환
    if param_count < 6e6:
        return 0  # ~6M
    elif 6e6 <= param_count < 7e6:
        return 1  # 6M~7M
    elif 7e6 <= param_count < 8e6:
        return 2  # 7M~8M
    elif 8e6 <= param_count < 9e6:
        return 3  # 8M~9M
    else:
        return 4  # 9M~

def spectral_norm(layer, device):
    if isinstance(layer, nn.Linear) and 'qkv' in layer._get_name() and layer.samples:
        # qkv의 가중치를 3등분하여 W_q와 W_k 추출
        W_qkv = layer.samples['weight']
        # print("W_qkv shape : ", W_qkv.shape)
        if W_qkv is not None:
            d = W_qkv.shape[0] // 3
            W_q = W_qkv[:d, :]
            W_k = W_qkv[d:2 * d, :]

            # W_q * W_k^T 연산
            W_qkT = torch.matmul(W_q, W_k.T)

            # W_q * W_k^T의 spectral norm 계산
            spectral_norm = torch.svd(W_qkT)[1].max().item()
            
            return torch.tensor(spectral_norm).to(device)
        else:
            return torch.tensor(0).to(device)
    else:
        return torch.tensor(0).to(device)
            

def dss(layer):
    # print(f"Layer Name: {layer._get_name()}")
    
    for name, param in layer.named_parameters():
        # print(f"Parameter: {name}, requires_grad: {param.requires_grad}")
        
        if name == "weight" and param.requires_grad is True:
            if layer._get_name() == 'PatchembedSuper':
                if param.grad is not None:
                    return torch.abs(param.grad * param)
                else:
                    return torch.zeros_like(param)
            
            if isinstance(layer, nn.Linear) and 'qkv' in layer._get_name() and layer.samples:
                if param.grad is not None:
                    return torch.abs(
                        torch.norm(param.grad, 'nuc') * torch.norm(param, 'nuc'))
                else:
                    return torch.zeros_like(param)
            
            if isinstance(layer, nn.Linear) and 'qkv' not in layer._get_name() and layer.out_features != layer.in_features and layer.out_features != 1000 and layer.samples:
                if param.grad is not None:
                    return torch.abs(param.grad * param)
                else:
                    return torch.zeros_like(param)
            
            elif isinstance(layer, torch.nn.Linear) and layer.out_features == 1000:
                if param.grad is not None:
                    return torch.abs(param.grad * param)
                else:
                    return torch.zeros_like(param)
            
            else:
                return torch.tensor(0).to(torch.device('cpu'))

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
                    teach_loss: torch.nn.Module = None, choices=None, mode='super', retrain_config=None,
                    candidate_pool=None, validation_data_loader=None, pool_sampling_prob=0, m=10, k=5, interval=1):
    model.train()
    criterion.train()

    # set random seed
    random.seed(epoch)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    
    config_list = []
    
    if mode == 'retrain':
        config = retrain_config
        model_module = unwrap_model(model)
        print(config)
        model_module.set_sample_config(config=config)
        print(model_module.get_sampled_params_numel(config))
    else:
        if interval == 1:
            # config_list = sample_config_from_topk(model, choices, m, k, device, candidate_pool, pool_sampling_prob) # 나중에 400ep으로 이거 0.8로 실험 한번 더
            config_list = sample_config_from_topk(model, choices, m, k, device, candidate_pool, 0) # 이렇게 실험해버린듯(첫 400ep)
            candidate_pool[:] = [config for config in config_list] # update candidate pool
        else:
            if not candidate_pool or epoch%interval == 0:
                # config_list = sample_config_from_topk(model, choices, m, k, device, candidate_pool, pool_sampling_prob)
                config_list = sample_config_from_topk(model, choices, m, k, device, candidate_pool, 0) # random pool
                candidate_pool[:] = [config for config in config_list] # update candidate pool
            else:
                config_list = candidate_pool
                
            print("config_list[:5] : ", config_list[:5])
            
        

    for iter, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # sample random config
        if mode == 'super':
            # config = sample_configs(choices=choices)
            config = config_list[iter%k]
            model_module = unwrap_model(model)
            model_module.set_sample_config(config=config)
            
            
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
