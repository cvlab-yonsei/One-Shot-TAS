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

def spectral_norm(layer, device):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for name, param in layer.named_parameters():
        # print(f"Parameter: {name}, requires_grad: {param.requires_grad}")
        if name == "weight" and param.requires_grad is True:
            if isinstance(layer, nn.Linear) and 'qkv' in layer._get_name() and layer.samples:
                print("param shape : ", param.shape)
                W_qkv = param
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
            
            for layer in model.modules():
                if mode == 'channel' and hasattr(layer, 'dont_ch_prune'):
                    continue
                if isinstance(layer, nn.Linear) and layer.samples:
                    print("hello")
                    # 텐서의 차원이 2D 이상일 때만 처리
                    if isinstance(layer, nn.Linear) and 'qkv' in layer._get_name() and layer.samples:
                        # qkv의 가중치를 3등분하여 W_q와 W_k 추출
                        W_qkv = layer.samples['weight']
                        print("W_qkv shape : ", W_qkv.shape)
            
            # config 설정 후 subnet이 제대로 반영되었는지 확인
            # print("=== Subnet 설정 후 각 Layer의 weight matrix 크기 확인 ===")
            n_parameters = sum(p.numel() for p in model_module.parameters() if p.requires_grad)
            print("model_module.parameters() : ", n_parameters)
            # for name, param in model_module.named_parameters():
            #     if param.requires_grad:  # 활성화된 weight만 출력
            #         print(f"Layer: {name}, Shape: {param.shape}")
            # print("==========================================")

            # for layer in model.modules():
            #     # print("layer : ", layer)
            #     if mode == 'channel' and hasattr(layer, 'dont_ch_prune'):
            #         continue
            #     if isinstance(layer, nn.Linear) and layer.samples:
            #         layer_metric = spectral_norm(layer, device)
            #         print("layer_metric : ", layer_metric)
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
