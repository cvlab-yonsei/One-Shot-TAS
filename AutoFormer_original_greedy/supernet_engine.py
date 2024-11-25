import math
import sys
from typing import Iterable, Optional
from timm.utils.model import unwrap_model
import torch
import concurrent.futures
from torch.nn.parallel import DataParallel
from timm.data import Mixup
from timm.utils import accuracy, ModelEma
from lib import utils
# from training_free import *
import random
import time

import torch.nn as nn
from model.supernet_transformer import Vision_TransformerSuper
import matplotlib.pyplot as plt
import os

# def sum_arr(arr):
#     sum = 0.
#     for i in range(len(arr)):
#         sum += torch.sum(arr[i])
#     return sum.item()

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
                    candidate_pool=None, validation_data_loader=None, pool_sampling_prob=0, m=10, k=5):
    
    model.train()
    criterion.train()

    # set random seed
    random.seed(epoch)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    
    T = len(data_loader)  # Total number of iterations (total data / batch size)
    print("pool_sampling_prob : ", pool_sampling_prob)
        
    if mode == 'super':
        
        model.eval()
        model_module = unwrap_model(model)
        total_iters = T // k  # T/k번 반복할 수 있도록 설정
        print("T : ", T)
        print("total_iters : ", total_iters)

        data_iter = iter(metric_logger.log_every(data_loader, print_freq, header))
        
        # Validation data iterator for selecting top-k paths
        # validation_data_iter = iter(validation_data_loader)
        
        
        ### layer normalization START
        
        sampled_config = {}
        sampled_config['layer_num'] = 14
        sampled_config['mlp_ratio'] = [4.0] * 14
        sampled_config['num_heads'] = [4] * 14
        sampled_config['embed_dim'] = [256] * 14
        
        # model_module = unwrap_model(model)
        # model_module.set_sample_config(config=sampled_config)
        set_arc(model, sampled_config)
        
        ###### DSS START
        # 모델을 학습 모드로 설정
        # model_module.train()
        # model_module.eval()
        # model.eval()

        # DSS 계산을 위한 전처리
        total_param_count = {
            'ImagePatchEmbedding': 0,
            'MSA': 0,
            'MLP': 0,
            'LastLayer': 0,
            'None': 0
        }

        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

        signs = linearize(model_module)

        # 입력 텐서를 모든 픽셀이 1인 상태로 설정
        input_dim = [3, 224, 224]  # 예시로, 이미지 입력 크기를 224x224로 가정
        # inputs = torch.ones([1] + input_dim).float().to(device).requires_grad_(True)
        inputs = torch.ones([1] + input_dim).float().to(device)  # requires_grad_(True) 제거
        
        # 네트워크 forward 및 backward
        # model_module.zero_grad()
        # output = model_module(inputs)
        # loss = torch.sum(output)  # 모든 출력의 합을 손실로 사용
        # loss.backward()  # 그라디언트 계산
        # 네트워크 forward (backward 불필요)
        
        # DSS 점수 계산
        metric_array = []
        supernet_indicators = {}
        
        total = 0
        
        for layer in model.modules():
            supernet_indicators[layer._get_name() + '_' + str(total)] = torch.tensor(0).to(device)
            if mode == 'channel' and hasattr(layer, 'dont_ch_prune'):
                continue
            if isinstance(layer, nn.Linear) and layer.samples:
                # 텐서의 차원이 2D 이상일 때만 처리
                if isinstance(layer, nn.Linear) and 'qkv' in layer._get_name() and layer.samples:
                    layer_metric = spectral_norm(layer, device)
                    layer_key = layer._get_name() + '_' + str(total)
                    
                    supernet_indicators[layer_key] = layer_metric
                    # print("hello")
                    # print("layer_metric : ", layer_metric)
                    
            total += 1
            
        ### layer normalization END
        
        
        for iter_num in range(total_iters):  # T/k번 반복
            print("total_iters : ", total_iters)
            
            sampled_paths = []
            groups = {0: [], 1: [], 2: [], 3: [], 4: []}  # 4개의 그룹으로 나눌 공간
            
            # Algorithm 1: Sample m paths (경로만 샘플링하고, 데이터는 이후에 로드)
            while len(sampled_paths) < m:
                # candidate_pool에서 샘플링하거나 새로운 config 샘플링
                if candidate_pool and random.random() < pool_sampling_prob:
                    config = random.choice(candidate_pool)
                else:
                    config = sample_configs(choices=choices)

                # Parameter 수 계산
                param_count = model_module.get_sampled_params_numel(config)
                group = get_group(param_count)

                # 각 그룹이 고르게 분포하도록 제어
                # if len(groups[group]) < m // 4:
                # 중복되지 않도록 확인 후 추가
                if config not in sampled_paths:
                    sampled_paths.append(config)
                    groups[group].append(config)
                        
            print("len(sampled_paths) : ", len(sampled_paths))
            print("sampled_paths[0] : ", sampled_paths[0])
                
            # Step 7-8: Evaluate the loss for each path (그때그때 validation 배치를 로드하여 손실 계산)
            losses = []
            # with torch.no_grad():  # 손실 계산에서 자동 기울기 추적을 방지하여 메모리 절약
            for config_idx, config in enumerate(sampled_paths):
                # model.eval()
                # print("config_idx : ", config_idx)
                
                # model_module = unwrap_model(model)
                # model_module.set_sample_config(config=config)
                set_arc(model, config)
                
                ###### DSS START
                # 모델을 학습 모드로 설정
                # model_module.train()
                # model_module.eval()
                # model.eval()

                # DSS 계산을 위한 전처리
                total_param_count = {
                    'ImagePatchEmbedding': 0,
                    'MSA': 0,
                    'MLP': 0,
                    'LastLayer': 0,
                    'None': 0
                }

                # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

                signs = linearize(model_module)

                # 입력 텐서를 모든 픽셀이 1인 상태로 설정
                input_dim = [3, 224, 224]  # 예시로, 이미지 입력 크기를 224x224로 가정
                # inputs = torch.ones([1] + input_dim).float().to(device).requires_grad_(True)
                inputs = torch.ones([1] + input_dim).float().to(device)

                # # 네트워크 forward 및 backward
                # model_module.zero_grad()
                # output = model_module(inputs)
                # loss = torch.sum(output)  # 모든 출력의 합을 손실로 사용
                # loss.backward()  # 그라디언트 계산
                
                # DSS 점수 계산
                metric_array = []
                total = 0
                # print("supernet_indicators : ", supernet_indicators)
                for layer in model.modules():
                    norm_val = supernet_indicators[layer._get_name()+'_'+str(total)]
                    if hasattr(layer, 'dont_ch_prune'):
                        continue
                    if isinstance(layer, nn.Linear) and layer.samples:
                        # metric_array.append(dss(layer))
                        if norm_val != 0:
                            metric_array.append(spectral_norm(layer, device) / norm_val)
                        
                    if isinstance(layer, torch.nn.Linear) and layer.out_features == 1000:
                        # metric_array.append(dss(layer))
                        if norm_val != 0:
                            metric_array.append(spectral_norm(layer, device) / norm_val)
                    total += 1

                # print("len metric_array : ", len(metric_array))
                # print("metric_array : ", metric_array)
                # DSS 점수 합계 계산
                sum_dss_score = sum_arr(metric_array)
                # sum_dss_score = sum([torch.sum(metric).item() for metric in metric_array if metric is not None])

                param_count = model_module.get_sampled_params_numel(config)
                group = get_group(param_count)  # 그룹 결정
                
                if config_idx % 100 == 0:
                    print(f"[{config_idx} / {len(sampled_paths)}] : (DSS Score: {sum_dss_score}, Config: {config}, Params: {param_count}, Group: {group})")

                losses.append((sum_dss_score, config, param_count, group))
                # 원래 네트워크 파라미터 복구
                nonlinearize(model_module, signs)
                
            # Step 9: 그룹별로 정렬 및 top-k 선택
            # target_per_group = k // 5
            # remainder = k % 5

            # top_k_paths = []

            # for group_id in range(5):
            #     # 각 그룹에 속하는 losses 필터링
            #     group_losses = [loss for loss in losses if loss[3] == group_id]
            #     group_losses.sort(key=lambda x: x[0], reverse=True)  # DSS 점수 기준 내림차순 정렬

            #     # 각 그룹에서 필요한 수 만큼 선택
            #     num_to_select = target_per_group + (1 if group_id < remainder else 0)
            #     top_k_paths.extend(group_losses[:num_to_select])
            
            # Step 9: 그룹별로 정렬 및 상위 50% 선택
            top_k_paths = []

            # 그룹별로 상위 50% 선택하고 top_k_paths에 추가
            remaining_items = []  # 상위 50%에 들지 못한 아이템들을 저장할 리스트

            for group_id in range(5):
                # 각 그룹에 속하는 losses 필터링
                group_losses = [loss for loss in losses if loss[3] == group_id]
                group_losses.sort(key=lambda x: x[0], reverse=True)  # DSS 점수 기준 내림차순 정렬

                # 그룹에서 상위 50% 선택
                num_to_select = max(1, len(group_losses) // 2)  # 상위 50% 선택, 최소 1개는 선택
                top_k_paths.extend(group_losses[:num_to_select])

                # 상위 50%에 들지 못한 나머지 아이템을 remaining_items에 추가
                remaining_items.extend(group_losses[num_to_select:])

            # 부족한 경우 상위에서 탈락한 아이템을 추가하여 k개로 만듦
            if len(top_k_paths) < k:
                # remaining_items를 DSS 점수 기준으로 내림차순 정렬
                remaining_items.sort(key=lambda x: x[0], reverse=True)
                # 부족한 수만큼 remaining_items에서 추가
                top_k_paths.extend(remaining_items[:k - len(top_k_paths)])
                
            # 최종 top_k_paths의 순서를 랜덤하게 섞기
            random.shuffle(top_k_paths)
            
            #########

            # top_k_paths = sampled_paths

            # 연산 종료 후, top_k_paths를 candidate_pool에 추가
            if candidate_pool is not None:
                # candidate_pool[:] = [config for _, config in top_k_paths]
                candidate_pool[:] = [config for _, config, _, _ in top_k_paths]
                # candidate_pool[:] = top_k_paths  # candidate_pool 값을 top_k_paths로 대체

            # CUDA 메모리 부족 방지: top_k_paths 출력
            # print("top_k_paths : ", top_k_paths)

            # Step 4: Train on top-k paths using actual training data
            for _ in range(k):  # 각 경로에 대해 실제 데이터로 학습
                samples, targets = next(data_iter)
                samples = samples.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                # Get config from top_k_paths
                config = top_k_paths.pop(0)[1]  # Get the configuration
                # config = top_k_paths.pop(0)

                model_module.set_sample_config(config=config)
                optimizer.zero_grad()

                if mixup_fn is not None:
                    samples, targets = mixup_fn(samples, targets)

                if amp:
                    with torch.cuda.amp.autocast():
                        outputs = model(samples)
                        loss = criterion(outputs, targets)
                else:
                    outputs = model(samples)
                    loss = criterion(outputs, targets)

                loss_value = loss.item()

                if not math.isfinite(loss_value):
                    print("Loss is {}, stopping training".format(loss_value))
                    sys.exit(1)

                # Backward and optimization
                if amp:
                    is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
                    loss_scaler(loss, optimizer, clip_grad=max_norm, parameters=model.parameters(), create_graph=is_second_order)
                else:
                    loss.backward()
                    optimizer.step()

                torch.cuda.synchronize()

                if model_ema is not None:
                    model_ema.update(model)

                metric_logger.update(loss=loss_value)
                metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            
        
    
    elif mode == 'retrain':
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
