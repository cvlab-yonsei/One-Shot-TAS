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
from training_free import *
import random
import time

import torch.nn as nn
from model.supernet_transformer import Vision_TransformerSuper
import matplotlib.pyplot as plt
import os

@torch.no_grad()
def set_arc(model, config):
    metric_logger = utils.MetricLogger(delimiter="  ")

    # switch to evaluation mode
    model.eval()
    model_module = unwrap_model(model)
    model_module.set_sample_config(config=config)

    return

def sum_arr(arr):
    sum = 0.
    for i in range(len(arr)):
        sum += torch.sum(arr[i])
    return sum.item()


def get_layer_metric_array_dss(net, metric, mode, device, inputs):
    metric_array = []

    for layer in net.modules():
        if mode == 'channel' and hasattr(layer, 'dont_ch_prune'):
            continue
        # if isinstance(layer, nn.Linear) and 'qkv' in layer._get_name() and layer.samples:
        # print(layer._get_name())
        if isinstance(layer, nn.Linear) and 'qkv' in layer._get_name() and layer.samples:
        # if 'Attention' in layer._get_name():
            val = metric(layer)
            if val is not None:
                metric_array.append(val)
        if isinstance(layer, nn.Linear) and layer.samples:
            metric_array.append(metric(layer))
        if isinstance(layer, torch.nn.Linear) and layer.out_features == 1000:
            metric_array.append(metric(layer))
    print("metric_length : ", len(metric_array))
    # 텐서를 float 값으로 변환하여 리스트에 저장
    float_metric_array = [t.item() for t in metric_array]

    # 결과 출력
    print("metric :", float_metric_array)
    
    return metric_array


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


def evaluate_single_path(config, model, criterion, validation_data_loader, mixup_fn, amp, device):
    # DataParallel로 감싼 모델의 실제 모델에 접근하기 위해 model.module 사용
    model_module = unwrap_model(model.module)
    model_module.set_sample_config(config=config)

    validation_data_iter = iter(validation_data_loader)  # 독립적인 반복자 생성

    # Validation 데이터에서 배치 하나 가져오기
    try:
        val_samples, val_targets = next(validation_data_iter)
    except StopIteration:
        # Validation 데이터가 끝났으면 다시 처음부터 시작
        validation_data_iter = iter(validation_data_loader)
        val_samples, val_targets = next(validation_data_iter)

    val_samples = val_samples.to(device, non_blocking=True)
    val_targets = val_targets.to(device, non_blocking=True)

    if mixup_fn is not None:
        val_samples, val_targets = mixup_fn(val_samples, val_targets)

    if amp:
        with torch.cuda.amp.autocast():
            val_outputs = model(val_samples)
            val_loss = criterion(val_outputs, val_targets)
    else:
        val_outputs = model(val_samples)
        val_loss = criterion(val_outputs, val_targets)

    return val_loss.item(), config

# 병렬 처리로 경로 평가
def evaluate_paths_parallel(sampled_paths, model, criterion, validation_data_loader, mixup_fn, amp, device, max_workers=2):
    # 병렬 처리에서 독립적인 validation data iterator를 사용하도록 설정
    losses = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(evaluate_single_path, config, model, criterion, validation_data_loader, mixup_fn, amp, device)
            for config in sampled_paths
        ]
        for future in concurrent.futures.as_completed(futures):
            losses.append(future.result())  # 손실 및 config 저장

    return losses

def evaluate_paths(sampled_paths, model, criterion, validation_data_loader, mixup_fn, amp, device):
    losses = []
    validation_data_iter = iter(validation_data_loader)

    with torch.no_grad():  # 손실 계산에서 자동 기울기 추적을 방지하여 메모리 절약
        for config in sampled_paths:
            model_module = unwrap_model(model)
            model_module.set_sample_config(config=config)

            # Validation 데이터에서 배치 하나 가져오기 (즉시 가져와 사용)
            try:
                val_samples, val_targets = next(validation_data_iter)
            except StopIteration:
                # Validation 데이터가 끝났으면 다시 처음부터 시작
                validation_data_iter = iter(validation_data_loader)
                val_samples, val_targets = next(validation_data_iter)

            val_samples = val_samples.to(device, non_blocking=True)
            val_targets = val_targets.to(device, non_blocking=True)

            if mixup_fn is not None:
                val_samples, val_targets = mixup_fn(val_samples, val_targets)

            if amp:
                with torch.cuda.amp.autocast():
                    val_outputs = model(val_samples)  # 병렬 처리 가능
                    val_loss = criterion(val_outputs, val_targets)
            else:
                val_outputs = model(val_samples)  # 병렬 처리 가능
                val_loss = criterion(val_outputs, val_targets)

            # 손실 및 설정 저장
            losses.append((val_loss.item(), config))

            # GPU 메모리 해제
            del val_outputs
            torch.cuda.empty_cache()  # 캐시된 메모리 해제

    return losses

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

    # Set random seed
    random.seed(epoch)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    # Calculate T from data_loader total size and batch size
    T = len(data_loader)  # Total number of iterations (total data / batch size)
    # print("pool_sampling_prob : ", pool_sampling_prob)

    if mode == 'super':
        model_module = unwrap_model(model)
        total_iters = T // k  # T/k번 반복할 수 있도록 설정

        data_iter = iter(metric_logger.log_every(data_loader, print_freq, header))
        
        # Validation data iterator for selecting top-k paths
        validation_data_iter = iter(validation_data_loader)

        for iter_num in range(total_iters):  # T/k번 반복
            sampled_paths = []
            
            # # Original (주석))
            # for _ in range(k):  # m개의 경로를 샘플링
            #     config = sample_configs(choices=choices)

            #     sampled_paths.append(config)
                
                
            # ########
            
            # Algorithm 1: Sample m paths (경로만 샘플링하고, 데이터는 이후에 로드)
            for _ in range(m):  # m개의 경로를 샘플링
                if candidate_pool and random.random() < pool_sampling_prob:
                    config = random.choice(candidate_pool)
                else:
                    config = sample_configs(choices=choices)

                sampled_paths.append(config)
            
            # CUDA 메모리 부족 방지: sampled_paths 출력
            # print("sampled_paths : ", sampled_paths)
            
            # # Example of using the evaluation function
            # losses = evaluate_paths(sampled_paths, model, criterion, validation_data_loader, mixup_fn, amp, device)

            ##########
            # Step 7-8: Evaluate the loss for each path (그때그때 validation 배치를 로드하여 손실 계산)
            losses = []
            # with torch.no_grad():  # 손실 계산에서 자동 기울기 추적을 방지하여 메모리 절약
            for config in sampled_paths:
                # model.eval()
                
                model_module = unwrap_model(model)
                model_module.set_sample_config(config=config)

                ###### loss original START
                # # Validation 데이터에서 배치 하나 가져오기 (즉시 가져와 사용)
                # try:
                #     val_samples, val_targets = next(validation_data_iter)
                # except StopIteration:
                #     # Validation 데이터가 끝났으면 다시 처음부터 시작
                #     validation_data_iter = iter(validation_data_loader)
                #     val_samples, val_targets = next(validation_data_iter)
                
                # # val_samples와 val_targets의 크기를 출력
                # # print(f"val_samples shape: {val_samples.shape}")
                # # print(f"val_targets shape: {val_targets.shape}")
                
                # val_samples = val_samples.to(device, non_blocking=True)
                # val_targets = val_targets.to(device, non_blocking=True)

                # if mixup_fn is not None:
                #     val_samples, val_targets = mixup_fn(val_samples, val_targets)

                # if amp:
                #     with torch.cuda.amp.autocast():
                #         val_outputs = model(val_samples)
                #         val_loss = criterion(val_outputs, val_targets)
                # else:
                #     val_outputs = model(val_samples)
                #     val_loss = criterion(val_outputs, val_targets)
                
                # # 손실 및 설정 저장
                # losses.append((val_loss.item(), config))
                
                ###### loss original END
                
                
                
                ###### DSS START
                # 모델을 학습 모드로 설정
                model_module.train()

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
                inputs = torch.ones([1] + input_dim).float().to(device).requires_grad_(True)

                # 네트워크 forward 및 backward
                model_module.zero_grad()
                output = model_module(inputs)
                loss = torch.sum(output)  # 모든 출력의 합을 손실로 사용
                loss.backward()  # 그라디언트 계산
                
                # DSS 점수 계산
                metric_array = []
                for layer in model_module.modules():
                    if hasattr(layer, 'dont_ch_prune'):
                        continue
                    if isinstance(layer, nn.Linear) and layer.samples:
                        metric_array.append(dss(layer))
                    if isinstance(layer, torch.nn.Linear) and layer.out_features == 1000:
                        metric_array.append(dss(layer))

                # DSS 점수 합계 계산
                sum_dss_score = sum_arr(metric_array)
                # sum_dss_score = sum([torch.sum(metric).item() for metric in metric_array if metric is not None])

                # DSS 점수 및 설정 저장
                losses.append((sum_dss_score, config, model_module.get_sampled_params_numel(config))) # n_parameters

                # 원래 네트워크 파라미터 복구
                nonlinearize(model_module, signs)
                # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
                # set_arc(model, config)

                # indicators = compute_indicators.find_indicators(model,
                #                     validation_data_loader,
                #                     ('random', 1, 1000),
                #                     device,
                #                     # indicator_names=['snip']
                #                     )
                # print('indicators : ', indicators)
                # DSS 점수 및 설정 저장
                # losses.append((indicators['dss'], config, model_module.get_sampled_params_numel(config))) # n_parameters

                    
                    ###### DSS END

                    # GPU 메모리 해제
                    # del val_outputs
                    # torch.cuda.empty_cache()  # 캐시된 메모리 해제
            #######
            # model_p = DataParallel(model)
            # # Example usage
            # losses = evaluate_paths_parallel(sampled_paths, model_p, criterion, validation_data_loader, mixup_fn, amp, device)

            # Step 9: Rank and keep top-k paths based on loss
            losses.sort(key=lambda x: x[0], reverse=True)  # Sort by loss(dss) value (higher is better)
            # losses.sort(key=lambda x: x[0])  # Sort by loss value (lower is better)
            top_k_paths = losses[:k]

            #########

            # top_k_paths = sampled_paths

            # 연산 종료 후, top_k_paths를 candidate_pool에 추가
            if candidate_pool is not None:
                candidate_pool[:] = [config for _, config in top_k_paths]
                # candidate_pool[:] = top_k_paths  # candidate_pool 값을 top_k_paths로 대체

            # CUDA 메모리 부족 방지: top_k_paths 출력
            print("top_k_paths : ", top_k_paths)

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

                # # 메모리 해제
                # del outputs
                # torch.cuda.empty_cache()

    elif mode == 'retrain':
        config = retrain_config
        model_module = unwrap_model(model)
        model_module.set_sample_config(config=config)
        print(config)
        print(model_module.get_sampled_params_numel(config))

        for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
            samples = samples.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

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
        # print('images shape : ', images.shape)
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