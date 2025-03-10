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

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import pickle
import ast

def sample_configs(choices):

    config = {}
    dimensions = ['mlp_ratio', 'num_heads']
    depth = random.choice(choices['depth'])
    for dimension in dimensions:
        config[dimension] = [random.choice(choices[dimension]) for _ in range(depth)]

    config['embed_dim'] = [random.choice(choices['embed_dim'])]*depth

    config['layer_num'] = depth
    return config

def train_one_epoch_original(model: torch.nn.Module, criterion: torch.nn.Module,
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
def evaluate_original(data_loader, model, device, amp=True, choices=None, mode='super', retrain_config=None):
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


# @torch.no_grad()
def evaluate_and_visualize(data_loader, model, device, amp=True, good_configs=None, bad_configs=None, good_configs2=None, bad_configs2=None, reference_config=None, reference_config2=None, reference_config3=None):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    # 모든 파라미터에 requires_grad=True 설정 (gradient 계산을 위해)
    for param in model.parameters():
        param.requires_grad = True

    gradients_good = []
    gradients_bad = []
    gradients_good_6M = []
    gradients_bad_10M = []

    def get_gradient(config):
        """주어진 subnet config에 대해 gradient를 계산"""
        
        # 문자열이면 딕셔너리로 변환
        if isinstance(config, str):
            config = ast.literal_eval(config)

        model_module = unwrap_model(model)
        model_module.set_sample_config(config=config)
        model.zero_grad()

        for images, target in data_loader:
            images, target = images.to(device), target.to(device)

            if amp:
                with torch.cuda.amp.autocast():
                    output = model(images)
                    loss = criterion(output, target)
            else:
                output = model(images)
                loss = criterion(output, target)

            loss.backward()  # Gradient 계산
            break  # 첫 번째 배치에 대해서만 수행

        gradients = []
        for param in model.parameters():
            if param.grad is not None:
                gradients.append(param.grad.view(-1).cpu().numpy())  # Gradient를 1D 벡터로 변환
        return np.concatenate(gradients)  # 1D 벡터로 반환

    ########
    # Reference Model (Ground Truth Gradient) 계산
    model.zero_grad()
    model_module = unwrap_model(model)
    model_module.set_sample_config(config=reference_config)  # 가장 강한 모델 또는 best-performing subnet
    for images, target in data_loader:
        images, target = images.to(device), target.to(device)

        if amp:
            with torch.cuda.amp.autocast():
                output = model(images)
                loss = criterion(output, target)
        else:
            output = model(images)
            loss = criterion(output, target)

        loss.backward()  # Reference gradient 계산
        break  # 첫 번째 배치만 사용

    reference_gradient = []
    for param in model.parameters():
        if param.grad is not None:
            reference_gradient.append(param.grad.view(-1).cpu().numpy())
    reference_gradient = np.concatenate(reference_gradient)
    ##########

    ########
    # Reference Model (Ground Truth Gradient)2 계산
    model.zero_grad()
    model_module = unwrap_model(model)
    model_module.set_sample_config(config=reference_config2)  # 가장 강한 모델 또는 best-performing subnet
    for images, target in data_loader:
        images, target = images.to(device), target.to(device)

        if amp:
            with torch.cuda.amp.autocast():
                output = model(images)
                loss = criterion(output, target)
        else:
            output = model(images)
            loss = criterion(output, target)

        loss.backward()  # Reference gradient 계산
        break  # 첫 번째 배치만 사용

    reference_gradient_6M = []
    for param in model.parameters():
        if param.grad is not None:
            reference_gradient_6M.append(param.grad.view(-1).cpu().numpy())
    reference_gradient_6M = np.concatenate(reference_gradient_6M)
    ##########

    ########
    # Reference Model (Ground Truth Gradient)3 계산
    model.zero_grad()
    model_module = unwrap_model(model)
    model_module.set_sample_config(config=reference_config3)  # 가장 강한 모델 또는 best-performing subnet
    for images, target in data_loader:
        images, target = images.to(device), target.to(device)

        if amp:
            with torch.cuda.amp.autocast():
                output = model(images)
                loss = criterion(output, target)
        else:
            output = model(images)
            loss = criterion(output, target)

        loss.backward()  # Reference gradient 계산
        break  # 첫 번째 배치만 사용

    reference_gradient_supernet = []
    for param in model.parameters():
        if param.grad is not None:
            reference_gradient_supernet.append(param.grad.view(-1).cpu().numpy())
    reference_gradient_supernet = np.concatenate(reference_gradient_supernet)
    ##########

    good = 1
    bad = 1
    # Good config에 대한 gradient 저장
    for config in good_configs:
        print("good ", good)
        # print('config : ', config)
        good += 1
        grad_vector = get_gradient(config['config'])
        gradients_good.append(grad_vector)

    # Bad config에 대한 gradient 저장
    for config in bad_configs:
        print("good ", bad)
        bad += 1
        grad_vector = get_gradient(config['config'])
        gradients_bad.append(grad_vector)


    good = 1
    bad = 1
    # Good config에 대한 gradient 저장
    for config in good_configs2:
        print("good ", good)
        # print('config : ', config)
        good += 1
        grad_vector = get_gradient(config['config'])
        gradients_good_6M.append(grad_vector)

    # Bad config에 대한 gradient 저장
    for config in bad_configs2:
        print("good ", bad)
        bad += 1
        grad_vector = get_gradient(config['config'])
        gradients_bad_10M.append(grad_vector)

    saved_data = {
        "gradients_good": gradients_good,
        "gradients_bad": gradients_bad,
        "reference_gradient": reference_gradient,
        "gradients_good_6M": gradients_good_6M,
        "gradients_bad_10M": gradients_bad_10M,
        "reference_direction_6M": reference_gradient_6M,
        "reference_direction_supernet": reference_gradient_supernet,
    }

    # pkl 파일로 저장
    with open("gradient_data_full_saved_data.pkl", "wb") as f:
        pickle.dump(saved_data, f)

    print("Gradient data has been saved to 'gradient_data_full_saved_data.pkl'")

    # 2D 시각화를 위한 PCA 변환
    all_gradients = np.vstack([gradients_good, gradients_bad, reference_gradient.reshape(1, -1)])
    pca = PCA(n_components=2)
    transformed_gradients = pca.fit_transform(all_gradients)

    # 변환된 gradient 분리
    transformed_good = transformed_gradients[:len(good_configs)]
    transformed_bad = transformed_gradients[len(good_configs):-1]
    transformed_reference = transformed_gradients[-1].reshape(1, -1)  # Reference gradient

    # Gradient 방향 및 크기 정규화
    good_directions = transformed_good / np.linalg.norm(transformed_good, axis=1, keepdims=True)
    bad_directions = transformed_bad / np.linalg.norm(transformed_bad, axis=1, keepdims=True)
    reference_direction = transformed_reference / np.linalg.norm(transformed_reference)

    gradient_data = {
        "transformed_good": transformed_good,
        "transformed_bad": transformed_bad,
        "transformed_reference": transformed_reference,
        "good_directions": good_directions,
        "bad_directions": bad_directions,
        "reference_direction": reference_direction
    }

    # pkl 파일로 저장
    with open("gradient_data.pkl", "wb") as f:
        pickle.dump(gradient_data, f)

    print("Gradient data has been saved to 'gradient_data.pkl'")

    # 시각화
    plt.figure(figsize=(8, 6))

    # Good config gradient (파란색 화살표)
    plt.quiver(transformed_good[:, 0], transformed_good[:, 1], 
               good_directions[:, 0], good_directions[:, 1], 
               angles='xy', scale_units='xy', scale=1.5, color='b', label='Good Subnet')

    # Bad config gradient (빨간색 화살표)
    plt.quiver(transformed_bad[:, 0], transformed_bad[:, 1], 
               bad_directions[:, 0], bad_directions[:, 1], 
               angles='xy', scale_units='xy', scale=1.5, color='r', label='Bad Subnet')

    # Reference Model 방향 (초록색)
    plt.quiver(0, 0, reference_direction[0, 0], reference_direction[0, 1], 
               angles='xy', scale_units='xy', scale=2, color='g', linewidth=2, label='Reference Model (Ground Truth)')

    plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("Comparison of Good and Bad Subnet Gradients")
    plt.legend()
    plt.grid()

    plt.savefig("gradient_visualization.png", dpi=300, bbox_inches='tight')

    plt.show()