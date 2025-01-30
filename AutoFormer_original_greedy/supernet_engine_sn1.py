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

import os

import pickle

# ✅ 모델 가중치가 NaN/Inf인지 검사하는 함수
def is_model_valid(model):
    for param in model.parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            return False  # NaN 또는 Inf 포함된 경우 비정상
    return True  # 모든 가중치가 정상

# ✅ 간소화된 체크포인트 저장 함수
def save_checkpoint(model, optimizer, checkpoint_path="last_checkpoint.pth"):
    if is_model_valid(model):
        try:
            utils.save_on_master({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, checkpoint_path)
            print(f"✅ Checkpoint saved: {checkpoint_path}")

            # try:
            #     checkpoint = torch.load("last_checkpoint.pth", map_location="cpu")
            #     print("✅ Checkpoint successfully loaded!")
            # except Exception as e:
            #     print(f"❌ Checkpoint file is corrupted: {e}")

        except Exception as e:
            print(f"❌ Failed to save checkpoint: {e}")
    else:
        print("⚠️ Warning: Model contains NaN/Inf, checkpoint not saved!")

# ✅ 간소화된 체크포인트 로드 함수
def load_checkpoint(model, optimizer, checkpoint_path="last_checkpoint.pth"):
    if not os.path.exists(checkpoint_path):
        print(f"⚠️ Checkpoint not found: {checkpoint_path}, skipping load.")
        return

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")  # GPU/CPU 불일치 방지
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"✅ Checkpoint restored from {checkpoint_path}")
    except Exception as e:
        print(f"❌ RuntimeError during checkpoint loading: {e}")


def sample_config_from_topk(model: torch.nn.Module, choices: Dict, m: int, k: int, device: torch.device, 
                            candidate_pool: List = None, pool_sampling_prob: float = 0.0) -> List:
    model.eval()
    model_module = unwrap_model(model)
    
    # 모델 상태 저장
    original_state = {name: param.clone() for name, param in model.state_dict().items()}

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
    # nonlinearize(model_module, signs)

    sampled_paths = []
    groups = {i: [] for i in range(5)}
    
    # 모델 상태 복원
    model.load_state_dict(original_state) # 이걸 아래 while문 안에 넣어야되나

    # Sample m paths
    with torch.no_grad():
        seen_configs = set()  # 중복 확인을 위한 집합 추가
        while len(sampled_paths) < m:
            if candidate_pool and random.random() <= pool_sampling_prob:
                config = random.choice(candidate_pool)
            else:
                config = sample_configs(choices)
            param_count = model_module.get_sampled_params_numel(config)
            group = get_group(param_count)
            
            # config를 튜플 형태로 변환하여 중복 확인 (딕셔너리는 해시 불가능하므로 튜플로 변환)
            config_tuple = tuple((k, tuple(v) if isinstance(v, list) else v) for k, v in sorted(config.items()))
            
            if config_tuple not in seen_configs:  # 중복되지 않은 경우에만 추가
                sampled_paths.append(config)
                groups[group].append(config)
                seen_configs.add(config_tuple)  # 중복 확인 집합에 추가

        losses = []
        for config in sampled_paths:
            set_arc(model, config)
            # signs = linearize(model_module)
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
            # nonlinearize(model_module, signs)
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
    random.shuffle(top_k_paths) # 아거 지워서도 실험해보자.
    
    nonlinearize(model_module, signs)

    # 모델 상태 복원
    model.load_state_dict(original_state)
    model.train()
    
    
    # top_k_paths에서 config만 반환
    return [config for _, config, _, _ in top_k_paths]
    
    # ####
    # random_k_paths = []
    
    # for _ in range(m):
    #     random_k_paths.append(sample_configs(choices))
        
    # return random_k_paths
    # ####


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
            
            # spectral_norm 값을 랜덤 값으로 변경 (기존 값의 영향 없이 생성)
            # random_spectral_norm = torch.empty(1).uniform_(0 * spectral_norm, 10).item()
            
            return torch.tensor(spectral_norm).to(device)
            # return torch.tensor(random_spectral_norm).to(device)
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
    print_freq = 1 # 원래 10
    
    config_list = []
    
    if mode == 'retrain':
        config = retrain_config
        model_module = unwrap_model(model)
        print(config)
        model_module.set_sample_config(config=config)
        print(model_module.get_sampled_params_numel(config))
    else:
        # 만약 candidate_pool이 not이면 pkl load.
        if not candidate_pool and os.path.exists('candidate_pool__midtraining12-no-train-random-440-m400k200-1batch5config-interval-1-top(original_pool_no_duplicate_full_0.8_linear)_467.pkl'):  # 조건이 참일 경우
            with open('candidate_pool__midtraining12-no-train-random-440-m400k200-1batch5config-interval-1-top(original_pool_no_duplicate_full_0.8_linear)_467.pkl', 'rb') as file:
                candidate_pool = pickle.load(file)
                print("candidate_pool__midtraining12 load : ", len(candidate_pool))

        if interval == 1:
            config_list = sample_config_from_topk(model, choices, m, k, device, candidate_pool, pool_sampling_prob) # 나중에 400ep으로 이거 0.8로 실험 한번 더
            # config_list = sample_config_from_topk(model, choices, m, k, device, candidate_pool, 0) # 이렇게 실험해버린듯(첫 400ep)
            candidate_pool[:] = [config for config in config_list] # update candidate pool
        else:
            if not candidate_pool or epoch%interval == 0:
                # config_list = sample_config_from_topk(model, choices, m, k, device, candidate_pool, pool_sampling_prob)
                config_list = sample_config_from_topk(model, choices, m, k, device, candidate_pool, 0) # random pool
                candidate_pool[:] = [config for config in config_list] # update candidate pool
            else:
                config_list = candidate_pool
                
            print("config_list[:5] : ", config_list[:5])
            

        # candidate_pool을 pkl 파일로 저장
        with open('candidate_pool__midtraining12-no-train-random-440-m400k200-1batch5config-interval-1-top(original_pool_no_duplicate_full_0.8_linear).pkl', 'wb') as f:
            pickle.dump(candidate_pool, f)

        print("candidate_pool이 candidate_pool.pkl 파일로 저장되었습니다.")

            
    last_saved_iter = 0  # 마지막으로 저장한 iteration (중복 저장 방지)  

    for iter, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # sample random config
        if mode == 'super':
            optimizer.zero_grad()  # 그래디언트 초기화 ???

            model_module = unwrap_model(model)

            # # config = sample_configs(choices=choices)
            # config = config_list[iter%k]
            # model_module = unwrap_model(model)
            # model_module.set_sample_config(config=config)

            # 1. 현재 iter에 대한 config + random sampling된 4개의 config 선택
            current_config = config_list[iter % k]
            additional_configs = random.sample(config_list, 4)  # config_list에서 4개 랜덤 선택
            selected_configs = [current_config] + additional_configs  # 총 5개의 config
            
            
        elif mode == 'retrain':
            config = retrain_config
            model_module = unwrap_model(model)
            model_module.set_sample_config(config=config)
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        loss = 0

        if amp:
            with torch.cuda.amp.autocast():
                if teacher_model:
                    with torch.no_grad():
                        teach_output = teacher_model(samples)
                    _, teacher_label = teach_output.topk(1, 1, True, True)
                    outputs = model(samples)
                    loss = 1/2 * criterion(outputs, targets) + 1/2 * teach_loss(outputs, teacher_label.squeeze())
                else:
                    # outputs = model(samples)
					# loss = criterion(outputs, targets)
					# 2. 각 config에 대해 forward 수행 및 손실 누적
                    for config in selected_configs:
                        model_module.set_sample_config(config=config)
                        with torch.cuda.amp.autocast(enabled=amp):
                            outputs = model(samples)
                            loss = criterion(outputs, targets) / len(selected_configs)  # Config별 손실 계산
                        loss.backward()  # 각 config마다 backward 수행
        else:
            if teacher_model:
                outputs = model(samples)
                with torch.no_grad():
                    teach_output = teacher_model(samples)
                _, teacher_label = teach_output.topk(1, 1, True, True)
                loss = 1 / 2 * criterion(outputs, targets) + 1 / 2 * teach_loss(outputs, teacher_label.squeeze())
            else:
                # outputs = model(samples)
                # loss = criterion(outputs, targets)
                # 2. 각 config에 대해 forward 수행 및 손실 누적
                for config in selected_configs:
                    model_module.set_sample_config(config=config)
                    with torch.cuda.amp.autocast(enabled=amp):
                        outputs = model(samples)
                        loss = criterion(outputs, targets) / len(selected_configs)
                    loss.backward()  # 각 config마다 backward 수행

        loss_value = loss.item()

        # if not math.isfinite(loss_value):
        #     print("Loss is {}, stopping training".format(loss_value))
        #     sys.exit(1)
        
        # # 손실 값이 NaN 또는 Inf일 경우 해당 배치 건너뛰기
        # if not math.isfinite(loss_value):
        #     print(f"Warning: Loss is {loss_value}, skipping batch {iter}")
        #     optimizer.zero_grad()  # 그래디언트 초기화
        #     continue  # 현재 배치를 건너뛰고 다음 배치로 이동

        # 손실 값이 NaN 또는 Inf일 경우 해당 배치 스킵 및 체크포인트 복구
        if not math.isfinite(loss_value):
            print(f"Warning: Loss is {loss_value}, skipping batch {iter}")

            # 모델 가중치가 NaN이 되었는지 확인
            any_nan = any(torch.isnan(p).any() for p in model.parameters())
            any_inf = any(torch.isinf(p).any() for p in model.parameters())

            if any_nan or any_inf:
                print(f"Detected NaN/Inf in model parameters at iteration {iter}, reloading last checkpoint...")
                load_checkpoint(model, optimizer, "last_checkpoint.pth")
                print("Checkpoint restored. Resuming training.")

            optimizer.zero_grad()  # 그래디언트 초기화
            continue  # 배치 스킵

        # optimizer.zero_grad() # 호출 뒤로

        # this attribute is added by timm on one optimizer (adahessian)
        if amp:
            # # is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            # # loss_scaler(loss, optimizer, clip_grad=max_norm,
            # #         parameters=model.parameters(), create_graph=is_second_order)
            # is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            # # loss_scaler를 통해 optimizer.step() 호출
            # loss_scaler(loss, optimizer, clip_grad=max_norm, parameters=model.parameters(), create_graph=is_second_order)
            optimizer.step()
        else:
            # loss.backward()
            optimizer.step()

        optimizer.zero_grad()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # 매 iteration마다 체크포인트 저장 (이전 저장 iteration과 중복 방지)
        if iter % 10 == 0 and iter != last_saved_iter:  # 10 iter마다 저장 (조절 가능)
            save_checkpoint(model, optimizer, "last_checkpoint.pth")
            last_saved_iter = iter  # 마지막 저장된 iteration 기록
            print(f"Checkpoint saved at iteration {iter}")

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