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
import random
import time

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
        
        sampled_paths = [{'mlp_ratio': [4, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 4, 4, 3.5, 4, 3.5, 4], 'num_heads': [3, 4, 4, 3, 3, 4, 3, 4, 3, 3, 4, 4, 4], 'embed_dim': [192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192], 'layer_num': 13}, {'mlp_ratio': [3.5, 4, 3.5, 4, 3.5, 4, 3.5, 4, 3.5, 4, 3.5, 3.5, 3.5], 'num_heads': [4, 3, 3, 3, 3, 3, 3, 4, 4, 3, 4, 3, 3], 'embed_dim': [216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216], 'layer_num': 13}, {'mlp_ratio': [3.5, 4, 4, 3.5, 4, 3.5, 3.5, 3.5, 4, 4, 4, 3.5, 4, 4], 'num_heads': [3, 3, 3, 4, 3, 3, 3, 3, 4, 3, 3, 4, 3, 3], 'embed_dim': [240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240], 'layer_num': 14}, {'mlp_ratio': [4, 3.5, 4, 3.5, 4, 3.5, 3.5, 4, 3.5, 3.5, 4, 3.5, 4], 'num_heads': [4, 4, 4, 4, 4, 4, 3, 3, 3, 4, 4, 4, 3], 'embed_dim': [216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216, 216], 'layer_num': 13}]
        
        losses = []
        with torch.no_grad():  # 손실 계산에서 자동 기울기 추적을 방지하여 메모리 절약
            for config in sampled_paths:
                model_module = unwrap_model(model)
                model_module.set_sample_config(config=config)

                # Evaluate the model on the entire validation dataset
                val_loss_total = 0
                num_batches = 0
                
                for val_samples, val_targets in validation_data_loader:
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

                    # 각 배치의 손실을 더함
                    val_loss_total += val_loss.item()
                    num_batches += 1

                # 전체 배치의 평균 손실을 저장
                val_loss_avg = val_loss_total / num_batches
                losses.append((val_loss_avg, config))
                
        losses.sort(key=lambda x: x[0])  # Sort by loss value (lower is better)
        top_k_paths = losses[:k]
        
        # bottom_k_paths: 나머지 경로들을 loss가 큰 순서로 정렬
        bottom_k_paths = sorted(losses[k:], key=lambda x: x[0], reverse=True)
        
        #########
        
        # top_k_paths = sampled_paths
        
        # 연산 종료 후, top_k_paths를 candidate_pool에 추가
        if candidate_pool is not None:
            candidate_pool[:] = [config for _, config in top_k_paths]
            # candidate_pool[:] = top_k_paths  # candidate_pool 값을 top_k_paths로 대체
        
        # CUDA 메모리 부족 방지: top_k_paths 출력
        print("top_k_paths : ", top_k_paths)
        print("bottom_k_paths : ", bottom_k_paths)