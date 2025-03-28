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
import torch.nn.functional as F
from sklearn.decomposition import PCA
import numpy as np

def pca_project(tensor, dim=16):
    flat = tensor.view(-1).unsqueeze(1).cpu().numpy()
    n_samples, n_features = flat.shape

    if np.std(flat) < 1e-8:  # ⚠️ 표준편차 거의 0이면 무의미
        return torch.zeros(dim, device=tensor.device)

    n_components = min(dim, n_samples, n_features)
    if n_components < 1:
        return torch.zeros(dim, device=tensor.device)

    try:
        pca = PCA(n_components=n_components)
        proj = pca.fit_transform(flat)
        proj_tensor = torch.tensor(proj, device=tensor.device, dtype=torch.float32)
        return proj_tensor.flatten()
    except Exception as e:
        print(f"[PCA Fail] {e}")
        return torch.zeros(dim, device=tensor.device)


def sample_configs(choices):
    config = {}
    dimensions = ['mlp_ratio', 'num_heads']
    depth = random.choice(choices['depth'])
    for dimension in dimensions:
        config[dimension] = [random.choice(choices[dimension]) for _ in range(depth)]

    config['embed_dim'] = [random.choice(choices['embed_dim'])]*depth
    config['layer_num'] = depth
    return config


def get_config_by_mode(choices, mode):
    depth = max(choices['depth']) if mode == 'maximum' else min(choices['depth'])
    config = {
        'embed_dim': [max(choices['embed_dim']) if mode == 'maximum' else min(choices['embed_dim'])] * depth,
        'num_heads': [max(choices['num_heads']) if mode == 'maximum' else min(choices['num_heads'])] * depth,
        'mlp_ratio': [max(choices['mlp_ratio']) if mode == 'maximum' else min(choices['mlp_ratio'])] * depth,
        'layer_num': depth
    }
    return config


def flatten_tensor(t):
    return t.view(-1)


def match_and_compare_grads(grad_small, grad_large):
    cos_sims_crop, mses_crop = [], []
    cos_sims_pad, mses_pad = [], []

    # print("grad_small : ", grad_small)
    # print("grad_large : ", grad_large)

    for name in grad_small:
        if name not in grad_large:
            continue

        g_small = grad_small[name]
        g_large = grad_large[name]

        if g_small is None or g_large is None:
            continue

        # print(f"{name}: small -> {g_small.shape}, large -> {g_large.shape}")

        # # Masked crop: zero-out g_large where g_small == 0
        # mask = (g_small != 0).to(dtype=g_large.dtype)  # 1.0 where g_small ≠ 0, 0.0 where g_small == 0
        # masked_large = g_large * mask  # same shape as g_large
        # # print("masked_large : ", )

        # # 마스킹: g_small이 0이 아닌 위치만 선택
        # mask = (g_small != 0.0000).to(dtype=torch.bool)

        # # 마스킹된 부분만 flatten해서 추출
        # flat_small = g_small[mask]
        # flat_large = masked_large[mask]

        # # PCA projection
        # proj_small = pca_project(flat_small)
        # proj_large_masked = pca_project(flat_large)

        # # Cosine similarity & MSE on PCA-reduced features
        # cos_crop = F.cosine_similarity(proj_small, proj_large_masked, dim=0).item()
        # mse_crop = F.mse_loss(proj_small, proj_large_masked).item()

        # cos_sims_crop.append(cos_crop)
        # mses_crop.append(mse_crop)

        # proj_small_pad = pca_project(g_small)
        # proj_large_pad = pca_project(g_large)

        # cos_pad = F.cosine_similarity(proj_small_pad, proj_large_pad, dim=0).item()
        # mse_pad = F.mse_loss(proj_small_pad, proj_large_pad).item()

        # cos_sims_pad.append(cos_pad)
        # mses_pad.append(mse_pad)



        # ============================
        mask = (g_small != 0.0000e+00).to(dtype=g_large.dtype)  # 1.0 where g_small ≠ 0, 0.0 where g_small == 0
        masked_large = g_large * mask  # same shape as g_large
        # print("masked_large : ", g_large)

        # 마스킹: g_small이 0이 아닌 위치만 선택
        mask = (g_small != 0.0000).to(dtype=torch.bool)

        # 마스킹된 부분만 flatten해서 추출
        flat_small = g_small[mask]
        flat_large = masked_large[mask]

        # cosine similarity & MSE 계산
        cos_crop = F.cosine_similarity(flat_small.unsqueeze(0), flat_large.unsqueeze(0), dim=1).item()
        mse_crop = F.mse_loss(flat_small, flat_large).item()

        cos_sims_crop.append(cos_crop)
        mses_crop.append(mse_crop)

        cos_pad = F.cosine_similarity(flatten_tensor(g_small), flatten_tensor(g_large), dim=0).item()
        mse_pad = F.mse_loss(g_small, g_large).item()

        cos_sims_pad.append(cos_pad)
        mses_pad.append(mse_pad)

    return {
        'cosine_crop': sum(cos_sims_crop) / len(cos_sims_crop),
        'mse_crop': sum(mses_crop) / len(mses_crop),
        'cosine_pad': sum(cos_sims_pad) / len(cos_sims_pad),
        'mse_pad': sum(mses_pad) / len(mses_pad)
    }


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    amp: bool = True, teacher_model: torch.nn.Module = None,
                    teach_loss: torch.nn.Module = None, choices=None, mode='super', retrain_config=None):

    model.train()
    criterion.train()
    random.seed(epoch)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    model_module = unwrap_model(model)

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        grads_dict = {}
        for config_name in ['maximum', 'minimum']:
            config = get_config_by_mode(choices, config_name)
            model_module.set_sample_config(config=config)

            print(config_name, config)

            print("lr : ", optimizer.param_groups[0]["lr"])

            optimizer.zero_grad()
            if mixup_fn is not None:
                mixed_samples, mixed_targets = mixup_fn(samples, targets)
            else:
                mixed_samples, mixed_targets = samples, targets

            if amp:
                with torch.cuda.amp.autocast():
                    outputs = model(mixed_samples)
                    loss = criterion(outputs, mixed_targets)
            else:
                outputs = model(mixed_samples)
                loss = criterion(outputs, mixed_targets)

            if amp:
                is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
                loss_scaler(loss, optimizer, clip_grad=max_norm,
                            parameters=model.parameters(), create_graph=is_second_order)
            else:
                loss.backward()

            grads_dict[config_name] = {
                name: param.grad.detach().clone() if param.grad is not None else None
                for name, param in model.named_parameters()
                if param.requires_grad
            }

        # Grad comparison
        results = match_and_compare_grads(grads_dict['minimum'], grads_dict['maximum'])
        print("[Grad Alignment Result]")
        print(f"Cosine Similarity (crop): {results['cosine_crop']:.20f}")
        print(f"MSE (crop): {results['mse_crop']:.20f}")
        print(f"Cosine Similarity (pad): {results['cosine_pad']:.20f}")
        print(f"MSE (pad): {results['mse_pad']:.20f}")

        break  # only run one batch for analysis

    return {}