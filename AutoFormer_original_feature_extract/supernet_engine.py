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

import matplotlib.pyplot as plt
import seaborn as sns

def sample_configs(choices):

    config = {}
    dimensions = ['mlp_ratio', 'num_heads']
    depth = random.choice(choices['depth'])
    for dimension in dimensions:
        config[dimension] = [random.choice(choices[dimension]) for _ in range(depth)]

    config['embed_dim'] = [random.choice(choices['embed_dim'])]*depth

    config['layer_num'] = depth
    return config

def extract_middle_feature(model, x):
    features = {}

    def hook_fn(module, input, output):
        features['feat'] = output

    # print(model)
    handle = model.module.blocks[0].register_forward_hook(hook_fn)
    _ = model(x)
    handle.remove()
    return features['feat']

@torch.no_grad()
def run_feature_alignment(model, data_loader, device, config_small, config_large):
    model_module = unwrap_model(model)
    model.eval()

    for images, _ in data_loader:
        images = images.to(device, non_blocking=True)
        x = images[:1]  # 하나의 이미지 샘플만 사용
        break

    model_module.set_sample_config(config_large) # maximum에서 학습시키고 큰 subnet에서 추출한 feature
    feat_large = extract_middle_feature(model, x)

    model_module.set_sample_config(config_small) # maximum에서 학습시키고 작은 subnet에서 추출한 feature
    feat_small = extract_middle_feature(model, x)

    print("feat_large : ", feat_large)
    print("feat_small : ", feat_small)

    print("feat_large.shape : ", feat_large.shape)
    print("feat_small.shape : ", feat_small.shape)

        # 1. Upsampling
    feat_small_upsampled = F.interpolate(feat_small.permute(0, 2, 1), size=feat_large.shape[-1], mode='linear', align_corners=False).permute(0, 2, 1)
    sim_upsample = F.cosine_similarity(feat_small_upsampled.flatten(1), feat_large.flatten(1), dim=1).mean().item()
    mse_upsample = F.mse_loss(feat_small_upsampled, feat_large).item()

    # 2. Crop large to small
    feat_large_cropped = feat_large[:, :, :feat_small.shape[-1]]
    sim_crop = F.cosine_similarity(feat_small.flatten(1), feat_large_cropped.flatten(1), dim=1).mean().item()
    mse_crop = F.mse_loss(feat_small, feat_large_cropped).item()

    # 3. Expand small to large (zero padding)
    B, N, D = feat_small.shape
    pad_dim = feat_large.shape[-1] - D
    feat_small_padded = torch.cat([feat_small, torch.zeros((B, N, pad_dim), device=feat_small.device)], dim=-1)
    sim_expand = F.cosine_similarity(feat_small_padded.flatten(1), feat_large.flatten(1), dim=1).mean().item()
    mse_expand = F.mse_loss(feat_small_padded, feat_large).item()

    print("[Feature Alignment Result]")
    print(f"Cosine Similarity (upsample): {sim_upsample:.4f}")
    print(f"Cosine Similarity (crop): {sim_crop:.4f}")
    print(f"Cosine Similarity (expand): {sim_expand:.4f}")
    print(f"MSE (upsample): {mse_upsample:.6f}")
    print(f"MSE (crop): {mse_crop:.6f}")
    print(f"MSE (expand): {mse_expand:.6f}")

    # Feature visualization
    feat_small_img = feat_small[0].cpu().numpy()
    feat_large_img = feat_large[0].cpu().numpy()

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.heatmap(feat_small_img, cmap='Blues')
    plt.title("Small Subnet Feature")

    plt.subplot(1, 2, 2)
    sns.heatmap(feat_large_img, cmap='Reds')
    plt.title("Large Subnet Feature")

    plt.tight_layout()
    plt.savefig("feature_alignment_vis.png")
    print("[Saved] feature_alignment_vis.png")

    # sim_size = F.cosine_similarity(feat_small.flatten(1), feat_large.flatten(1), dim=1).mean().item()

    # mse_size = F.mse_loss(feat_small, feat_large).item()

    # print("[Feature Alignment Result]")
    # print(f"Cosine Similarity (sim_size): {sim_size:.4f}")
    # print(f"MSE (crop): {mse_size:.6f}")


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

    small_config = {
        'embed_dim': [192] * config['layer_num'],
        'num_heads': [3] * config['layer_num'],
        'mlp_ratio': [3.5] * config['layer_num'],
        'layer_num': config['layer_num']
    }
    large_config = {
        'embed_dim': [240] * config['layer_num'],
        'num_heads': [4] * config['layer_num'],
        'mlp_ratio': [4.0] * config['layer_num'],
        'layer_num': config['layer_num']
    }

    run_feature_alignment(model, data_loader, device, config_small=small_config, config_large=large_config)

    return {}

    # for images, target in metric_logger.log_every(data_loader, 10, header):
    #     images = images.to(device, non_blocking=True)
    #     target = target.to(device, non_blocking=True)
    #     # compute output
    #     if amp:
    #         with torch.cuda.amp.autocast():
    #             output = model(images)
    #             loss = criterion(output, target)
    #     else:
    #         output = model(images)
    #         loss = criterion(output, target)

    #     acc1, acc5 = accuracy(output, target, topk=(1, 5))

    #     batch_size = images.shape[0]
    #     metric_logger.update(loss=loss.item())
    #     metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
    #     metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # # gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    # print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
    #       .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    # return {k: meter.global_avg for k, meter in metric_logger.meters.items()}