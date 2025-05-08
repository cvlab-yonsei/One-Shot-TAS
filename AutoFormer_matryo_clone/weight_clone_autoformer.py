import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import yaml
from pathlib import Path
from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler
from lib.datasets import build_dataset
from supernet_engine import train_one_epoch, evaluate
from lib.samplers import RASampler
from lib import utils
from lib.config import cfg, update_config_from_file
from model.supernet_transformer import Vision_TransformerSuper
from model_matryo.supernet_transformer import Vision_TransformerSuper as Vision_TransformerSuper_Matryo
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

def copy_case2_parameters(model, model_matryo):
    """
    model의 파라미터 중에서, model_matryo의 split_weight/split_bias 구조에서
    case_num == 2 로 requires_grad 되어 있는 파라미터들만 복사해주는 함수
    """
    from torch.nn import Parameter

    model_dict = dict(model.named_parameters())
    matryo_modules = dict(model_matryo.named_modules())  # DistributedDataParallel 처리

    with torch.no_grad():
        for name, module in matryo_modules.items():

            # --- 일반 weight 복사 (split_weights[w1], w_1, w1_1) ---
            if hasattr(module, 'split_weights') and isinstance(module.split_weights, nn.ParameterDict):
                weight_name = name + '.weight'
                if weight_name not in model_dict:
                    continue
                full_weight = model_dict[weight_name].data

                offset = 0
                for key, split_param in module.split_weights.items():
                    if not split_param.requires_grad:
                        offset += split_param.shape[0] if full_weight.dim() == 4 else 0
                        continue

                    if key.startswith('w') and '_' not in key:  # w1 등 (Conv2d)
                        if full_weight.dim() == 4:
                            oc, ic, kh, kw = split_param.shape
                            cropped = full_weight[offset:offset+oc, :ic, :kh, :kw]
                            offset += oc
                        elif full_weight.dim() == 2:
                            h, w = split_param.shape
                            cropped = full_weight[offset:offset+h, :w]
                            offset += h
                        else:
                            continue
                        split_param.copy_(cropped)

                    elif '_' in key and key.startswith('w_'):  # w_1 등 (LayerNorm 등)
                        i = int(key.replace('w_', ''))
                        offset = sum(module.split_weights[f'w_{k}'].shape[0] for k in range(1, i))
                        cropped = full_weight[offset:offset + split_param.shape[0]]
                        split_param.copy_(cropped)

                    elif '_' in key and key.startswith('w'):  # w1_1 등 (Linear 계층)
                        i, j = map(int, key.replace('w', '').split('_'))
                        dim0 = sum(module.split_weights[f'w{k}_{j}'].shape[0] for k in range(1, i))
                        dim1 = sum(module.split_weights[f'w{i}_{k}'].shape[1] for k in range(1, j))
                        h, w = split_param.shape
                        cropped = full_weight[dim0:dim0+h, dim1:dim1+w]
                        split_param.copy_(cropped)

            # --- split_bias 처리: bias_1 등 ---
            if hasattr(module, 'split_bias') and isinstance(module.split_bias, nn.ParameterDict):
                bias_name = name + '.bias'
                if bias_name not in model_dict:
                    continue
                full_bias = model_dict[bias_name].data

                offset = 0
                for i in range(len(module.split_bias)):
                    key = f'bias_{i+1}'
                    if key not in module.split_bias:
                        continue
                    split_param = module.split_bias[key]
                    length = split_param.shape[0]
                    if split_param.requires_grad:
                        block = full_bias[offset:offset+length]
                        split_param.copy_(block)
                    offset += length

            # --- split_biases 처리: b1, b2, ... (Conv2d) ---
            if hasattr(module, 'split_biases') and isinstance(module.split_biases, nn.ParameterDict):
                bias_name = name + '.bias'
                if bias_name not in model_dict:
                    continue
                full_bias = model_dict[bias_name].data

                offset = 0
                for key, split_param in module.split_biases.items():
                    if not split_param.requires_grad:
                        offset += split_param.shape[0]
                        continue
                    length = split_param.shape[0]
                    cropped = full_bias[offset:offset+length]
                    split_param.copy_(cropped)
                    offset += length

            # --- split_embeddings_v/h 처리 ---
            for embed_dir in ['split_embeddings_v', 'split_embeddings_h']:
                if hasattr(module, embed_dir):
                    embed_dict = getattr(module, embed_dir)
                    if not isinstance(embed_dict, nn.ParameterDict):
                        continue
                    table_key = 'embeddings_table_v' if 'v' in embed_dir else 'embeddings_table_h'
                    table_name = name + f'.{table_key}'
                    if table_name not in model_dict:
                        continue
                    full_embed = model_dict[table_name].data

                    offset = 0
                    for i in range(len(embed_dict)):
                        key = f'w{i+1}'
                        if key not in embed_dict:
                            continue
                        split_param = embed_dict[key]
                        h, w = split_param.shape
                        if split_param.requires_grad:
                            cropped = full_embed[offset:offset+h, :w]
                            split_param.copy_(cropped)
                        offset += h
