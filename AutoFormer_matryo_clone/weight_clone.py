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


def init_model_matryo_from_model(model, model_matryo):
    # 1. 우선 이름이 정확히 매칭되는 것부터 모두 복사
    with torch.no_grad():
        model_dict = dict(model.named_parameters())
        matryo_dict = dict(model_matryo.named_parameters())

        for name, param in matryo_dict.items():
            if name in model_dict:
                param.copy_(model_dict[name].data.clone())

        model_dict = dict(model.named_parameters())
        matryo_modules = dict(model_matryo.named_modules())

        # --- split_weights + split_bias + split_biases ---
        for name, module in matryo_modules.items():
            # --- split_weights 처리 ---
            if hasattr(module, 'split_weights'):
                for split_name, split_param in module.split_weights.items():
                    if not isinstance(split_param, nn.Parameter):
                        continue
                    if 'w' not in split_name:
                        continue

                    base_name = name + '.weight'
                    if base_name not in model_dict:
                        print(f"[⚠] {base_name} not found in model_dict")
                        continue
                    full_weight = model_dict[base_name]

                    if '_' in split_name:  # ex: w2_1
                        i, j = map(int, split_name.replace('w', '').split('_'))
                        dim0 = sum(module.split_weights[f'w{k}_{j}'].shape[0] for k in range(1, i))
                        dim1 = sum(module.split_weights[f'w{i}_{k}'].shape[1] for k in range(1, j))
                        h, w = split_param.shape
                        cropped = full_weight[dim0:dim0+h, dim1:dim1+w]
                    # else:  # ex: w1
                    #     h, w = split_param.shape
                    #     cropped = full_weight[:h, :w]
                    elif split_name == 'w1':
                        if full_weight.dim() == 4:
                            # [out_c, in_c, k_h, k_w]
                            oc, ic, kh, kw = split_param.shape
                            cropped = full_weight[:oc, :ic, :kh, :kw]
                        elif full_weight.dim() == 2:
                            h, w = split_param.shape
                            cropped = full_weight[:h, :w]
                        else:
                            raise ValueError(f"Unsupported weight shape: {full_weight.shape}")
                        split_param.copy_(cropped)

                    else:
                        print(f"[⚠] Unknown split_name format: {split_name}")
                        continue

                    split_param.copy_(cropped)

            # --- split_bias 처리 ---
            if hasattr(module, 'split_bias'):
                for bias_name, bias_param in module.split_bias.items():
                    if not isinstance(bias_param, nn.Parameter):
                        continue

                    base_name = name + '.bias'
                    if base_name not in model_dict:
                        print(f"[⚠] {base_name} not found in model_dict")
                        continue
                    full_bias = model_dict[base_name]

                    if 'bias_' in bias_name:
                        i = int(bias_name.replace('bias_', ''))
                        offset = sum(module.split_bias[f'bias_{k}'].shape[0] for k in range(1, i))
                        cropped = full_bias[offset:offset + bias_param.shape[0]]
                        bias_param.copy_(cropped)
                    elif bias_name == 'bias':
                        bias_param.copy_(full_bias[:bias_param.shape[0]])

            # --- split_biases 처리 (복수형: b1, b2, ...) ---
            if hasattr(module, 'split_biases'):
                for bias_name, bias_param in module.split_biases.items():
                    if not isinstance(bias_param, nn.Parameter):
                        continue

                    # b1일 때만 처리
                    if bias_name != 'b1':
                        continue

                    base_name = name + '.bias'
                    if base_name not in model_dict:
                        print(f"[⚠] {base_name} not found in model_dict")
                        continue

                    full_bias = model_dict[base_name]
                    # print("full_bias shape : ", full_bias.shape)
                    # print("bias_param shape : ", bias_param.shape)
                    h = bias_param.shape[0]
                    cropped = full_bias[:h]
                    bias_param.copy_(cropped)

            # --- split_embeddings_v / h 처리 (w1만) ---
            if hasattr(module, 'split_embeddings_v') and hasattr(module, 'split_embeddings_h'):
                # vertical
                if 'w1' in module.split_embeddings_v:
                    split_param = module.split_embeddings_v['w1']
                    base_name = name + '.embeddings_table_v'
                    if base_name not in model_dict:
                        print(f"[⚠] {base_name} not found in model_dict")
                    else:
                        full_weight = model_dict[base_name]
                        h, w = split_param.shape
                        cropped = full_weight[:h, :w]
                        split_param.copy_(cropped)

                # horizontal
                if 'w1' in module.split_embeddings_h:
                    split_param = module.split_embeddings_h['w1']
                    base_name = name + '.embeddings_table_h'
                    if base_name not in model_dict:
                        print(f"[⚠] {base_name} not found in model_dict")
                    else:
                        full_weight = model_dict[base_name]
                        h, w = split_param.shape
                        cropped = full_weight[:h, :w]
                        split_param.copy_(cropped)


        # --- fallback copy: 이름과 shape 일치하는 경우 그냥 복사 ---
        model_dict = dict(model.named_parameters())
        matryo_dict = dict(model_matryo.named_parameters())
        for name, param in matryo_dict.items():
            if name in model_dict and param.shape == model_dict[name].shape:
                param.copy_(model_dict[name].clone())

        # # 2. 이후 w1 관련된 파라미터만 crop해서 다시 복사
        # for name, param in matryo_dict.items():
        #     if 'w1' in name:
        #         # 예시: name = 'blocks.0.attn.qkv.w1'
        #         # 대응되는 전체 weight 이름은 'blocks.0.attn.qkv.weight' 일 가능성 높음
        #         target_name = name.replace('w1', 'weight')
        #         if target_name in model_dict:
        #             full_weight = model_dict[target_name].data
        #             crop_shape = param.shape
        #             cropped = full_weight[:crop_shape[0], :crop_shape[1]] if full_weight.dim() == 2 else full_weight[:crop_shape[0]]
        #             param.copy_(cropped.clone())
        #         else:
        #             print(f"⚠️ Cannot find matching weight for {name} → tried {target_name}")
