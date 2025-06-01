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

# def copy_case2_parameters_matryo(model_super, model_matryo):
#     """
#     model_matryo의 requires_grad=True인 파라미터만,
#     model_super로부터 이름 기준으로 복사함.
#     두 모델은 동일한 구조여야 함.
#     """
#     super_params = dict(model_super.named_parameters())
#     matryo_params = dict(model_matryo.named_parameters())

#     with torch.no_grad():
#         for name, target_param in matryo_params.items():
#             if not target_param.requires_grad:
#                 continue
#             if name in super_params:
#                 target_param.copy_(super_params[name].data)

def copy_case2_parameters_matryo(model_super, model_matryo):
    """
    model_matryo의 requires_grad=True인 파라미터 중,
    일부 예외를 제외하고 model_super로부터 이름 기준 복사.

    예외:
    - split_embeddings_v/h
    - head 모듈의 split_bias['bias']
    """

    super_params = dict(model_super.named_parameters())
    matryo_params = dict(model_matryo.named_parameters())

    with torch.no_grad():
        for name, target_param in matryo_params.items():
            if not target_param.requires_grad:
                continue

            # 1. split_embeddings_v 또는 split_embeddings_h는 복사 제외
            if any(key in name for key in ['split_embeddings_v', 'split_embeddings_h']):
                continue

            # 2. head 모듈의 split_bias['bias']는 복사 제외
            if 'head' in name and 'split_bias.bias' in name:
                continue

            # 복사 수행
            if name in super_params:
                target_param.copy_(super_params[name].data)
