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


def zero_out_frozen_parameters(model_matryo):
    """
    model_matryo 내부에서 requires_grad=False로 설정된 split_* 파라미터들
    (weight, bias, embedding 등)을 0으로 채움
    """

    matryo_modules = dict(model_matryo.named_modules())  # DDP 대응

    with torch.no_grad():
        for name, module in matryo_modules.items():
            # --- split_weights ---
            if hasattr(module, 'split_weights') and isinstance(module.split_weights, nn.ParameterDict):
                for key, split_param in module.split_weights.items():
                    if not split_param.requires_grad:
                        split_param.zero_()

            # --- split_bias ---
            if hasattr(module, 'split_bias') and isinstance(module.split_bias, nn.ParameterDict):
                for key, split_param in module.split_bias.items():
                    if not split_param.requires_grad:
                        split_param.zero_()

            # --- split_biases ---
            if hasattr(module, 'split_biases') and isinstance(module.split_biases, nn.ParameterDict):
                for key, split_param in module.split_biases.items():
                    if not split_param.requires_grad:
                        split_param.zero_()

            # --- split_embeddings_v/h ---
            for embed_dir in ['split_embeddings_v', 'split_embeddings_h']:
                if hasattr(module, embed_dir):
                    embed_dict = getattr(module, embed_dir)
                    if isinstance(embed_dict, nn.ParameterDict):
                        for key, split_param in embed_dict.items():
                            if not split_param.requires_grad:
                                split_param.zero_()
