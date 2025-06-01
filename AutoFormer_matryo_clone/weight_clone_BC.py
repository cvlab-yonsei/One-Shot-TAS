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

def copy_trainable_parameters(model, model_matryo):
    """
    model_matryo의 파라미터 중 requires_grad=True인 것들만,
    동일한 이름의 model 파라미터 값으로 덮어씀.
    """
    model_params = dict(model.named_parameters())
    matryo_params = dict(model_matryo.named_parameters())

    with torch.no_grad():
        for name, matryo_param in matryo_params.items():
            if name not in model_params:
                continue
            if matryo_param.requires_grad:
                matryo_param.copy_(model_params[name].data)
