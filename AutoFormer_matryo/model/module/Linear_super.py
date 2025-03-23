import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LinearSuper(nn.Linear):
    def __init__(self, super_in_dim, super_out_dim, bias=True, uniform_=None, non_linear='linear', scale=False):
        super().__init__(super_in_dim, super_out_dim, bias=bias)

        # super_in_dim and super_out_dim indicate the largest network!
        self.super_in_dim = super_in_dim
        self.super_out_dim = super_out_dim

        # input_dim and output_dim indicate the current sampled size
        self.sample_in_dim = None
        self.sample_out_dim = None
        self.sample_in_dim_prev = None
        self.sample_out_dim_prev = None

        self.samples = {}
        self.weight_frozen = None  # freeze할 weight (왼쪽 위)
        self.weight_trainable_top_right = None  # 학습 가능한 weight (오른쪽 위)
        self.weight_trainable_bottom = None  # 학습 가능한 weight (아래쪽 전체)

        self.scale = scale
        self._reset_parameters(bias, uniform_, non_linear)
        self.profiling = False

    def profile(self, mode=True):
        self.profiling = mode

    def sample_parameters(self, resample=False):
        if self.profiling or resample:
            return self._sample_parameters()
        return self.samples

    def _reset_parameters(self, bias, uniform_, non_linear):
        nn.init.xavier_uniform_(self.weight) if uniform_ is None else uniform_(
            self.weight, non_linear=non_linear)
        if bias:
            nn.init.constant_(self.bias, 0.)

    def set_sample_config(self, sample_in_dim, sample_out_dim, sample_in_dim_prev=None, sample_out_dim_prev=None):
        # print("_prev None check LinearSuper : ", sample_in_dim_prev, sample_out_dim_prev)
        self.sample_in_dim = sample_in_dim
        self.sample_out_dim = sample_out_dim
        self.sample_in_dim_prev = sample_in_dim_prev
        self.sample_out_dim_prev = sample_out_dim_prev

        self._sample_parameters()

    def _sample_parameters(self):
        self.samples['weight'] = sample_weight(self.weight, self.sample_in_dim, self.sample_out_dim, self.sample_in_dim_prev, self.sample_out_dim_prev)
        self.samples['bias'] = self.bias
        self.sample_scale = self.super_out_dim/self.sample_out_dim
        if self.bias is not None:
            self.samples['bias'] = sample_bias(self.bias, self.sample_out_dim, self.sample_out_dim_prev)
        return self.samples

    def forward(self, x):
        self.sample_parameters()
        return F.linear(x, self.samples['weight'], self.samples['bias']) * (self.sample_scale if self.scale else 1)

    def calc_sampled_param_num(self):
        assert 'weight' in self.samples.keys()
        weight_numel = self.samples['weight'].numel()

        if self.samples['bias'] is not None:
            bias_numel = self.samples['bias'].numel()
        else:
            bias_numel = 0

        return weight_numel + bias_numel
    def get_complexity(self, sequence_length):
        total_flops = 0
        total_flops += sequence_length *  np.prod(self.samples['weight'].size())
        return total_flops

def sample_weight(weight, sample_in_dim, sample_out_dim, sample_in_dim_prev=None, sample_out_dim_prev=None):
    """
    weight: 원본 weight tensor, shape = (total_out_dim, total_in_dim)
    sample_in_dim: 최종으로 사용할 입력 차원 (columns)
    sample_out_dim: 최종으로 사용할 출력 차원 (rows)
    sample_in_dim_prev: 이전 단계에서 사용된 입력 차원 (freeze할 영역, columns)
    sample_out_dim_prev: 이전 단계에서 사용된 출력 차원 (freeze할 영역, rows)
    
    반환:
        최종적으로 샘플링된 weight tensor.
        - 만약 sample_in_dim_prev와 sample_out_dim_prev가 둘 다 제공되지 않으면,
          단순히 [sample_out_dim, sample_in_dim] 영역을 반환합니다.
        - 하나라도 None이면, 각각 sample_in_dim, sample_out_dim으로 대체하여
          왼쪽 위 영역은 detach()를 통해 frozen 처리되고,
          나머지 영역은 trainable하게 남습니다.
    """
    # print("_prev None check LinearSuper sample_weight: ", sample_in_dim_prev, sample_out_dim_prev)

    # 최종 영역 슬라이싱
    sampled_weight = weight[:, :sample_in_dim]
    sampled_weight = sampled_weight[:sample_out_dim, :]
    
    # 둘 다 None이면 그냥 반환
    if sample_in_dim_prev is None and sample_out_dim_prev is None: # 여기만 다르게
        return sampled_weight
    if sample_in_dim_prev is None and sample_out_dim_prev == 1000: #일단 휴리스틱하게.. (num_class인경우)
        return sampled_weight

    # None인 경우 각각 대체
    if sample_in_dim_prev is None:
        sample_in_dim_prev = sample_in_dim
    if sample_out_dim_prev is None:
        sample_out_dim_prev = sample_out_dim

    # 왼쪽 위 (frozen): rows 0:sample_out_dim_prev, cols 0:sample_in_dim_prev
    frozen = sampled_weight[:sample_out_dim_prev, :sample_in_dim_prev].detach()
    # 오른쪽 위 (trainable): rows 0:sample_out_dim_prev, cols sample_in_dim_prev:sample_in_dim
    top_right = sampled_weight[:sample_out_dim_prev, sample_in_dim_prev:sample_in_dim]
    # 아래쪽 전체 (trainable): rows sample_out_dim_prev:sample_out_dim, 모든 columns
    bottom = sampled_weight[sample_out_dim_prev:sample_out_dim, :]
    
    # 상단 부분 결합 (frozen와 trainable 영역)
    top_combined = torch.cat([frozen, top_right], dim=1)
    # 최종 weight 결합: 상단 + 아래쪽
    sampled_weight = torch.cat([top_combined, bottom], dim=0)
    
    return sampled_weight


def sample_bias(bias, sample_out_dim, sample_out_dim_prev=None):
    """
    bias: 원본 bias tensor, shape = (total_out_dim,)
    sample_out_dim: 최종으로 사용할 출력 차원 (elements)
    sample_out_dim_prev: 이전 단계에서 사용된 출력 차원 (freeze할 영역, elements)
    
    반환:
        최종적으로 샘플링된 bias tensor.
        만약 sample_out_dim_prev가 제공되면, 앞쪽 영역은 detach()를 통해 frozen 처리됩니다.
    """
    # print("_prev None check LinearSuper sample_bias : ", sample_out_dim_prev)
    sampled_bias = bias[:sample_out_dim]
    if sample_out_dim_prev is not None:
        frozen = sampled_bias[:sample_out_dim_prev].detach()
        trainable = sampled_bias[sample_out_dim_prev:sample_out_dim]
        sampled_bias = torch.cat([frozen, trainable], dim=0)
    return sampled_bias


