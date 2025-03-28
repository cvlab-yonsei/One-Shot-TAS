import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class qkv_super(nn.Linear):
    def __init__(self, super_in_dim, super_out_dim, bias=True, uniform_=None, non_linear='linear', scale=False):
        super().__init__(super_in_dim, super_out_dim, bias=bias)

        # super_in_dim and super_out_dim indicate the largest network!
        self.super_in_dim = super_in_dim
        self.super_out_dim = super_out_dim

        # input_dim and output_dim indicate the current sampled size
        self.sample_in_dim = None
        self.sample_out_dim = None

        # 추가: 이전 sampled size 정보 (freeze를 위한)
        self.sample_in_dim_prev = None
        self.sample_out_dim_prev = None

        self.samples = {}

        self.scale = scale
        # self._reset_parameters(bias, uniform_, non_linear)
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
        # print("_prev None check qkvSuper : ", sample_in_dim_prev, sample_out_dim_prev)
        self.sample_in_dim = sample_in_dim
        self.sample_out_dim = sample_out_dim
        self.sample_in_dim_prev = sample_in_dim_prev
        self.sample_out_dim_prev = sample_out_dim_prev

        self._sample_parameters()

    def _sample_parameters(self):
        self.samples['weight'] = sample_weight(self.weight, self.sample_in_dim, self.sample_out_dim,
                                               self.sample_in_dim_prev, self.sample_out_dim_prev)
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
    weight: (super_out_dim, super_in_dim)
    sample_in_dim, sample_out_dim: 최종적으로 사용할 column, row 크기 (예: 9, 9)
    sample_in_dim_prev, sample_out_dim_prev: freeze할 이전 영역의 크기 (예: 6, 6)
      - 둘 다 None이면 기존 방식대로, 아니라면 좌상단 영역은 freeze(detach) 처리.
    """
    # print("_prev None check qkvSuper_sample_weight : ", sample_in_dim_prev, sample_out_dim_prev)
    # 우선, 열은 sample_in_dim까지 슬라이스
    sw = weight[:, :sample_in_dim]
    
    # 만약 둘 다 None이면 기존 방식대로 처리
    if sample_in_dim_prev is None and sample_out_dim_prev is None:
        sw = torch.cat([sw[i:sample_out_dim:3, :] for i in range(3)], dim=0)
        return sw
    else:
        # 하나라도 None이면 각각 대체
        if sample_in_dim_prev is None:
            sample_in_dim_prev = sample_in_dim
        if sample_out_dim_prev is None:
            sample_out_dim_prev = sample_out_dim

        groups = []
        for i in range(3):
            # 각 그룹: sw의 row를 step=3으로 슬라이스
            group_i = sw[i:sample_out_dim:3, :]  # shape: (n, sample_in_dim)
            # global row 인덱스는 i, i+3, i+6, ... 
            # frozen할 행의 개수: global row index < sample_out_dim_prev
            if sample_out_dim_prev > i:
                k_max = (sample_out_dim_prev - i + 3 - 1) // 3
            else:
                k_max = 0
            k_max = min(k_max, group_i.size(0))  # group_i의 행 수를 초과하지 않도록
            
            # 첫 k_max 행은 frozen 영역으로 처리:  
            # 이들 행 중, 왼쪽 sample_in_dim_prev 열은 freeze(detach), 나머지 열은 그대로.
            if k_max > 0:
                frozen_rows = group_i[:k_max, :]  # frozen row 전체
                frozen_part = frozen_rows[:, :sample_in_dim_prev].detach()  # 좌측 freeze
                trainable_part_in_frozen = frozen_rows[:, sample_in_dim_prev:]  # 우측은 학습 가능
                processed_frozen = torch.cat([frozen_part, trainable_part_in_frozen], dim=1)
            else:
                processed_frozen = torch.empty(0, group_i.size(1), device=group_i.device)
            
            # 나머지 행은 모두 학습 가능
            non_frozen = group_i[k_max:, :]
            group_processed = torch.cat([processed_frozen, non_frozen], dim=0)
            groups.append(group_processed)
        return torch.cat(groups, dim=0)


def sample_bias(bias, sample_out_dim, sample_out_dim_prev=None):
    # print("_prev None check qkvSuper_sample_bias : ", sample_out_dim_prev)
    sample_bias = bias[:sample_out_dim]
    if sample_out_dim_prev is None:
        return sample_bias
    else:
        frozen = bias[:sample_out_dim_prev].detach()
        trainable = bias[sample_out_dim_prev:sample_out_dim]
        return torch.cat([frozen, trainable], dim=0)

