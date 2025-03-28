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

        self.weight.requires_grad = False
        self.bias.requires_grad = False

        self.w1 = nn.Parameter(torch.randn(super_out_dim//2, super_in_dim//2), requires_grad=True)
        self.w2 = nn.Parameter(torch.randn(super_out_dim//2, super_in_dim//4), requires_grad=True)
        # self.w3 = nn.Parameter(torch.randn(0, super_in_dim), requires_grad=True)
        self.w3 = nn.Parameter(torch.randn(super_out_dim//4, (super_in_dim//4)*3), requires_grad=True) # 여긴 일단 parameter 셀때 안보게 False로 둘까
        
        # 무조건 freeze인 나머지 바깥 테두리
        self.w4 = nn.Parameter(torch.randn((super_out_dim//4)*3, super_in_dim//4), requires_grad=True)
        self.w5 = nn.Parameter(torch.randn(super_out_dim//4, super_in_dim), requires_grad=True)

        
        # # Q, K, V 각각에 대해 w1~w5 선언
        # for prefix in ['q', 'k', 'v']:
        #     setattr(self, f"{prefix}_w1", nn.Parameter(torch.randn(super_out_dim//6, super_in_dim//2), requires_grad=False))
        #     setattr(self, f"{prefix}_w2", nn.Parameter(torch.randn(super_out_dim//6, super_in_dim//4), requires_grad=True))
        #     setattr(self, f"{prefix}_w3", nn.Parameter(torch.randn(super_out_dim//12, (super_in_dim//4)*3), requires_grad=True))
        #     setattr(self, f"{prefix}_w4", nn.Parameter(torch.randn(super_out_dim//4, super_in_dim//4), requires_grad=False))
        #     setattr(self, f"{prefix}_w5", nn.Parameter(torch.randn(super_out_dim//12, super_in_dim), requires_grad=False))

        self.bias1 = nn.Parameter(torch.rand(super_out_dim//2), requires_grad=True)
        self.bias2 = nn.Parameter(torch.rand(super_out_dim//4), requires_grad=True)
        self.bias3 = nn.Parameter(torch.rand(super_out_dim//4), requires_grad=True)

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
        self.sample_in_dim = sample_in_dim
        self.sample_out_dim = sample_out_dim
        self.sample_in_dim_prev = sample_in_dim_prev
        self.sample_out_dim_prev = sample_out_dim_prev

        self._sample_parameters()

    def _sample_parameters(self):
        self.samples['weight'] = sample_weight(self, self.weight, self.sample_in_dim, self.sample_out_dim,
                                               self.sample_in_dim_prev, self.sample_out_dim_prev)
        self.samples['bias'] = self.bias
        self.sample_scale = self.super_out_dim/self.sample_out_dim
        if self.bias is not None:
            self.samples['bias'] = sample_bias(self, self.bias, self.sample_out_dim, self.sample_out_dim_prev)
        return self.samples
    
    @property
    def weight(self):
        sw = self.w2
        # 만약 frozen 영역이 없으면, w1와 w3가 빈 텐서여야 합니다.
        if self.w1.shape[0] == 0 and self.w3.shape[0] == 0:
            sw = self.w2
        else:
            top = torch.cat([self.w1, self.w2], dim=1)
            full = torch.cat([top, self.w3], dim=0)
            sw = full

        #sw는 전체 weight에서 sample_in_dim 파트까지만 crop을 이미 한거
        # print("sw.shape : ", sw.shape)
            
        sample_weight = torch.cat([sw[i:self.sample_out_dim:3, :] for i in range(3)], dim =0)
        # print("sample_weight.shape : ", sample_weight.shape)
        return sample_weight
    
    @property
    def bias(self):
        # frozen bias가 없으면, bias1은 빈 텐서.
        if self.bias1.numel() == 0:
            # print("here 1")
            # print("self.bias2.shape : ", self.bias2.shape)
            return self.bias2
        else:
            sample_bias = torch.cat([self.bias1, self.bias2], dim=0)
            # print("here 2")
            # print("sample_bias.shape : ", sample_bias.shape)
            return sample_bias


    def forward(self, x):
        self.sample_parameters()
        # print("self.weight.shape : ", self.weight.shape)
        # print("self.bias.shape : ", self.bias.shape)
        # print("self.samples['weight'].shape : ", self.samples['weight'].shape)
        # print("self.self.samples['bias'].shape : ", self.samples['bias'].shape)
        return F.linear(x, self.weight, self.bias) * (self.sample_scale if self.scale else 1)
        # return F.linear(x, self.samples['weight'], self.samples['bias']) * (self.sample_scale if self.scale else 1)

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

def sample_weight(self, weight, sample_in_dim, sample_out_dim, sample_in_dim_prev=None, sample_out_dim_prev=None):
    """
    weight: (super_out_dim, super_in_dim)
    sample_in_dim, sample_out_dim: 최종적으로 사용할 column, row 크기 (예: 9, 9)
    sample_in_dim_prev, sample_out_dim_prev: freeze할 이전 영역의 크기 (예: 6, 6)
      - 둘 다 None이면 기존 방식대로, 아니라면 좌상단 영역은 freeze(detach) 처리.
    """
    # 우선, 열은 sample_in_dim까지 슬라이스
    sw = weight[:, :sample_in_dim]

    full_weight = self.w2
    full_top = self.w2
    if self.w1.shape[0] == 0 and self.w3.shape[0] == 0:
        full_weight = self.w2
    else:
        full_top = torch.cat([self.w1, self.w2], dim=1)
        full_weight = torch.cat([full_top, self.w3], dim=0)

    # print("full_weight.shape : ", full_weight.shape)
    # print("self.w4.shape : ", self.w4.shape)
    full_top_out = torch.cat([full_weight, self.w4], dim=1)
    full_weight_out = torch.cat([full_top_out, self.w5], dim=0)

    
    # 만약 둘 다 None이면 기존 방식대로 처리
    if sample_in_dim_prev is None and sample_out_dim_prev is None:
        new_w1 = torch.empty(0, sample_in_dim, device=weight.device)
        new_w2 = full_weight_out[:sample_out_dim, :sample_in_dim].clone()
        new_w3 = torch.empty(0, sample_in_dim, device=weight.device)
        new_w4 = full_weight_out[:sample_out_dim, sample_in_dim:self.super_in_dim].clone()
        new_w5 = full_weight_out[sample_out_dim:self.super_out_dim, :self.super_in_dim].clone()

        self.w1 = nn.Parameter(new_w1, requires_grad=False)
        self.w2 = nn.Parameter(new_w2, requires_grad=True)
        self.w3 = nn.Parameter(new_w3, requires_grad=True)
        self.w4 = nn.Parameter(new_w4, requires_grad=False)
        self.w5 = nn.Parameter(new_w5, requires_grad=False)

        return torch.cat([sw[i:sample_out_dim:3, :] for i in range(3)], dim=0)
    else:
        # 하나라도 None이면 각각 대체
        if sample_in_dim_prev is None:
            sample_in_dim_prev = sample_in_dim
        if sample_out_dim_prev is None:
            sample_out_dim_prev = sample_out_dim

        new_w1 = full_weight_out[:sample_out_dim_prev, :sample_in_dim_prev].detach()
        new_w2 = full_weight_out[:sample_out_dim_prev, sample_in_dim_prev:sample_in_dim].clone()
        new_w3 = full_weight_out[sample_out_dim_prev:sample_out_dim, :sample_in_dim].clone()
        new_w4 = full_weight_out[:sample_out_dim, sample_in_dim:self.super_in_dim].detach()
        new_w5 = full_weight_out[sample_out_dim:self.super_out_dim, :self.super_in_dim].detach()

        self.w1 = nn.Parameter(new_w1, requires_grad=False)
        self.w2 = nn.Parameter(new_w2, requires_grad=True)
        self.w3 = nn.Parameter(new_w3, requires_grad=True)
        self.w4 = nn.Parameter(new_w4, requires_grad=False)
        self.w5 = nn.Parameter(new_w5, requires_grad=False)

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


def sample_bias(self, bias, sample_out_dim, sample_out_dim_prev=None):
    full_bias = torch.cat([self.bias1, self.bias2], dim=0)
    full_bias_out = torch.cat([full_bias, self.bias3], dim=0)
    new_bias1 = torch.empty(0, device=bias.device)
    new_bias2 = full_bias_out[:sample_out_dim]
    new_bias3 = full_bias_out[sample_out_dim:self.super_out_dim]
    self.bias1 = nn.Parameter(new_bias1, requires_grad=False)
    self.bias2 = nn.Parameter(new_bias2, requires_grad=True)
    self.bias3 = nn.Parameter(new_bias3, requires_grad=False)

    if sample_out_dim_prev is None:
        return bias[:sample_out_dim]
    else:
        new_bias1 = full_bias_out[:sample_out_dim_prev].detach()
        new_bias2 = full_bias_out[sample_out_dim_prev:sample_out_dim]
        new_bias3 = full_bias_out[sample_out_dim:self.super_out_dim].detach()

        self.bias1 = nn.Parameter(new_bias1, requires_grad=False)
        self.bias2 = nn.Parameter(new_bias2, requires_grad=True)
        self.bias3 = nn.Parameter(new_bias3, requires_grad=False)

        frozen = bias[:sample_out_dim_prev].detach()
        trainable = bias[sample_out_dim_prev:sample_out_dim]
        return torch.cat([frozen, trainable], dim=0)

