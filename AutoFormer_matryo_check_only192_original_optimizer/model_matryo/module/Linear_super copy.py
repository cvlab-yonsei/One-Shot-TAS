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

        self.weight.requires_grad = False
        self.bias.requires_grad = False

        self.w1 = nn.Parameter(torch.randn(super_out_dim//2, super_in_dim//2), requires_grad=True)
        self.w2 = nn.Parameter(torch.randn(super_out_dim//2, super_in_dim//4), requires_grad=True)
        # self.w3 = nn.Parameter(torch.randn(0, super_in_dim), requires_grad=True)
        self.w3 = nn.Parameter(torch.randn(super_out_dim//4, (super_in_dim//4)*3), requires_grad=True) # 여긴 일단 parameter 셀때 안보게 False로 둘까
        
        # 무조건 freeze인 나머지 바깥 테두리
        self.w4 = nn.Parameter(torch.randn((super_out_dim//4)*3, super_in_dim//4), requires_grad=True)
        self.w5 = nn.Parameter(torch.randn(super_out_dim//4, super_in_dim), requires_grad=True)

        self.bias1 = nn.Parameter(torch.rand(0), requires_grad=True)
        self.bias2 = nn.Parameter(torch.rand(super_out_dim), requires_grad=True)

        # 무조건 freeze인 나머지 바깥 테두리
        self.bias3 = nn.Parameter(torch.rand(0), requires_grad=True)

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
        self.samples['weight'] = sample_weight(self, self.weight, self.sample_in_dim, self.sample_out_dim, self.sample_in_dim_prev, self.sample_out_dim_prev)
        self.samples['bias'] = self.bias
        self.sample_scale = self.super_out_dim/self.sample_out_dim
        if self.bias is not None:
            self.samples['bias'] = sample_bias(self, self.bias, self.sample_out_dim, self.sample_out_dim_prev)
        return self.samples
    
    def full_weight(self):
        # 만약 frozen 영역이 없으면, w1와 w3가 빈 텐서여야 합니다.
        if self.w1.shape[0] == 0 and self.w3.shape[0] == 0:
            return self.w2
        else:
            # print("self.w1.shape : ", self.w1.shape)
            # print("self.w2.shape : ", self.w2.shape)
            # print("self.w3.shape : ", self.w3.shape)
            top = torch.cat([self.w1, self.w2], dim=1)
            full = torch.cat([top, self.w3], dim=0)
            return full
    
    def full_bias(self):
        # frozen bias가 없으면, bias1은 빈 텐서.
        if self.bias1.numel() == 0:
            return self.bias2
        else:
            return torch.cat([self.bias1, self.bias2], dim=0)

    def forward(self, x):
        self.sample_parameters()
        weight = self.full_weight()
        bias = self.full_bias()
        self.weight = weight
        self.bias = nn.Parameter(bias, requires_grad = False)
        # print("self.super_in_dim : ", self.super_in_dim)
        # print("self.super_out_dim : ", self.super_out_dim)
        # print("self.sample_in_dim : ", self.sample_in_dim)
        # print("self.sample_out_dim : ", self.sample_out_dim)
        # print("self.sample_in_dim_prev : ", self.sample_in_dim_prev)
        # print("self.sample_out_dim_prev : ", self.sample_out_dim_prev)
        # print("self.w1.shape : ", self.w1.shape) # (0, 256)
        # print("self.w2.shape : ", self.w2.shape) # (216, 192)
        # print("self.w3.shape : ", self.w3.shape) # (0, 256)
        # print("self.w4.shape : ", self.w4.shape) # (216, 192)
        # print("self.w5.shape : ", self.w5.shape) # (0, 256)
        # print("self.weight.shape : ", self.weight.shape)
        # print("self.samples['weight'].shape : ", self.samples['weight'].shape)
        # print("self.bias.shape : ", self.bias.shape)
        # print("self.sample_scale : ", self.sample_scale)
        # print("x.shape : ", x.shape)
        return F.linear(x, weight, bias) * (self.sample_scale if self.scale else 1)
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
    
    full_weight = self.w2
    full_top = self.w2
    if self.w1.shape[0] == 0 and self.w3.shape[0] == 0:
        full_weight = self.w2
    else:
        full_top = torch.cat([self.w1, self.w2], dim=1)
        full_weight = torch.cat([full_top, self.w3], dim=0)

    full_top_out = torch.cat([full_weight, self.w4], dim=1)
    full_weight_out = torch.cat([full_top_out, self.w5], dim=0)

    
    # =====================================
    # 최종 영역 슬라이싱
    sampled_weight = weight[:, :sample_in_dim]
    sampled_weight = sampled_weight[:sample_out_dim, :]


    # 둘 다 None이면 그냥 반환
    #일단 휴리스틱하게.. (num_class인경우)
    if (sample_in_dim_prev is None and sample_out_dim_prev is None) or (sample_in_dim_prev is None and sample_out_dim_prev == 1000): # 여기만 다르게
        new_w1 = torch.empty(0, sample_in_dim, device=weight.device)
        new_w2 = full_weight_out[:sample_out_dim, :sample_in_dim].clone()
        new_w3 = torch.empty(0, sample_in_dim, device=weight.device)
        new_w4 = full_weight_out[:sample_out_dim, sample_in_dim:self.super_in_dim].detach()
        new_w5 = full_weight_out[sample_out_dim:self.super_out_dim, :self.super_in_dim].detach()

        self.w1 = nn.Parameter(new_w1, requires_grad=False)
        self.w2 = nn.Parameter(new_w2, requires_grad=True)
        self.w3 = nn.Parameter(new_w3, requires_grad=True)
        self.w4 = nn.Parameter(new_w4, requires_grad=False)
        self.w5 = nn.Parameter(new_w5, requires_grad=False)

        return sampled_weight

    # None인 경우 각각 대체
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
    
    return sampled_weight


def sample_bias(self, bias, sample_out_dim, sample_out_dim_prev=None):
    """
    bias: 원본 bias tensor, shape = (total_out_dim,)
    sample_out_dim: 최종으로 사용할 출력 차원 (elements)
    sample_out_dim_prev: 이전 단계에서 사용된 출력 차원 (freeze할 영역, elements)
    
    반환:
        최종적으로 샘플링된 bias tensor.
        만약 sample_out_dim_prev가 제공되면, 앞쪽 영역은 detach()를 통해 frozen 처리됩니다.
    """
    
    full_bias = torch.cat([self.bias1, self.bias2], dim=0)
    full_bias_out = torch.cat([full_bias, self.bias3], dim=0)
    new_bias1 = torch.empty(0, device=bias.device)
    new_bias2 = full_bias_out[:sample_out_dim].clone()
    new_bias3 = full_bias_out[sample_out_dim:self.super_out_dim].detach()
    self.bias1 = nn.Parameter(new_bias1, requires_grad=False)
    self.bias2 = nn.Parameter(new_bias2, requires_grad=True)
    self.bias3 = nn.Parameter(new_bias3, requires_grad=False)


    if sample_out_dim_prev is not None:
        new_bias1 = full_bias_out[:sample_out_dim_prev].detach()
        new_bias2 = full_bias_out[sample_out_dim_prev:sample_out_dim].clone()
        new_bias3 = full_bias_out[sample_out_dim:self.super_out_dim].detach()

        self.bias1 = nn.Parameter(new_bias1, requires_grad=False)
        self.bias2 = nn.Parameter(new_bias2, requires_grad=True)
        self.bias3 = nn.Parameter(new_bias3, requires_grad=False)

    ######## 
     
    sampled_bias = bias[:sample_out_dim]
      
    return sampled_bias
