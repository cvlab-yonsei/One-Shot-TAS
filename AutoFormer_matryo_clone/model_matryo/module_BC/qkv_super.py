import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models.layers import trunc_normal_
import torch.nn.init as init

def init_split_parameters_with_gaussian(param_dict: nn.ParameterDict):
    """
    param_dict: nn.ParameterDict (e.g., split_weights or split_bias)
    modifies in-place the parameters with requires_grad=True using the mean/std of the first param
    """
    if not param_dict:
        return
    
    # 기준 파라미터: Dict의 첫 번째 entry
    first_key = next(iter(param_dict))
    reference_tensor = param_dict[first_key].detach()
    ref_mean = reference_tensor.mean().item()
    ref_std = reference_tensor.std(unbiased=False).item() + 1e-8  # std 0 방지

    for key, param in param_dict.items():
        if param.requires_grad:
            with torch.no_grad():
                init.normal_(param, mean=ref_mean, std=ref_std)


class qkv_super(nn.Linear):
    def __init__(self, super_in_dim, super_out_dim, bias=True, uniform_=None, non_linear='linear', scale=False, choices=None):
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

        self.weight.requires_grad = False
        self.bias.requires_grad = False

        self.samples = {}

        self.scale = scale
        # self._reset_parameters(bias, uniform_, non_linear)
        self.profiling = False

        # Split ranges
        embed_dims = sorted(set(choices['embed_dim']))
        self.dim0_splits = [3 * e for e in embed_dims] + [self.super_out_dim]
        self.dim1_splits = embed_dims + [self.super_in_dim]

        # Split weight and bias into ParameterDict
        self.split_weights = nn.ParameterDict()
        for i in range(len(self.dim0_splits)):
            for j in range(len(self.dim1_splits)):
                start_dim0 = 0 if i == 0 else self.dim0_splits[i - 1]
                end_dim0 = self.dim0_splits[i]
                start_dim1 = 0 if j == 0 else self.dim1_splits[j - 1]
                end_dim1 = self.dim1_splits[j]
                shape = (end_dim0 - start_dim0, end_dim1 - start_dim1)
                param_name = f'w{i+1}_{j+1}'
                param = nn.Parameter(torch.empty(shape))
                trunc_normal_(param, std=0.02)
                self.split_weights[param_name] = param

        self.split_bias = nn.ParameterDict()
        for i in range(len(self.dim0_splits)):
            start_dim0 = 0 if i == 0 else self.dim0_splits[i - 1]
            end_dim0 = self.dim0_splits[i]
            shape = (end_dim0 - start_dim0,)
            param_name = f'bias_{i+1}'
            param = nn.Parameter(torch.zeros(shape))
            self.split_bias[param_name] = param

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

    def set_sample_config(self, sample_in_dim, sample_out_dim, sample_in_dim_prev=None, sample_out_dim_prev=None, case_num=None):
        self.sample_in_dim = sample_in_dim
        self.sample_out_dim = sample_out_dim
        self.sample_in_dim_prev = sample_in_dim_prev
        self.sample_out_dim_prev = sample_out_dim_prev
        self.case_num=case_num

        self._sample_parameters()

    def _sample_parameters(self):
        sample_w = sample_weight(self.split_weights, self.sample_in_dim, self.sample_out_dim,
                                 self.sample_in_dim_prev, self.sample_out_dim_prev,
                                 self.dim0_splits, self.dim1_splits, self.case_num)

        sample_b = sample_bias(self.split_bias, self.sample_out_dim, self.sample_out_dim_prev, self.dim0_splits, self.case_num)
        self.samples['weight'] = sample_w
        self.samples['bias'] = sample_b
        self.sample_scale = self.super_out_dim/self.sample_out_dim

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

def sample_weight(split_weights, sample_in_dim, sample_out_dim, sample_in_dim_prev, sample_out_dim_prev, dim0_splits, dim1_splits, case_num=None):
    i_end = next(i for i, val in enumerate(dim0_splits) if val >= sample_out_dim)
    j_end = next(j for j, val in enumerate(dim1_splits) if val >= sample_in_dim)

    # for key in split_weights:
    #     split_weights[key].requires_grad = False
    # key = f'w{i_end+1}_{j_end+1}'
    # split_weights[key].requires_grad = True
    
    if case_num is not None:
        if case_num == 1:
            true_label = [(1, 1)]
        elif case_num == 2:
            # true_label = [(1, 2), (2, 1), (2, 2)]
            true_label = [(1, 2), (2, 1), (2, 2),
                          (1, 3), (2, 3), (3, 3),
                (3, 1), (3, 2)]
        elif case_num == 3:
            true_label = [
                (1, 3), (2, 3), (3, 3),
                (3, 1), (3, 2)
            ]

        for i in range(len(dim0_splits)):
            for j in range(len(dim1_splits)):
                key = f'w{i+1}_{j+1}'
                split_weights[key].requires_grad = ((i + 1, j + 1) in true_label)

        # if case_num is not None:
        #     init_split_parameters_with_gaussian(split_weights)

    row_blocks = []
    for i in range(len(dim0_splits)):
        col_blocks = []
        for j in range(len(dim1_splits)):
            key = f'w{i+1}_{j+1}'
            col_blocks.append(split_weights[key])
        row_blocks.append(torch.cat(col_blocks, dim=1))
    full_weight = torch.cat(row_blocks, dim=0)

    # print(f"\n[🔍 qkv_super - Weight requires_grad status]")
    # for key in split_weights:
    #     print(f"  {key:10s} -> {split_weights[key].requires_grad}")

    sample_weight = torch.cat([full_weight[i:sample_out_dim:3, :sample_in_dim] for i in range(3)], dim =0)

    return sample_weight


def sample_bias(split_bias, sample_out_dim, sample_out_dim_prev, dim0_splits, case_num=None):
    i_end = next(i for i, val in enumerate(dim0_splits) if val >= sample_out_dim)

    # for key in split_bias:
    #     split_bias[key].requires_grad = False
    # key = f'bias_{i_end+1}'
    # split_bias[key].requires_grad = True
    if case_num is not None:
        if case_num == 1:
            true_label = [(1)]
        elif case_num == 2:
            # true_label = [(2)]
            true_label = [(2), (3)]
        elif case_num == 3:
            true_label = [(3)]

        for i in range(len(dim0_splits)):
            split_bias[f'bias_{i+1}'].requires_grad = ((i + 1) in true_label)
            # split_bias[f'bias_{i+1}'].requires_grad = True

        # if case_num is not None:
        #     init_split_parameters_with_gaussian(split_bias)


    full_bias = torch.cat([split_bias[f'bias_{i+1}'] for i in range(len(dim0_splits))], dim=0)
    
    # print(f"\n[🔍 qkv_super - Bias requires_grad status]")
    # for key in split_bias:
    #     print(f"  {key:10s} -> {split_bias[key].requires_grad}")
    
    sample_bias = full_bias[:sample_out_dim]
    return sample_bias

