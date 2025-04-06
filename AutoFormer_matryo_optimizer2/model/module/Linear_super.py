import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from timm.models.layers import trunc_normal_


class LinearSuper(nn.Linear):
    def __init__(self, super_in_dim, super_out_dim, bias=True, uniform_=None, non_linear='linear', scale=False, choices=None, name=None):
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

        self.name = name
        self.choices = choices

        if choices is not None and name is not None:
            if name == 'fc1' or name == 'fc2':
                embed_dims = sorted(set(choices['embed_dim']))
                mlp_ratios = sorted(set(choices['mlp_ratio']))
                
                # dim 0 기준 분할: embed_dim 기반
                dim0_splits = embed_dims + [self.super_in_dim if name == 'fc1' else self.super_out_dim]  # 예: [192, 216, 240, super_out_dim]
                
                # dim 1 기준 분할: 모든 mlp_ratio * embed_dim 조합 (중복 제거 후 정렬)
                dim1_sizes = set()
                for e in embed_dims:
                    for r in mlp_ratios:
                        dim1_sizes.add(int(e * r))
                dim1_splits = sorted(dim1_sizes) + [self.super_out_dim if name == 'fc1' else self.super_in_dim]

                self.split_weights = nn.ParameterDict()

                if name == 'fc1':
                    dim1_splits_tmp = dim1_splits
                    dim1_splits = dim0_splits
                    dim0_splits = dim1_splits_tmp
                    # print("dim0_splits : ", dim0_splits)
                    # print("dim1_splits : ", dim1_splits)
                    for i in range(len(dim0_splits)):
                        for j in range(len(dim1_splits)):
                            start_dim0 = 0 if i == 0 else dim0_splits[i - 1]
                            end_dim0 = dim0_splits[i]
                            start_dim1 = 0 if j == 0 else dim1_splits[j - 1]
                            end_dim1 = dim1_splits[j]
                            shape = (end_dim0 - start_dim0, end_dim1 - start_dim1) 
                            param_name = f'w{i+1}_{j+1}'
                            # self.split_weights[param_name] = nn.Parameter(torch.zeros(shape))
                            self.split_weights[param_name] = nn.Parameter(torch.empty(shape))
                            self._init_split_param(self.split_weights[param_name])  # 초기화

                    # print(f"Split fc1 into {len(dim0_splits)}x{len(dim1_splits)} nn.Parameters.")

                    self.split_bias = nn.ParameterDict()
                    for i in range(len(dim0_splits)):
                        start = dim0_splits[i - 1] if i > 0 else 0
                        end = dim0_splits[i]
                        key = f'bias_{i+1}'
                        # self.split_bias[key] = nn.Parameter(torch.zeros(end - start))
                        self.split_bias[key] = nn.Parameter(torch.empty(end - start))
                        self._init_split_param(self.split_bias[key], is_bias=True)

                else:
                    for i in range(len(dim0_splits)):
                        for j in range(len(dim1_splits)):
                            start_dim0 = 0 if i == 0 else dim0_splits[i - 1]
                            end_dim0 = dim0_splits[i]
                            start_dim1 = 0 if j == 0 else dim1_splits[j - 1]
                            end_dim1 = dim1_splits[j]
                            shape = (end_dim0 - start_dim0, end_dim1 - start_dim1)
                            param_name = f'w{i+1}_{j+1}'
                            # self.split_weights[param_name] = nn.Parameter(torch.zeros(shape))
                            self.split_weights[param_name] = nn.Parameter(torch.empty(shape))
                            self._init_split_param(self.split_weights[param_name])  # 초기화

                    # print(f"Split fc2 into {len(dim0_splits)}x{len(dim1_splits)} nn.Parameters.")

                    self.split_bias = nn.ParameterDict()
                    for i in range(len(dim0_splits)):
                        start = dim0_splits[i - 1] if i > 0 else 0
                        end = dim0_splits[i]
                        key = f'bias_{i+1}'
                        # self.split_bias[key] = nn.Parameter(torch.zeros(end - start))
                        self.split_bias[key] = nn.Parameter(torch.empty(end - start))
                        self._init_split_param(self.split_bias[key], is_bias=True)


            elif name == 'head':
                embed_dims = sorted(set(choices['embed_dim']))

                # dim 1 기준: embed_dim 기준으로 쪼개기
                dim1_splits = embed_dims + [self.super_in_dim]  # dim=1은 input 방향

                self.split_weights = nn.ParameterDict()
                for j in range(len(dim1_splits)):
                    start_dim1 = 0 if j == 0 else dim1_splits[j - 1]
                    end_dim1 = dim1_splits[j]
                    shape = (self.super_out_dim, end_dim1 - start_dim1)  # dim=0은 전체 사용
                    param_name = f'w1_{j+1}'
                    # self.split_weights[param_name] = nn.Parameter(torch.zeros(shape))
                    self.split_weights[param_name] = nn.Parameter(torch.empty(shape))
                    self._init_split_param(self.split_weights[param_name])  # 초기화

                # bias는 고정
                self.split_bias = nn.ParameterDict()
                # self.split_bias['bias'] = nn.Parameter(torch.zeros(self.super_out_dim))
                self.split_bias['bias'] = nn.Parameter(torch.empty(self.super_out_dim))
                self._init_split_param(self.split_bias['bias'], is_bias=True)

                # print(f"Split head into 1x{len(dim1_splits)} weight segments and 1 bias.")

            elif name == 'qkv' or name == 'proj':
                embed_dims = sorted(set(choices['embed_dim']))

                if name == 'qkv':
                    dim0_splits = [d * 3 for d in embed_dims] + [self.super_out_dim]
                    dim1_splits = embed_dims + [self.super_in_dim]
                else:  # 'proj'
                    dim0_splits = embed_dims + [self.super_out_dim]
                    dim1_splits = embed_dims + [self.super_in_dim]

                self.split_weights = nn.ParameterDict()
                for i in range(len(dim0_splits)):
                    for j in range(len(dim1_splits)):
                        start_dim0 = 0 if i == 0 else dim0_splits[i - 1]
                        end_dim0 = dim0_splits[i]
                        start_dim1 = 0 if j == 0 else dim1_splits[j - 1]
                        end_dim1 = dim1_splits[j]
                        shape = (end_dim0 - start_dim0, end_dim1 - start_dim1)
                        param_name = f'w{i+1}_{j+1}'
                        self.split_weights[param_name] = nn.Parameter(torch.empty(shape))
                        self._init_split_param(self.split_weights[param_name])

                self.split_bias = nn.ParameterDict()
                for i in range(len(dim0_splits)):
                    start = dim0_splits[i - 1] if i > 0 else 0
                    end = dim0_splits[i]
                    key = f'bias_{i+1}'
                    self.split_bias[key] = nn.Parameter(torch.empty(end - start))
                    self._init_split_param(self.split_bias[key], is_bias=True)

        # self.w1 = nn.Parameter(self.weight.data.clone(), requires_grad=True)
        
        # self.bias1 = nn.Parameter(self.bias.data.clone(), requires_grad=True)

        self.scale = scale
        self._reset_parameters(bias, uniform_, non_linear)
        self.profiling = False

    def _init_split_param(self, param, is_bias=False):
        if is_bias:
            nn.init.constant_(param, 0)
        else:
            trunc_normal_(param, std=0.02)

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
        self.samples['weight'] = sample_weight(self, self.sample_in_dim, self.sample_out_dim, self.sample_in_dim_prev, self.sample_out_dim_prev)
        self.samples['bias'] = self.bias
        self.sample_scale = self.super_out_dim/self.sample_out_dim
        if self.bias is not None:
            self.samples['bias'] = sample_bias(self, self.sample_out_dim, self.sample_out_dim_prev)
        return self.samples

    def forward(self, x):
        self.sample_parameters()
        # print("self.samples['weight'].shape : ", self.samples['weight'].shape)
        # print("self.samples['bias'].shape : ", self.samples['bias'].shape)
        # return F.linear(x, self.w1, self.bias1) * (self.sample_scale if self.scale else 1)
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

# 수정된 sample_weight 함수로, 주어진 법칙대로 requires_grad를 설정함

def sample_weight(self, sample_in_dim, sample_out_dim, sample_in_dim_prev=None, sample_out_dim_prev=None):
    name = self.name
    choices = self.choices

    dim0_splits = [self.super_out_dim]
    dim1_splits = [self.super_in_dim]

    if name in ['fc1', 'fc2']:
        embed_dims = sorted(set(choices['embed_dim']))
        mlp_ratios = sorted(set(choices['mlp_ratio']))

        dim_embed = embed_dims + [self.super_in_dim if name == 'fc1' else self.super_out_dim]
        dim_mlp = sorted({int(e * r) for e in embed_dims for r in mlp_ratios})
        dim_mlp += [self.super_out_dim if name == 'fc1' else self.super_in_dim]

        if name == 'fc1':
            dim0_splits = dim_mlp
            dim1_splits = dim_embed
        else:
            dim0_splits = dim_embed
            dim1_splits = dim_mlp

        # 현재 config에 해당하는 (embed_dim, mlp_ratio) 페어 구하기
        current_pair = None
        for e in embed_dims:
            for r in mlp_ratios:
                out = int(e * r)
                if name == 'fc1':
                    if out >= sample_out_dim and e >= sample_in_dim:
                        current_pair = (e, r)
                        break
                else:  # fc2
                    if e >= sample_out_dim and out >= sample_in_dim:
                        current_pair = (e, r)
                        break
            if current_pair:
                break

        # print("current_pair: ", current_pair)

        assert current_pair is not None, "현재 config에 맞는 (embed_dim, mlp_ratio) 페어를 찾을 수 없습니다."

        # 앞선 페어들 정의
        preceding_pairs = []
        for e in embed_dims:
            for r in mlp_ratios:
                if (e < current_pair[0]) or (e == current_pair[0] and r < current_pair[1]):
                    preceding_pairs.append((e, r))

        # i_active, j_active 계산
        i_active = next(i for i, val in enumerate(dim0_splits) if val >= sample_out_dim)
        j_active = next(j for j, val in enumerate(dim1_splits) if val >= sample_in_dim)

        # 🔹 Step 1: 전체 weight에 대해 i <= i_active and j <= j_active 면 True, 나머지 False
        for i in range(len(dim0_splits)):
            for j in range(len(dim1_splits)):
                key = f'w{i+1}_{j+1}'
                self.split_weights[key].requires_grad = (i <= i_active and j <= j_active)

        # 🔹 Step 2: preceding_pairs 에 해당하는 (e, r)에 해당하는 block 전체를 False로 덮어쓰기
        for (e, r) in preceding_pairs:
            out = int(e * r)
            if name == 'fc1':
                i_block = next(i for i, val in enumerate(dim0_splits) if val >= out)
                j_block = next(j for j, val in enumerate(dim1_splits) if val >= e)
            else:  # fc2
                i_block = next(i for i, val in enumerate(dim0_splits) if val >= e)
                j_block = next(j for j, val in enumerate(dim1_splits) if val >= out)

            for i in range(i_block + 1):
                for j in range(j_block + 1):
                    key = f'w{i+1}_{j+1}'
                    self.split_weights[key].requires_grad = False

    elif name == 'head':
        embed_dims = sorted(set(choices['embed_dim']))
        dim1_splits = embed_dims + [self.super_in_dim]

        j_active = next(j for j, val in enumerate(dim1_splits) if val >= sample_in_dim)

        for j in range(len(dim1_splits)):
            key = f'w1_{j+1}'
            self.split_weights[key].requires_grad = (j == j_active)

    elif name == 'qkv':
        embed_dims = sorted(set(choices['embed_dim']))
        dim0_splits = [3 * e for e in embed_dims] + [self.super_out_dim]
        dim1_splits = embed_dims + [self.super_in_dim]

        i_active = next(i for i, val in enumerate(dim0_splits) if val >= sample_out_dim)
        j_active = next(j for j, val in enumerate(dim1_splits) if val >= sample_in_dim)

        for i in range(len(dim0_splits)):
            for j in range(len(dim1_splits)):
                key = f'w{i+1}_{j+1}'
                self.split_weights[key].requires_grad = (i == i_active and j == j_active)

    elif name == 'proj':
        embed_dims = sorted(set(choices['embed_dim']))
        dim0_splits = embed_dims + [self.super_out_dim]
        dim1_splits = embed_dims + [self.super_in_dim]

        i_active = next(i for i, val in enumerate(dim0_splits) if val >= sample_out_dim)
        j_active = next(j for j, val in enumerate(dim1_splits) if val >= sample_in_dim)

        for i in range(len(dim0_splits)):
            for j in range(len(dim1_splits)):
                key = f'w{i+1}_{j+1}'
                self.split_weights[key].requires_grad = (i == i_active and j == j_active)

    # weight 조립
    row_blocks = []
    for i in range(len(dim0_splits)):
        col_blocks = []
        for j in range(len(dim1_splits)):
            key = f'w{i+1}_{j+1}'
            col_blocks.append(self.split_weights[key])
        row_blocks.append(torch.cat(col_blocks, dim=1))
    full_weight = torch.cat(row_blocks, dim=0)

    sample_weight = full_weight[:sample_out_dim, :sample_in_dim]

    # print(f"\n[🔍 {self.name} - Weight requires_grad status]")
    # for key in self.split_weights:
    #     print(f"  {key:10s} -> {self.split_weights[key].requires_grad}")

    return sample_weight


def sample_bias(self, sample_out_dim, sample_out_dim_prev=None):
    name = self.name
    choices = self.choices

    if name in ['fc1', 'fc2']:
        embed_dims = sorted(set(choices['embed_dim']))
        mlp_ratios = sorted(set(choices['mlp_ratio']))

        if name == 'fc1':
            dim0_sizes = sorted({int(e * r) for e in embed_dims for r in mlp_ratios})
            dim0_sizes += [self.super_out_dim]
        else:
            dim0_sizes = embed_dims + [self.super_out_dim]

        i_active = next(i for i, val in enumerate(dim0_sizes) if val >= sample_out_dim)

        collected_bias = []
        for i in range(len(dim0_sizes)):
            key = f'bias_{i+1}'
            self.split_bias[key].requires_grad = (i == i_active)
            collected_bias.append(self.split_bias[key])

        full_bias = torch.cat(collected_bias, dim=0)
        sample_bias = full_bias[:sample_out_dim]

        # print(f"\n[🔍 {self.name} - Bias requires_grad status]")
        # for key in self.split_bias:
        #     print(f"  {key:10s} -> {self.split_bias[key].requires_grad}")

        return sample_bias

    elif name == 'head':
        self.split_bias['bias'].requires_grad = True
        sample_bias = self.split_bias['bias'][:sample_out_dim]

        # print(f"\n[🔍 {self.name} - Bias requires_grad status]")
        # for key in self.split_bias:
        #     print(f"  {key:10s} -> {self.split_bias[key].requires_grad}")

        return sample_bias
    
    elif name == 'qkv':
        embed_dims = sorted(set(choices['embed_dim']))
        dim0_sizes = [3 * e for e in embed_dims] + [self.super_out_dim]

        i_active = next(i for i, val in enumerate(dim0_sizes) if val >= sample_out_dim)

        collected_bias = []
        for i in range(len(dim0_sizes)):
            key = f'bias_{i+1}'
            self.split_bias[key].requires_grad = (i == i_active)
            collected_bias.append(self.split_bias[key])

        full_bias = torch.cat(collected_bias, dim=0)
        return full_bias[:sample_out_dim]

    elif name == 'proj':
        embed_dims = sorted(set(choices['embed_dim']))
        dim0_sizes = embed_dims + [self.super_out_dim]

        i_active = next(i for i, val in enumerate(dim0_sizes) if val >= sample_out_dim)

        collected_bias = []
        for i in range(len(dim0_sizes)):
            key = f'bias_{i+1}'
            self.split_bias[key].requires_grad = (i == i_active)
            collected_bias.append(self.split_bias[key])

        full_bias = torch.cat(collected_bias, dim=0)
        return full_bias[:sample_out_dim]
