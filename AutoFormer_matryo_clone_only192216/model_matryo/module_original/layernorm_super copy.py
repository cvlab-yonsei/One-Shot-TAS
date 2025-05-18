import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_


class LayerNormSuper(torch.nn.LayerNorm):
    def __init__(self, super_embed_dim, choices=None):
        super().__init__(super_embed_dim)

        # the largest embed dim
        self.super_embed_dim = super_embed_dim

        # the current sampled embed dim
        self.sample_embed_dim = None
        self.sample_embed_dim_prev = None

        self.weight.requires_grad = False
        self.bias.requires_grad = False

        self.samples = {}
        self.profiling = False

        # split 기준 설정
        embed_dims = sorted(set(choices['embed_dim']))
        self.dim_splits = embed_dims + [super_embed_dim]

        # 각 구간에 대해 weight와 bias 분할 정의
        self.split_weights = nn.ParameterDict()
        self.split_biases = nn.ParameterDict()
        for i in range(len(self.dim_splits)):
            start = self.dim_splits[i - 1] if i > 0 else 0
            end = self.dim_splits[i]
            dim = end - start
            w = nn.Parameter(torch.empty(dim))
            b = nn.Parameter(torch.empty(dim))
            trunc_normal_(w, std=0.02)
            nn.init.constant_(b, 0)
            self.split_weights[f'w{i+1}'] = w
            self.split_biases[f'b{i+1}'] = b

    def profile(self, mode=True):
        self.profiling = mode

    def sample_parameters(self, resample=False):
        if self.profiling or resample:
            return self._sample_parameters()
        return self.samples

    def _sample_parameters(self):
        embed_dim = self.sample_embed_dim
        splits = self.dim_splits

        # 활성화할 index 구간 찾기
        i_end = next(i for i, v in enumerate(splits) if v >= embed_dim)

        # 모든 파라미터 requires_grad = False
        for key in self.split_weights:
            self.split_weights[key].requires_grad = False
        for key in self.split_biases:
            self.split_biases[key].requires_grad = False

        # 오직 해당 블록만 requires_grad = True
        w_key = f'w{i_end+1}'
        b_key = f'b{i_end+1}'
        self.split_weights[w_key].requires_grad = True
        self.split_biases[b_key].requires_grad = True

        # 모든 weight, bias를 순서대로 concat
        all_weight = torch.cat([self.split_weights[f'w{i+1}'] for i in range(len(splits))], dim=0)
        all_bias = torch.cat([self.split_biases[f'b{i+1}'] for i in range(len(splits))], dim=0)

        # embed_dim만큼 잘라서 샘플 weight/bias 생성
        self.samples['weight'] = all_weight[:embed_dim]
        self.samples['bias'] = all_bias[:embed_dim]

        # print(f"\n[🔍 LayerNormSuper - Weight requires_grad status]")
        # for key in self.split_weights:
        #     print(f"  {key:10s} -> {self.split_weights[key].requires_grad}")

        # print(f"\n[🔍 LayerNormSuper - Bias requires_grad status]")
        # for key in self.split_biases:
        #     print(f"  {key:10s} -> {self.split_biases[key].requires_grad}")

        return self.samples

    def set_sample_config(self, sample_embed_dim, sample_embed_dim_prev=None):
        self.sample_embed_dim = sample_embed_dim
        self._sample_parameters()

    def forward(self, x):
        self.sample_parameters()
        return F.layer_norm(x, (self.sample_embed_dim,), weight=self.samples['weight'], bias=self.samples['bias'], eps=self.eps)

    def calc_sampled_param_num(self):
        assert 'weight' in self.samples.keys()
        assert 'bias' in self.samples.keys()
        return self.samples['weight'].numel() + self.samples['bias'].numel()

    def get_complexity(self, sequence_length):
        return sequence_length * self.sample_embed_dim
