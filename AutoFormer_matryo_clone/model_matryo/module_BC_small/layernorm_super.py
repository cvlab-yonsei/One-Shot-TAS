import torch
import torch.nn as nn
# import torch.nn.functional as F
from torch.nn import functional as F
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

class LayerNormSuper(nn.LayerNorm):
    def __init__(self, super_embed_dim, eps=1e-5, choices=None):
        super().__init__(normalized_shape=super_embed_dim, eps=eps, elementwise_affine=False)
        self.super_embed_dim = super_embed_dim
        self.sample_embed_dim = None
        self.sample_embed_dim_prev = None
        self.eps = eps
        self.profiling = False
        self.samples = {}

        self.choices = choices

        embed_dims = sorted(set(choices['embed_dim']))

        self.dim0_splits = embed_dims

        self.split_weights = nn.ParameterDict()
        self.split_bias = nn.ParameterDict()

        for i in range(len(self.dim0_splits)):
            start_dim0 = 0 if i == 0 else self.dim0_splits[i - 1]
            end_dim0 = self.dim0_splits[i]
            shape = (end_dim0 - start_dim0,)

            self.split_bias[f'bias_{i+1}'] = nn.Parameter(torch.empty(shape))  # 초기화 X
            self.split_weights[f'w_{i+1}'] = nn.Parameter(torch.empty(shape))  # 초기화 X

    def _concat_params(self):
        weight = torch.cat([self.split_weights[k] for k in sorted(self.split_weights.keys())], dim=0)
        bias = torch.cat([self.split_bias[k] for k in sorted(self.split_bias.keys())], dim=0)
        return weight, bias

    def _sample_parameters(self, sample_embed_dim=None, case_num=None):
        choices = self.choices

        weight, bias = self._concat_params()

        embed_dims = sorted(set(choices['embed_dim']))
        dim0_sizes = embed_dims 

        # 활성화 대상: sample_embed_dim 이상인 것들 전부
        for i, dim in enumerate(dim0_sizes):
            # active = (dim >= sample_embed_dim) # 이거 embed_dims[1]로 하드코딩
            active = (dim >= embed_dims[1]) # 이거 embed_dims[1]로 하드코딩
            self.split_bias[f'bias_{i+1}'].requires_grad = active
            # self.split_bias[f'bias_{i+1}'].requires_grad = True
            self.split_weights[f'w_{i+1}'].requires_grad = active


        # i_active = next(i for i, val in enumerate(dim0_sizes) if val >= sample_embed_dim)

        # for i in range(len(dim0_sizes)):
        #     key = f'bias_{i+1}'
        #     self.split_bias[key].requires_grad = (i == i_active)

        #     key2 = f'w_{i+1}'
        #     self.split_weights[key2].requires_grad = (i == i_active)

        # if case_num is not None:
        #     init_split_parameters_with_gaussian(self.split_weights)
        # if case_num is not None:
        #     init_split_parameters_with_gaussian(self.split_bias)

        # ##############################

        # # 🔹 weight alignment
        # with torch.no_grad():
        #     mask = torch.zeros_like(weight, dtype=torch.bool)
        #     offset = 0
        #     for i in range(len(dim0_sizes)):
        #         w = self.split_weights[f'w_{i+1}']
        #         L = w.shape[0]
        #         requires_grad = w.requires_grad
        #         mask[offset:offset + L] = requires_grad
        #         offset += L

        #     W_true = weight[mask]
        #     W_false = weight[~mask]

        #     if W_true.numel() > 0 and W_false.numel() > 0:
        #         mean_true = W_true.norm(p=2) / W_true.numel()
        #         mean_false = W_false.norm(p=2) / W_false.numel()
        #         λ = (mean_false / mean_true).detach()
        #         # λ = (mean_false / (mean_true + 1e-8)).clamp(min=0.1, max=5.0).detach()

        #         weight[mask] *= λ

        # # 🔹 bias alignment
        # with torch.no_grad():
        #     mask = torch.zeros_like(bias, dtype=torch.bool)
        #     offset = 0
        #     for i in range(len(dim0_sizes)):
        #         b = self.split_bias[f'bias_{i+1}']
        #         L = b.shape[0]
        #         requires_grad = b.requires_grad
        #         mask[offset:offset + L] = requires_grad
        #         offset += L

        #     B_true = bias[mask]
        #     B_false = bias[~mask]

        #     if B_true.numel() > 0 and B_false.numel() > 0:
        #         mean_true = B_true.abs().mean()
        #         mean_false = B_false.abs().mean()
        #         λ = (mean_false / mean_true).detach()
        #         # λ = (mean_false / (mean_true + 1e-8)).clamp(min=0.1, max=5.0).detach()

        #         bias[mask] *= λ

        # ##############################

        self.samples['weight'] = weight[:self.sample_embed_dim]
        self.samples['bias'] = bias[:self.sample_embed_dim]
        return self.samples

    def sample_parameters(self, resample=False):
        if self.profiling or resample:
            return self._sample_parameters()
        return self.samples

    def set_sample_config(self, sample_embed_dim, sample_embed_dim_prev=None, case_num=None):
        self.sample_embed_dim = sample_embed_dim
        self.sample_embed_dim_prev = sample_embed_dim_prev
        self._sample_parameters(sample_embed_dim=sample_embed_dim, case_num=case_num)

    def forward(self, x):
        self.sample_parameters()
        return F.layer_norm(x, (self.sample_embed_dim,),
                            weight=self.samples['weight'],
                            bias=self.samples['bias'],
                            eps=self.eps)

    def calc_sampled_param_num(self):
        return self.samples['weight'].numel() + self.samples['bias'].numel()

    def get_complexity(self, sequence_length):
        return sequence_length * self.sample_embed_dim
