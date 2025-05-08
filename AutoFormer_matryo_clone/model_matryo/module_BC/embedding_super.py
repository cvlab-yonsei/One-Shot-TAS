import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils import to_2tuple
import numpy as np
from collections import OrderedDict
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

class PatchembedConvSuper(nn.Conv2d):
    def __init__(self, in_chans, super_embed_dim, patch_size, choices):
        super().__init__(in_chans, super_embed_dim, kernel_size=patch_size, stride=patch_size)
        self.super_embed_dim = super_embed_dim
        self.choices = choices

        embed_dims = sorted(set(choices['embed_dim']))
        self.dim0_splits = embed_dims + [super_embed_dim]

        self.weight.requires_grad = False
        self.bias.requires_grad = False

        self.split_weights = nn.ParameterDict()
        self.split_biases = nn.ParameterDict()

        for i in range(len(self.dim0_splits)):
            start = self.dim0_splits[i - 1] if i > 0 else 0
            end = self.dim0_splits[i]
            weight_shape = (end - start, self.in_channels, *self.kernel_size)
            bias_shape = (end - start,)

            self.split_weights[f'w{i+1}'] = nn.Parameter(torch.empty(weight_shape))
            self.split_biases[f'b{i+1}'] = nn.Parameter(torch.empty(bias_shape))

            nn.init.kaiming_normal_(self.split_weights[f'w{i+1}'], mode='fan_out', nonlinearity='relu')
            nn.init.constant_(self.split_biases[f'b{i+1}'], 0)

        self.sampled_weight = None
        self.sampled_bias = None

    def set_sample_config(self, sample_embed_dim, case_num=None):
        j_end = next(i for i, v in enumerate(self.dim0_splits) if v >= sample_embed_dim)
        # for i in range(len(self.dim0_splits)):
        #     w = self.split_weights[f'w{i+1}']
        #     b = self.split_biases[f'b{i+1}']
        #     w.requires_grad = (i == j_end)
        #     b.requires_grad = (i == j_end)
        if case_num is not None:
            if case_num == 1:
                true_label = [(1)]
            elif case_num == 2:
                # true_label = [(2)]
                true_label = [(2), (3)]
            elif case_num == 3:
                true_label = [(3)]

            for i in range(len(self.dim0_splits)):
                self.split_weights[f'w{i+1}'].requires_grad = ((i + 1) in true_label)
                self.split_biases[f'b{i+1}'].requires_grad = ((i + 1) in true_label)
                # self.split_biases[f'b{i+1}'].requires_grad = True

            # if case_num is not None:
            #     init_split_parameters_with_gaussian(self.split_weights)

            # if case_num is not None:
            #     init_split_parameters_with_gaussian(self.split_biases)

        weights = [self.split_weights[f'w{i+1}'] for i in range(j_end + 1)]
        biases = [self.split_biases[f'b{i+1}'] for i in range(j_end + 1)]
        full_weight = torch.cat(weights, dim=0)
        full_bias = torch.cat(biases, dim=0)

        self.sampled_weight = full_weight[:sample_embed_dim]
        self.sampled_bias = full_bias[:sample_embed_dim]

        # ###########################

        # # 🔹 weight alignment
        # with torch.no_grad():
        #     mask = torch.zeros_like(full_weight, dtype=torch.bool)
        #     offset = 0
        #     for i in range(j_end + 1):
        #         w = self.split_weights[f'w{i+1}']
        #         h = w.shape[0]
        #         mask[offset:offset + h] = w.requires_grad
        #         offset += h

        #     W_true = full_weight[mask]
        #     W_false = full_weight[~mask]

        #     if W_true.numel() > 0 and W_false.numel() > 0:
        #         norm_true = W_true.norm(p=2)
        #         norm_false = W_false.norm(p=2)
        #         mean_true = norm_true / W_true.numel()
        #         mean_false = norm_false / W_false.numel()
        #         λ = (mean_false / mean_true).detach()
        #         # λ = (mean_false / (mean_true + 1e-8)).clamp(min=0.1, max=5.0).detach()

        #         full_weight[mask] *= λ

        # self.sampled_weight = full_weight[:sample_embed_dim]

        # # 🔹 bias alignment
        # with torch.no_grad():
        #     mask = torch.zeros_like(full_bias, dtype=torch.bool)
        #     offset = 0
        #     for i in range(j_end + 1):
        #         b = self.split_biases[f'b{i+1}']
        #         h = b.shape[0]
        #         mask[offset:offset + h] = b.requires_grad
        #         offset += h

        #     bias_true = full_bias[mask]
        #     bias_false = full_bias[~mask]

        #     if bias_true.numel() > 0 and bias_false.numel() > 0:
        #         mean_true = bias_true.abs().mean()
        #         mean_false = bias_false.abs().mean()
        #         λ = (mean_false / mean_true).detach()
        #         # λ = (mean_false / (mean_true + 1e-8)).clamp(min=0.1, max=5.0).detach()

        #         full_bias[mask] *= λ

        # self.sampled_bias = full_bias[:sample_embed_dim]

        # ###########################

        # # Print requires_grad status
        # print(f"\n[🔍 PatchembedConvSuper - Weight requires_grad status]")
        # for key in self.split_weights:
        #     print(f"  {key:10s} -> {self.split_weights[key].requires_grad}")
        # if self.split_biases:
        #     print(f"\n[🔍 PatchembedConvSuper - Bias requires_grad status]")
        #     for key in self.split_biases:
        #         print(f"  {key:10s} -> {self.split_biases[key].requires_grad}")


    def forward(self, x):
        return F.conv2d(x, self.sampled_weight, self.sampled_bias,
                        stride=self.stride, padding=self.padding, dilation=self.dilation)


class PatchembedSuper(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, scale=False, choices=None):
        super(PatchembedSuper, self).__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        # self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.super_embed_dim = embed_dim
        self.scale = scale

    # sampled_
        self.sample_embed_dim = None
        self.sampled_weight = None
        self.sampled_bias = None
        self.sampled_scale = None

        self.sample_embed_dim_prev = None
        self.sampled_weight_prev = None
        self.sampled_bias_prev = None
        self.sampled_scale_prev = None

        self.choices = choices

        self.proj = PatchembedConvSuper(in_chans, embed_dim, patch_size, choices)

    def set_sample_config(self, sample_embed_dim, sample_embed_dim_prev=None, case_num=None):
        self.sample_embed_dim = sample_embed_dim
        # self.sampled_weight = self.proj.weight[:sample_embed_dim, ...]
        # self.sampled_bias = self.proj.bias[:self.sample_embed_dim, ...]
        self.proj.set_sample_config(sample_embed_dim, case_num=case_num)
        if self.scale:
            self.sampled_scale = self.super_embed_dim / sample_embed_dim

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        # x = F.conv2d(x, self.sampled_weight, self.sampled_bias, stride=self.patch_size, padding=self.proj.padding, dilation=self.proj.dilation).flatten(2).transpose(1,2)
        if self.scale:
            return x * self.sampled_scale
        return x
    
    def calc_sampled_param_num(self):
        return self.proj.sampled_weight.numel() + self.proj.sampled_bias.numel()

    def get_complexity(self, sequence_length):
        total_flops = 0
        if self.proj.sampled_bias is not None:
             total_flops += self.proj.sampled_bias.size(0)
        total_flops += sequence_length * np.prod(self.proj.sampled_weight.size())
        return total_flops
    
    # def calc_sampled_param_num(self):
    #     return  self.sampled_weight.numel() + self.sampled_bias.numel()

    # def get_complexity(self, sequence_length):
    #     total_flops = 0
    #     if self.sampled_bias is not None:
    #          total_flops += self.sampled_bias.size(0)
    #     total_flops += sequence_length * np.prod(self.sampled_weight.size())
    #     return total_flops