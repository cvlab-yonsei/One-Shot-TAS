import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils import to_2tuple
import numpy as np

class Conv2dSuper(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.weight.requires_grad = False
        self.bias.requires_grad = False

    @property
    def weight(self):
        # 만약 self.w1이 빈 텐서라면, 그냥 self.w2를 반환
        if self.w1.numel() == 0:
            return self.w2
        else:
            return torch.cat([self.w1, self.w2], dim=0)
        
    @property
    def bias(self):
        # frozen bias가 없으면, bias1은 빈 텐서.
        if self.bias1.numel() == 0:
            return self.bias2
        else:
            return torch.cat([self.bias1, self.bias2], dim=0)


class PatchembedSuper(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, scale=False):
        super(PatchembedSuper, self).__init__()
        
        patch_size_tmp = patch_size
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = Conv2dSuper(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

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

        self.proj.w1 = nn.Parameter(torch.rand(embed_dim//4, in_chans, *patch_size), requires_grad=True)
        self.proj.w2 = nn.Parameter(torch.rand(embed_dim//4, in_chans, *patch_size), requires_grad=True)
        self.proj.w3 = nn.Parameter(torch.rand(embed_dim//2, in_chans, *patch_size), requires_grad=True)

        # print("self.proj.weight.shape : ", self.proj.weight.shape)
        # print("self.proj.w1.shape : ", self.proj.w1.shape)
        # print("self.proj.w2.shape : ", self.proj.w2.shape)
        # print("self.proj.w3.shape : ", self.proj.w3.shape)

        self.proj.bias1 = nn.Parameter(torch.rand(embed_dim//4), requires_grad=True)
        self.proj.bias2 = nn.Parameter(torch.rand(embed_dim//4), requires_grad=True)
        self.proj.bias3 = nn.Parameter(torch.rand(embed_dim//2), requires_grad=True)

    def set_sample_config(self, sample_embed_dim, sample_embed_dim_prev=None):
        self.sample_embed_dim = sample_embed_dim

        full_weight = torch.cat([self.proj.w1, self.proj.w2], dim=0)
        full_weight_out = torch.cat([full_weight, self.proj.w3], dim=0)
        
        full_bias = torch.cat([self.proj.bias1, self.proj.bias2], dim=0)
        full_bias_out = torch.cat([full_bias, self.proj.bias3], dim=0)
        
        if sample_embed_dim_prev is None:
            new_w1 = torch.empty(0, device=self.proj.weight.device)
            new_w2 = full_weight_out[:sample_embed_dim, ...]
            new_w3 = full_weight_out[sample_embed_dim:self.super_embed_dim, ...]
            self.proj.w1 = nn.Parameter(new_w1, requires_grad=False)
            self.proj.w2 = nn.Parameter(new_w2, requires_grad=True)
            self.proj.w3 = nn.Parameter(new_w3, requires_grad=False)

            new_bias1 = torch.empty(0, device=self.proj.bias.device)
            new_bias2 = full_bias_out[:sample_embed_dim, ...]
            new_bias3 = full_bias_out[sample_embed_dim:self.super_embed_dim, ...]
            self.proj.bias1 = nn.Parameter(new_bias1, requires_grad=False)
            self.proj.bias2 = nn.Parameter(new_bias2, requires_grad=True)
            self.proj.bias3 = nn.Parameter(new_bias3, requires_grad=False)

            self.sampled_weight = self.proj.weight[:sample_embed_dim, ...]
            self.sampled_bias = self.proj.bias[:sample_embed_dim, ...]
        else:
            new_w1 = full_weight_out[:sample_embed_dim_prev, ...].detach()
            new_w2 = full_weight_out[sample_embed_dim_prev:sample_embed_dim, ...]
            new_w3 = full_weight_out[sample_embed_dim:self.super_embed_dim, ...].detach()

            self.proj.w1 = nn.Parameter(new_w1, requires_grad=False)
            self.proj.w2 = nn.Parameter(new_w2, requires_grad=True)
            self.proj.w3 = nn.Parameter(new_w3, requires_grad=False)

            new_bias1 = full_bias_out[:sample_embed_dim_prev, ...].detach()
            new_bias2 = full_bias_out[sample_embed_dim_prev:sample_embed_dim, ...]
            new_bias3 = full_bias_out[sample_embed_dim:self.super_embed_dim, ...].detach()

            self.proj.bias1 = nn.Parameter(new_bias1, requires_grad=False)
            self.proj.bias2 = nn.Parameter(new_bias2, requires_grad=True)
            self.proj.bias3 = nn.Parameter(new_bias3, requires_grad=False)


            frozen_weight = self.proj.weight[:sample_embed_dim_prev, ...].detach()
            # trainable: 나머지 영역 (channels: sample_embed_dim_prev ~ sample_embed_dim)
            trainable_weight = self.proj.weight[sample_embed_dim_prev:sample_embed_dim, ...]
            self.sampled_weight = torch.cat([frozen_weight, trainable_weight], dim=0)

            frozen_bias = self.proj.bias[:sample_embed_dim_prev].detach()
            trainable_bias = self.proj.bias[sample_embed_dim_prev:sample_embed_dim]
            self.sampled_bias = torch.cat([frozen_bias, trainable_bias], dim=0)
            
        if self.scale:
            self.sampled_scale = self.super_embed_dim / sample_embed_dim

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        
        x = F.conv2d(x, self.proj.weight, self.proj.bias, stride=self.patch_size, padding=self.proj.padding, dilation=self.proj.dilation).flatten(2).transpose(1,2)
        # x = F.conv2d(x, self.sampled_weight, self.sampled_bias, stride=self.patch_size, padding=self.proj.padding, dilation=self.proj.dilation).flatten(2).transpose(1,2)
        if self.scale:
            return x * self.sampled_scale
        return x
    
    def calc_sampled_param_num(self):
        return  self.sampled_weight.numel() + self.sampled_bias.numel()

    def get_complexity(self, sequence_length):
        total_flops = 0
        if self.sampled_bias is not None:
             total_flops += self.sampled_bias.size(0)
        total_flops += sequence_length * np.prod(self.sampled_weight.size())
        return total_flops