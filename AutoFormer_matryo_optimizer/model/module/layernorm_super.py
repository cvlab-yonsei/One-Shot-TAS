import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNormSuper(torch.nn.LayerNorm):
    def __init__(self, super_embed_dim):
        super().__init__(super_embed_dim)

        # the largest embed dim
        self.super_embed_dim = super_embed_dim

        # the current sampled embed dim
        self.sample_embed_dim = None
        self.sample_embed_dim_prev = None

        self.weight.requires_grad = False

        self.w1 = nn.Parameter(torch.rand(0), requires_grad=False)
        self.w2 = nn.Parameter(torch.rand(super_embed_dim), requires_grad=True)
        self.w3 = nn.Parameter(torch.rand(0), requires_grad=False)

        self.bias1 = nn.Parameter(torch.rand(0), requires_grad=False)
        self.bias2 = nn.Parameter(torch.rand(super_embed_dim), requires_grad=True)
        self.bias3 = nn.Parameter(torch.rand(0), requires_grad=False)

        self.samples = {}
        self.profiling = False

    def profile(self, mode=True):
        self.profiling = mode

    def sample_parameters(self, resample=False):
        if self.profiling or resample:
            return self._sample_parameters()
        return self.samples

    def _sample_parameters(self):
        full_weight = torch.cat([self.w1, self.w2], dim=0)
        full_weight_out = torch.cat([full_weight, self.w3], dim=0)
        
        full_bias = torch.cat([self.bias1, self.bias2], dim=0)
        full_bias_out = torch.cat([full_bias, self.bias3], dim=0)

        if self.sample_embed_dim_prev is None:
            new_w1 = torch.empty(0, device=self.weight.device)
            new_w2 = full_weight_out[:self.sample_embed_dim]
            new_w3 = full_weight_out[self.sample_embed_dim:self.super_embed_dim]
            self.w1 = nn.Parameter(new_w1, requires_grad=False)
            self.w2 = nn.Parameter(new_w2, requires_grad=True)
            self.w3 = nn.Parameter(new_w3, requires_grad=False)

            new_bias1 = torch.empty(0, device=self.bias.device)
            new_bias2 = full_bias_out[:self.sample_embed_dim]
            new_bias3 = full_bias_out[self.sample_embed_dim:self.super_embed_dim]
            self.bias1 = nn.Parameter(new_bias1, requires_grad=False)
            self.bias2 = nn.Parameter(new_bias2, requires_grad=True)
            self.bias3 = nn.Parameter(new_bias3, requires_grad=False)

            self.samples['weight'] = self.weight[:self.sample_embed_dim]
            self.samples['bias'] = self.bias[:self.sample_embed_dim]
        else:
            new_w1 = full_weight_out[:self.sample_embed_dim_prev].detach()
            new_w2 = full_weight_out[self.sample_embed_dim_prev:self.sample_embed_dim]
            new_w3 = full_weight_out[self.sample_embed_dim:self.super_embed_dim].detach()

            self.w1 = nn.Parameter(new_w1, requires_grad=False)
            self.w2 = nn.Parameter(new_w2, requires_grad=True)
            self.w3 = nn.Parameter(new_w3, requires_grad=False)

            new_bias1 = full_bias_out[:self.sample_embed_dim_prev].detach()
            new_bias2 = full_bias_out[self.sample_embed_dim_prev:self.sample_embed_dim]
            new_bias3 = full_bias_out[self.sample_embed_dim:self.super_embed_dim].detach()

            self.bias1 = nn.Parameter(new_bias1, requires_grad=False)
            self.bias2 = nn.Parameter(new_bias2, requires_grad=True)
            self.bias3 = nn.Parameter(new_bias3, requires_grad=False)

            frozen_weight = self.weight[:self.sample_embed_dim_prev].detach()
            trainable_weight = self.weight[self.sample_embed_dim_prev:self.sample_embed_dim]
            self.samples['weight'] = torch.cat([frozen_weight, trainable_weight], dim=0)

            frozen_bias = self.bias[:self.sample_embed_dim_prev].detach()
            trainable_bias = self.bias[self.sample_embed_dim_prev:self.sample_embed_dim]
            self.samples['bias'] = torch.cat([frozen_bias, trainable_bias], dim=0)
        return self.samples

    def set_sample_config(self, sample_embed_dim, sample_embed_dim_prev=None):
        self.sample_embed_dim = sample_embed_dim
        self.sample_embed_dim_prev = sample_embed_dim_prev
        self._sample_parameters()

    @property
    def weight(self):
        # frozen bias가 없으면, bias1은 빈 텐서.
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

    def forward(self, x):
        self.sample_parameters()
        return F.layer_norm(x, (self.sample_embed_dim,), weight=self.weight, bias=self.bias, eps=self.eps)
        # return F.layer_norm(x, (self.sample_embed_dim,), weight=self.samples['weight'], bias=self.samples['bias'], eps=self.eps)

    def calc_sampled_param_num(self):
        assert 'weight' in self.samples.keys()
        assert 'bias' in self.samples.keys()
        return self.samples['weight'].numel() + self.samples['bias'].numel()

    def get_complexity(self, sequence_length):
        return sequence_length * self.sample_embed_dim
