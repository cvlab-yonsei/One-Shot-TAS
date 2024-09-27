import torch
import torch.nn as nn
import torch.nn.functional as F

def uniform_element_selection(tensor, target_dim):
    """
    Uniformly selects elements from the tensor along the specified dimension.
    
    Parameters:
    tensor (torch.Tensor): The input tensor.
    target_dim (int): The target dimension size.
    
    Returns:
    torch.Tensor: A tensor with the selected elements.
    """
    original_dim = tensor.size(0)
    indices = torch.linspace(0, original_dim - 1, target_dim).long().to(tensor.device)
    return tensor[indices]

class LayerNormSuper(torch.nn.LayerNorm):
    def __init__(self, super_embed_dim):
        super().__init__(super_embed_dim)

        # the largest embed dim
        self.super_embed_dim = super_embed_dim

        # the current sampled embed dim
        self.sample_embed_dim = None

        self.samples = {}
        self.profiling = False

    def profile(self, mode=True):
        self.profiling = mode

    def sample_parameters(self, resample=False):
        if self.profiling or resample:
            return self._sample_parameters()
        return self.samples

    def _sample_parameters(self):
        # self.samples['weight'] = self.weight[:self.sample_embed_dim]
        # self.samples['bias'] = self.bias[:self.sample_embed_dim]
        self.samples['weight'] = uniform_element_selection(self.weight, self.sample_embed_dim)
        self.samples['bias'] = uniform_element_selection(self.bias, self.sample_embed_dim)
        return self.samples

    def set_sample_config(self, sample_embed_dim):
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
