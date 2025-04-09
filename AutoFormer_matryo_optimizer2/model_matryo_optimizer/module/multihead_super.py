import torch
# from torch import nn
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from .Linear_super import LinearSuper
from .qkv_super import qkv_super
from ..utils import trunc_normal_
def softmax(x, dim, onnx_trace=False):
    if onnx_trace:
        return F.softmax(x.float(), dim=dim)
    else:
        return F.softmax(x, dim=dim, dtype=torch.float32)

class RelativePosition2D_super(nn.Module):

    def __init__(self, num_units, max_relative_position, super_embed_dim=None):
        super().__init__()
        
        self.super_embed_dim = super_embed_dim
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        # The first element in embeddings_table_v is the vertical embedding for the class
        self.embeddings_table_v = nn.Parameter(torch.randn(max_relative_position * 2 + 2, num_units), requires_grad=False) # false 추가
        self.embeddings_table_h = nn.Parameter(torch.randn(max_relative_position * 2 + 2, num_units), requires_grad=False) # false 추가

        trunc_normal_(self.embeddings_table_v, std=.02)
        trunc_normal_(self.embeddings_table_h, std=.02)

        self.v1 = nn.Parameter(torch.rand(max_relative_position * 2 + 2, super_embed_dim//2), requires_grad=True)
        self.v2 = nn.Parameter(torch.rand(max_relative_position * 2 + 2, super_embed_dim//4), requires_grad=True)
        self.v3 = nn.Parameter(torch.rand(max_relative_position * 2 + 2, super_embed_dim//4), requires_grad=True)

        self.h1 = nn.Parameter(torch.rand(max_relative_position * 2 + 2, super_embed_dim//2), requires_grad=True)
        self.h2 = nn.Parameter(torch.rand(max_relative_position * 2 + 2, super_embed_dim//4), requires_grad=True)
        self.h3 = nn.Parameter(torch.rand(max_relative_position * 2 + 2, super_embed_dim//4), requires_grad=True)

        trunc_normal_(self.v1, std=.02)
        trunc_normal_(self.v2, std=.02)
        trunc_normal_(self.v3, std=.02)
        trunc_normal_(self.h1, std=.02)
        trunc_normal_(self.h2, std=.02)
        trunc_normal_(self.h3, std=.02)

        self.sample_head_dim = None
        self.sample_embeddings_table_h = None
        self.sample_embeddings_table_v = None

    def set_sample_config(self, sample_head_dim, sample_head_dim_prev=None):
        self.sample_head_dim = sample_head_dim

        full_v = torch.cat([self.v1, self.v2], dim=1)
        full_v_out = torch.cat([full_v, self.v3], dim=1)
        
        full_h = torch.cat([self.h1, self.h2], dim=1)
        full_h_out = torch.cat([full_h, self.h3], dim=1)

        if sample_head_dim_prev is None:
            new_v1 = torch.empty(0, device=self.weight.device)
            new_v2 = full_v_out[..., :sample_head_dim]
            new_v3 = full_v_out[..., sample_head_dim:self.super_embed_dim]
            self.v1 = nn.Parameter(new_v1, requires_grad=False)
            self.v2 = nn.Parameter(new_v2, requires_grad=True)
            self.v3 = nn.Parameter(new_v3, requires_grad=False)

            new_h1 = torch.empty(0, device=self.bias.device)
            new_h2 = full_h_out[..., :sample_head_dim]
            new_h3 = full_h_out[..., sample_head_dim:self.super_embed_dim]
            self.h1 = nn.Parameter(new_h1, requires_grad=False)
            self.h2 = nn.Parameter(new_h2, requires_grad=True)
            self.h3 = nn.Parameter(new_h3, requires_grad=False)

            self.sample_embeddings_table_h = self.embeddings_table_h[:, :sample_head_dim]
            self.sample_embeddings_table_v = self.embeddings_table_v[:, :sample_head_dim]
        else:
            new_v1 = full_v_out[..., :sample_head_dim_prev].detach()
            new_v2 = full_v_out[..., sample_head_dim_prev:sample_head_dim]
            new_v3 = full_v_out[..., sample_head_dim:self.super_embed_dim].detach()
            self.v1 = nn.Parameter(new_v1, requires_grad=False)
            self.v2 = nn.Parameter(new_v2, requires_grad=True)
            self.v3 = nn.Parameter(new_v3, requires_grad=False)

            new_h1 = full_h_out[..., :sample_head_dim_prev].detach()
            new_h2 = full_h_out[..., sample_head_dim_prev:sample_head_dim]
            new_h3 = full_h_out[..., sample_head_dim:self.super_embed_dim].detach()
            self.h1 = nn.Parameter(new_h1, requires_grad=False)
            self.h2 = nn.Parameter(new_h2, requires_grad=True)
            self.h3 = nn.Parameter(new_h3, requires_grad=False)

            frozen_h = self.embeddings_table_h[:, :sample_head_dim_prev].detach()
            trainable_h = self.embeddings_table_h[:, sample_head_dim_prev:sample_head_dim]
            self.sample_embeddings_table_h = torch.cat([frozen_h, trainable_h], dim=1)

            frozen_v = self.embeddings_table_v[:, :sample_head_dim_prev].detach()
            trainable_v = self.embeddings_table_v[:, sample_head_dim_prev:sample_head_dim]
            self.sample_embeddings_table_v = torch.cat([frozen_v, trainable_v], dim=1)


    def calc_sampled_param_num(self):
        return self.sample_embeddings_table_h.numel() + self.sample_embeddings_table_v.numel()

    @property
    def embeddings_table_v(self):
        # frozen bias가 없으면, bias1은 빈 텐서.
        if self.v1.numel() == 0:
            return self.v2
        else:
            return torch.cat([self.v1, self.v2], dim=1)

    @property
    def embeddings_table_h(self):
        # frozen bias가 없으면, bias1은 빈 텐서.
        if self.h1.numel() == 0:
            return self.h2
        else:
            return torch.cat([self.h1, self.h2], dim=1)

    def forward(self, length_q, length_k):
        # remove the first cls token distance computation
        length_q = length_q - 1
        length_k = length_k - 1
        device = self.embeddings_table_v.device
        range_vec_q = torch.arange(length_q, device=device)
        range_vec_k = torch.arange(length_k, device=device)
        # compute the row and column distance
        distance_mat_v = (range_vec_k[None, :] // int(length_q ** 0.5 )  - range_vec_q[:, None] // int(length_q ** 0.5 ))
        distance_mat_h = (range_vec_k[None, :] % int(length_q ** 0.5 ) - range_vec_q[:, None] % int(length_q ** 0.5 ))
        # clip the distance to the range of [-max_relative_position, max_relative_position]
        distance_mat_clipped_v = torch.clamp(distance_mat_v, -self.max_relative_position, self.max_relative_position)
        distance_mat_clipped_h = torch.clamp(distance_mat_h, -self.max_relative_position, self.max_relative_position)

        # translate the distance from [1, 2 * max_relative_position + 1], 0 is for the cls token
        final_mat_v = distance_mat_clipped_v + self.max_relative_position + 1
        final_mat_h = distance_mat_clipped_h + self.max_relative_position + 1
        # pad the 0 which represent the cls token
        final_mat_v = torch.nn.functional.pad(final_mat_v, (1,0,1,0), "constant", 0)
        final_mat_h = torch.nn.functional.pad(final_mat_h, (1,0,1,0), "constant", 0)

        final_mat_v = final_mat_v.long()
        final_mat_h = final_mat_h.long()
        # get the embeddings with the corresponding distance
        embeddings = self.embeddings_table_v[final_mat_v] + self.embeddings_table_h[final_mat_h]
        # embeddings = self.sample_embeddings_table_v[final_mat_v] + self.sample_embeddings_table_h[final_mat_h]

        return embeddings

class AttentionSuper(nn.Module):
    def __init__(self, super_embed_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., normalization = False, relative_position = False,
                 num_patches = None, max_relative_position=14, scale=False, change_qkv = False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = super_embed_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.super_embed_dim = super_embed_dim

        self.fc_scale = scale
        self.change_qkv = change_qkv
        if change_qkv:
            self.qkv = qkv_super(super_embed_dim, 3 * super_embed_dim, bias=qkv_bias)
        else:
            self.qkv = LinearSuper(super_embed_dim, 3 * super_embed_dim, bias=qkv_bias)

        self.relative_position = relative_position
        if self.relative_position:
            self.rel_pos_embed_k = RelativePosition2D_super(super_embed_dim //num_heads, max_relative_position, super_embed_dim)
            self.rel_pos_embed_v = RelativePosition2D_super(super_embed_dim //num_heads, max_relative_position, super_embed_dim)
        self.max_relative_position = max_relative_position
        self.sample_qk_embed_dim = None
        self.sample_v_embed_dim = None
        self.sample_num_heads = None
        self.sample_scale = None
        self.sample_in_embed_dim = None

        self.sample_qk_embed_dim_prev = None
        self.sample_v_embed_dim_prev = None
        self.sample_num_heads_prev = None
        self.sample_scale_prev = None
        self.sample_in_embed_dim_prev = None

        self.proj = LinearSuper(super_embed_dim, super_embed_dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def set_sample_config(self, sample_q_embed_dim=None, sample_num_heads=None, sample_in_embed_dim=None,
                          sample_q_embed_dim_prev=None, sample_num_heads_prev=None, sample_in_embed_dim_prev=None):

        self.sample_in_embed_dim = sample_in_embed_dim
        self.sample_num_heads = sample_num_heads

        self.sample_in_embed_dim_prev = sample_in_embed_dim_prev
        self.sample_num_heads_prev = sample_num_heads_prev

        if not self.change_qkv:
            self.sample_qk_embed_dim = self.super_embed_dim
            self.sample_scale = (sample_in_embed_dim // self.sample_num_heads) ** -0.5

        else:
            self.sample_qk_embed_dim = sample_q_embed_dim
            self.sample_scale = (self.sample_qk_embed_dim // self.sample_num_heads) ** -0.5


        # sample_scale_prev는 사실 정의가 필요 없는거같음.
        if not self.change_qkv:
            # super_embed_dim이 None이면 sample_qk_embed_dim_prev는 None으로 설정
            self.sample_qk_embed_dim_prev = self.super_embed_dim if self.super_embed_dim is not None else None

            # sample_in_embed_dim_prev와 self.sample_num_heads_prev가 모두 None이 아니라면 계산, 아니면 None
            if sample_in_embed_dim_prev is not None and self.sample_num_heads_prev is not None:
                self.sample_scale_prev = (sample_in_embed_dim_prev // self.sample_num_heads_prev) ** -0.5
            else:
                self.sample_scale_prev = None

        else:
            # sample_q_embed_dim_prev가 None이면 sample_qk_embed_dim_prev는 None
            self.sample_qk_embed_dim_prev = sample_q_embed_dim_prev if sample_q_embed_dim_prev is not None else None

            # sample_qk_embed_dim_prev와 self.sample_num_heads_prev가 모두 None이 아니라면 계산, 아니면 None
            if self.sample_qk_embed_dim_prev is not None and self.sample_num_heads_prev is not None:
                self.sample_scale_prev = (self.sample_qk_embed_dim_prev // self.sample_num_heads_prev) ** -0.5
            else:
                self.sample_scale_prev = None


        self.qkv.set_sample_config(sample_in_dim=sample_in_embed_dim, sample_out_dim=3*self.sample_qk_embed_dim, sample_in_dim_prev=sample_in_embed_dim_prev, sample_out_dim_prev=(3*self.sample_qk_embed_dim_prev) if self.sample_qk_embed_dim_prev is not None else None)
        self.proj.set_sample_config(sample_in_dim=self.sample_qk_embed_dim, sample_out_dim=sample_in_embed_dim, sample_in_dim_prev=self.sample_qk_embed_dim_prev, sample_out_dim_prev=self.sample_in_embed_dim_prev)
        
        if sample_num_heads_prev is None:
            sample_num_heads_prev = sample_num_heads
        if self.sample_qk_embed_dim_prev is None:
            self.sample_qk_embed_dim_prev = self.sample_qk_embed_dim

        
        if self.relative_position:
            self.rel_pos_embed_k.set_sample_config(self.sample_qk_embed_dim // sample_num_heads, self.sample_qk_embed_dim_prev // sample_num_heads_prev)
            self.rel_pos_embed_v.set_sample_config(self.sample_qk_embed_dim // sample_num_heads, self.sample_qk_embed_dim_prev // sample_num_heads_prev)
    def calc_sampled_param_num(self):

        return 0
    def get_complexity(self, sequence_length):
        total_flops = 0
        total_flops += self.qkv.get_complexity(sequence_length)
        # attn
        total_flops += sequence_length * sequence_length * self.sample_qk_embed_dim
        # x
        total_flops += sequence_length * sequence_length * self.sample_qk_embed_dim
        total_flops += self.proj.get_complexity(sequence_length)
        if self.relative_position:
            total_flops += self.max_relative_position * sequence_length * sequence_length + sequence_length * sequence_length / 2.0
            total_flops += self.max_relative_position * sequence_length * sequence_length + sequence_length * self.sample_qk_embed_dim / 2.0
        return total_flops

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.sample_num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.sample_scale
        if self.relative_position:
            r_p_k = self.rel_pos_embed_k(N, N)
            attn = attn + (q.permute(2, 0, 1, 3).reshape(N, self.sample_num_heads * B, -1) @ r_p_k.transpose(2, 1)) \
                .transpose(1, 0).reshape(B, self.sample_num_heads, N, N) * self.sample_scale

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1,2).reshape(B, N, -1)
        if self.relative_position:
            r_p_v = self.rel_pos_embed_v(N, N)
            attn_1 = attn.permute(2, 0, 1, 3).reshape(N, B * self.sample_num_heads, -1)
            # The size of attention is (B, num_heads, N, N), reshape it to (N, B*num_heads, N) and do batch matmul with
            # the relative position embedding of V (N, N, head_dim) get shape like (N, B*num_heads, head_dim). We reshape it to the
            # same size as x (B, num_heads, N, hidden_dim)
            x = x + (attn_1 @ r_p_v).transpose(1, 0).reshape(B, self.sample_num_heads, N, -1).transpose(2,1).reshape(B, N, -1)

        if self.fc_scale:
            x = x * (self.super_embed_dim / self.sample_qk_embed_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
