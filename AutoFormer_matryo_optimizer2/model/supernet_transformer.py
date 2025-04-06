import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.module.Linear_super import LinearSuper
from model.module.layernorm_super import LayerNormSuper
from model.module.multihead_super import AttentionSuper
from model.module.embedding_super import PatchembedSuper
from model.utils import trunc_normal_
from model.utils import DropPath
import numpy as np

def gelu(x: torch.Tensor) -> torch.Tensor:
    if hasattr(torch.nn.functional, 'gelu'):
        return torch.nn.functional.gelu(x.float()).type_as(x)
    else:
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class Vision_TransformerSuper(nn.Module):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., pre_norm=True, scale=False, gp=False, relative_position=False, change_qkv=False, abs_pos = True, max_relative_position=14, choices=None):
        super(Vision_TransformerSuper, self).__init__()
        # the configs of super arch
        self.super_embed_dim = embed_dim
        # self.super_embed_dim = args.embed_dim
        self.super_mlp_ratio = mlp_ratio
        self.super_layer_num = depth
        self.super_num_heads = num_heads
        self.super_dropout = drop_rate
        self.super_attn_dropout = attn_drop_rate
        self.num_classes = num_classes
        self.pre_norm=pre_norm
        self.scale=scale
        self.patch_embed_super = PatchembedSuper(img_size=img_size, patch_size=patch_size,
                                                 in_chans=in_chans, embed_dim=embed_dim, choices=choices)
        self.gp = gp

        # configs for the sampled subTransformer
        self.sample_embed_dim = None
        self.sample_mlp_ratio = None
        self.sample_layer_num = None
        self.sample_num_heads = None
        self.sample_dropout = None
        self.sample_output_dim = None

        # configs_prev for the sampled subTransformer
        self.sample_embed_dim_prev = None
        self.sample_mlp_ratio_prev = None
        self.sample_layer_num_prev = None
        self.sample_num_heads_prev = None
        self.sample_dropout_prev = None
        self.sample_out_dim_prev = None
        self.sample_attn_dropout_prev = None
        self.sample_output_dim_prev = None

        self.choices = choices


        self.blocks = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        for i in range(depth):
            self.blocks.append(TransformerEncoderLayer(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                                       qkv_bias=qkv_bias, qk_scale=qk_scale, dropout=drop_rate,
                                                       attn_drop=attn_drop_rate, drop_path=dpr[i],
                                                       pre_norm=pre_norm, scale=self.scale,
                                                       change_qkv=change_qkv, relative_position=relative_position,
                                                       max_relative_position=max_relative_position, choices=choices))

        # parameters for vision transformer
        num_patches = self.patch_embed_super.num_patches

        self.abs_pos = abs_pos
        if self.abs_pos:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
            trunc_normal_(self.pos_embed, std=.02)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        trunc_normal_(self.cls_token, std=.02)

        # self.pos_drop = nn.Dropout(p=drop_rate)
        if self.pre_norm:
            self.norm = LayerNormSuper(super_embed_dim=embed_dim, choices=choices)

        # classifier head
        self.head = LinearSuper(embed_dim, num_classes, choices=choices, name="head") if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # def _init_weights(self, m):
    #     def is_bias(name):
    #         return 'bias' in name

    #     def init_param(p, is_bias):
    #         if is_bias:
    #             nn.init.constant_(p, 0)
    #         else:
    #             trunc_normal_(p, std=.02)

    #     # 기본 nn.Linear와 nn.LayerNorm 초기화
    #     if isinstance(m, nn.Linear):
    #         if hasattr(m, 'weight') and m.weight is not None:
    #             trunc_normal_(m.weight, std=.02)
    #         if hasattr(m, 'bias') and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LayerNorm):
    #         if hasattr(m, 'weight') and m.weight is not None:
    #             nn.init.constant_(m.weight, 1.0)
    #         if hasattr(m, 'bias') and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)

    #     # 사용자 정의 weight / bias 파라미터 초기화
    #     for name in ['w1', 'w2', 'w3', 'w4', 'w5', 
    #                 'bias1', 'bias2', 'bias3', 
    #                 # 'v1', 'v2', 'v3',
    #                 # 'h1', 'h2', 'h3'
    #                 ]:
    #         if hasattr(m, name):
    #             param = getattr(m, name)
    #             if param is not None:
    #                 init_param(param, is_bias(name))

    #     # proj 안의 파라미터도 마찬가지로 처리
    #     if hasattr(m, 'proj'):
    #         proj = m.proj
    #         for name in ['w1', 'w2', 'w3', 'bias1', 'bias2', 'bias3']:
    #             if hasattr(proj, name):
    #                 param = getattr(proj, name)
    #                 if param is not None:
    #                     init_param(param, is_bias(name))

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'rel_pos_embed'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def set_sample_config(self, config: dict, config_prev: dict = None): # 이 안에 문제가 있다.
        self.sample_embed_dim = config['embed_dim']
        self.sample_mlp_ratio = config['mlp_ratio']
        self.sample_layer_num = config['layer_num']
        self.sample_num_heads = config['num_heads']
        self.sample_embed_dim_prev = None
        self.sample_mlp_ratio_prev = None
        self.sample_layer_num_prev = None
        self.sample_num_heads_prev = None

        if config_prev is not None:
            self.sample_embed_dim_prev = config_prev['embed_dim']
            self.sample_mlp_ratio_prev = config_prev['mlp_ratio']
            self.sample_layer_num_prev = config_prev['layer_num']
            self.sample_num_heads_prev = config_prev['num_heads']


        self.sample_dropout = calc_dropout(self.super_dropout, self.sample_embed_dim[0], self.super_embed_dim)
        self.patch_embed_super.set_sample_config(self.sample_embed_dim[0], self.sample_embed_dim_prev[0] if self.sample_embed_dim_prev is not None else None)
        self.sample_output_dim = [out_dim for out_dim in self.sample_embed_dim[1:]] + [self.sample_embed_dim[-1]]

        self.sample_dropout_prev = (
            calc_dropout(self.super_dropout, self.sample_embed_dim_prev[0], self.super_embed_dim)
            if self.sample_embed_dim_prev is not None and self.sample_embed_dim_prev[0] is not None
            else None)

        self.sample_output_dim_prev = [out_dim for out_dim in self.sample_embed_dim_prev[1:]] + [self.sample_embed_dim_prev[-1]]  if self.sample_embed_dim_prev is not None else None

        for i, blocks in enumerate(self.blocks):
            # not exceed sample layer number
            if i < self.sample_layer_num:
                sample_dropout = calc_dropout(self.super_dropout, self.sample_embed_dim[i], self.super_embed_dim)
                sample_attn_dropout = calc_dropout(self.super_attn_dropout, self.sample_embed_dim[i], self.super_embed_dim)

                sample_dropout_prev = (
                    calc_dropout(self.super_dropout, self.sample_embed_dim_prev[i], self.super_embed_dim)
                    if self.sample_embed_dim_prev is not None and self.sample_embed_dim_prev[i] is not None
                    else None
                )
                sample_attn_dropout_prev = (
                    calc_dropout(self.super_attn_dropout, self.sample_embed_dim_prev[i], self.super_embed_dim)
                    if self.sample_embed_dim_prev is not None and self.sample_embed_dim_prev[i] is not None
                    else None
                )
                blocks.set_sample_config(
                    is_identity_layer=False,
                    sample_embed_dim=self.sample_embed_dim[i],
                    sample_mlp_ratio=self.sample_mlp_ratio[i],
                    sample_num_heads=self.sample_num_heads[i],
                    sample_dropout=sample_dropout,
                    sample_attn_dropout=sample_attn_dropout,
                    sample_out_dim=self.sample_output_dim[i],
                    sample_embed_dim_prev=self.sample_embed_dim_prev[i] if self.sample_embed_dim_prev is not None else None,
                    sample_mlp_ratio_prev=self.sample_mlp_ratio_prev[i] if self.sample_mlp_ratio_prev is not None else None,
                    sample_num_heads_prev=self.sample_num_heads_prev[i] if self.sample_num_heads_prev is not None else None,
                    sample_dropout_prev=sample_dropout_prev,
                    sample_attn_dropout_prev=sample_attn_dropout_prev,
                    sample_out_dim_prev=self.sample_output_dim_prev[i] if self.sample_output_dim_prev is not None else None,
                )
            # exceeds sample layer number
            else:
                blocks.set_sample_config(is_identity_layer=True)
        if self.pre_norm:
            self.norm.set_sample_config(self.sample_embed_dim[-1], self.sample_embed_dim_prev[-1] if self.sample_embed_dim_prev is not None else None)
        self.head.set_sample_config(self.sample_embed_dim[-1], self.num_classes, self.sample_embed_dim_prev[-1] if self.sample_embed_dim_prev is not None else None, self.num_classes)

    def get_sampled_params_numel(self, config):
        self.set_sample_config(config)
        numels = []
        for name, module in self.named_modules():
            if hasattr(module, 'calc_sampled_param_num'):
                if name.split('.')[0] == 'blocks' and int(name.split('.')[1]) >= config['layer_num']:
                    continue
                numels.append(module.calc_sampled_param_num())

        return sum(numels) + self.sample_embed_dim[0]* (2 +self.patch_embed_super.num_patches)
    def get_complexity(self, sequence_length):
        total_flops = 0
        total_flops += self.patch_embed_super.get_complexity(sequence_length)
        total_flops += np.prod(self.pos_embed[..., :self.sample_embed_dim[0]].size()) / 2.0
        for blk in self.blocks:
            total_flops +=  blk.get_complexity(sequence_length+1)
        total_flops += self.head.get_complexity(sequence_length+1)
        return total_flops
    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed_super(x)
        cls_tokens = self.cls_token[..., :self.sample_embed_dim[0]].expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        if self.abs_pos:
            x = x + self.pos_embed[..., :self.sample_embed_dim[0]]

        x = F.dropout(x, p=self.sample_dropout, training=self.training)

        # start_time = time.time()
        for blk in self.blocks:
            x = blk(x)
        # print(time.time()-start_time)
        if self.pre_norm:
            x = self.norm(x)

        if self.gp:
            return torch.mean(x[:, 1:] , dim=1)

        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """Encoder layer block.

    Args:
        args (argparse.Namespace): parsed command-line arguments which
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, dropout=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, pre_norm=True, scale=False,
                 relative_position=False, change_qkv=False, max_relative_position=14, choices=None):
        super().__init__()

        # the configs of super arch of the encoder, three dimension [embed_dim, mlp_ratio, and num_heads]
        self.super_embed_dim = dim
        self.super_mlp_ratio = mlp_ratio
        self.super_ffn_embed_dim_this_layer = int(mlp_ratio * dim)
        self.super_num_heads = num_heads
        self.normalize_before = pre_norm
        self.super_dropout = attn_drop
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.scale = scale
        self.relative_position = relative_position
        # self.super_activation_dropout = getattr(args, 'activation_dropout', 0)

        # the configs of current sampled arch
        self.sample_embed_dim = None
        self.sample_mlp_ratio = None
        self.sample_ffn_embed_dim_this_layer = None
        self.sample_num_heads_this_layer = None
        self.sample_scale = None
        self.sample_dropout = None
        self.sample_attn_dropout = None

        # pref value
        self.sample_embed_dim_prev = None
        self.sample_mlp_ratio_prev = None
        self.sample_num_heads_prev = None
        self.sample_dropout_prev = None
        self.sample_attn_dropout_prev = None
        self.sample_out_dim_prev = None
        self.sample_ffn_embed_dim_this_layer_prev = None

        self.choices = choices

        self.is_identity_layer = None
        self.attn = AttentionSuper(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
            proj_drop=dropout, scale=self.scale, relative_position=self.relative_position, change_qkv=change_qkv,
            max_relative_position=max_relative_position, choices=choices
        )

        self.attn_layer_norm = LayerNormSuper(self.super_embed_dim, choices=choices)
        self.ffn_layer_norm = LayerNormSuper(self.super_embed_dim, choices=choices)
        # self.dropout = dropout
        self.activation_fn = gelu
        # self.normalize_before = args.encoder_normalize_before

        self.fc1 = LinearSuper(super_in_dim=self.super_embed_dim, super_out_dim=self.super_ffn_embed_dim_this_layer, choices=choices, name="fc1")
        self.fc2 = LinearSuper(super_in_dim=self.super_ffn_embed_dim_this_layer, super_out_dim=self.super_embed_dim, choices=choices, name="fc2")


    def set_sample_config(self, is_identity_layer, sample_embed_dim=None, sample_mlp_ratio=None, sample_num_heads=None, sample_dropout=None, sample_attn_dropout=None, sample_out_dim=None, sample_embed_dim_prev=None,
                      sample_mlp_ratio_prev=None,
                      sample_num_heads_prev=None,
                      sample_dropout_prev=None,
                      sample_attn_dropout_prev=None,
                      sample_out_dim_prev=None): # 여긴 문제 없음 

        if is_identity_layer:
            self.is_identity_layer = True
            return

        self.is_identity_layer = False

        self.sample_embed_dim = sample_embed_dim
        self.sample_out_dim = sample_out_dim
        self.sample_mlp_ratio = sample_mlp_ratio
        self.sample_ffn_embed_dim_this_layer = int(sample_embed_dim*sample_mlp_ratio)
        self.sample_num_heads_this_layer = sample_num_heads

        self.sample_dropout = sample_dropout
        self.sample_attn_dropout = sample_attn_dropout

        # 저장된 _prev 인자들 (추후 가중치 분리 등의 freeze 로직에 활용할 수 있음)
        
        # 악의 원인
        # if sample_embed_dim_prev is None:
        #     sample_embed_dim_prev = sample_embed_dim
        # if sample_mlp_ratio_prev is None:
        #     sample_mlp_ratio_prev = sample_mlp_ratio

        self.sample_embed_dim_prev = sample_embed_dim_prev
        self.sample_mlp_ratio_prev = sample_mlp_ratio_prev
        self.sample_num_heads_prev = sample_num_heads_prev
        self.sample_dropout_prev = sample_dropout_prev
        self.sample_attn_dropout_prev = sample_attn_dropout_prev
        self.sample_out_dim_prev = sample_out_dim_prev

        if sample_embed_dim_prev is None and sample_mlp_ratio_prev is None:
            self.sample_ffn_embed_dim_this_layer_prev = None
        else:
            embed_dim_val = sample_embed_dim_prev if sample_embed_dim_prev is not None else sample_embed_dim
            mlp_ratio_val = sample_mlp_ratio_prev if sample_mlp_ratio_prev is not None else sample_mlp_ratio
            self.sample_ffn_embed_dim_this_layer_prev = int(embed_dim_val * mlp_ratio_val)


        self.attn_layer_norm.set_sample_config(sample_embed_dim=self.sample_embed_dim, sample_embed_dim_prev=self.sample_embed_dim_prev)

        self.attn.set_sample_config(sample_q_embed_dim=self.sample_num_heads_this_layer*64, sample_num_heads=self.sample_num_heads_this_layer, sample_in_embed_dim=self.sample_embed_dim,
                                    sample_q_embed_dim_prev=(self.sample_num_heads_prev * 64) if self.sample_num_heads_prev is not None else None, sample_num_heads_prev=self.sample_num_heads_prev, sample_in_embed_dim_prev=self.sample_embed_dim_prev)

        self.fc1.set_sample_config(sample_in_dim=self.sample_embed_dim, sample_out_dim=self.sample_ffn_embed_dim_this_layer,
                                   sample_in_dim_prev=self.sample_embed_dim_prev, sample_out_dim_prev=self.sample_ffn_embed_dim_this_layer_prev)
        self.fc2.set_sample_config(sample_in_dim=self.sample_ffn_embed_dim_this_layer, sample_out_dim=self.sample_out_dim,
                                   sample_in_dim_prev=self.sample_ffn_embed_dim_this_layer_prev, sample_out_dim_prev=self.sample_out_dim_prev)

        self.ffn_layer_norm.set_sample_config(sample_embed_dim=self.sample_embed_dim, sample_embed_dim_prev=self.sample_embed_dim_prev)


    def forward(self, x):
        """
        Args:
            x (Tensor): input to the layer of shape `(batch, patch_num , sample_embed_dim)`

        Returns:
            encoded output of shape `(batch, patch_num, sample_embed_dim)`
        """
        if self.is_identity_layer:
            return x

        # compute attn
        # start_time = time.time()

        residual = x
        x = self.maybe_layer_norm(self.attn_layer_norm, x, before=True)
        x = self.attn(x)
        x = F.dropout(x, p=self.sample_attn_dropout, training=self.training)
        x = self.drop_path(x)
        x = residual + x
        x = self.maybe_layer_norm(self.attn_layer_norm, x, after=True)
        # print("attn :", time.time() - start_time)
        # compute the ffn
        # start_time = time.time()
        residual = x
        x = self.maybe_layer_norm(self.ffn_layer_norm, x, before=True)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.sample_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.sample_dropout, training=self.training)
        if self.scale:
            x = x * (self.super_mlp_ratio / self.sample_mlp_ratio)
        x = self.drop_path(x)
        x = residual + x
        x = self.maybe_layer_norm(self.ffn_layer_norm, x, after=True)
        # print("ffn :", time.time() - start_time)
        return x

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x
    def get_complexity(self, sequence_length):
        total_flops = 0
        if self.is_identity_layer:
            return total_flops
        total_flops += self.attn_layer_norm.get_complexity(sequence_length+1)
        total_flops += self.attn.get_complexity(sequence_length+1)
        total_flops += self.ffn_layer_norm.get_complexity(sequence_length+1)
        total_flops += self.fc1.get_complexity(sequence_length+1)
        total_flops += self.fc2.get_complexity(sequence_length+1)
        return total_flops

def calc_dropout(dropout, sample_embed_dim, super_embed_dim):
    return dropout * 1.0 * sample_embed_dim / super_embed_dim





