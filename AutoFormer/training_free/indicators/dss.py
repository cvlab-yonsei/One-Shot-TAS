import torch

from . import indicator
from ..p_utils import get_layer_metric_array_dss
import torch.nn as nn

import numpy as np
from scipy.fft import fft2, fftshift
import seaborn as sns
import matplotlib.pyplot as plt

@indicator('dss', bn=False, mode='param')
def compute_dss_per_weight(net, inputs, targets, mode, split_data=1, loss_fn=None):

    device = inputs.device

    @torch.no_grad()
    def linearize(net):
        signs = {}
        for name, param in net.state_dict().items():
            signs[name] = torch.sign(param)
            param.abs_()
        return signs

    @torch.no_grad()
    def nonlinearize(net, signs):
        for name, param in net.state_dict().items():
            if 'weight_mask' not in name:
                param.mul_(signs[name])

    signs = linearize(net)

    net.zero_grad()
    input_dim = list(inputs[0,:].shape)
    inputs = torch.ones([1] + input_dim).float().to(device)
    output = net.forward(inputs)
    torch.sum(output).backward()
    
    def plot_head_distribution(layer_scores, layer_name, filename):
        plt.figure(figsize=(12, 8))
        
        for i, magnitude_spectrum in enumerate(layer_scores):
            sns.kdeplot(magnitude_spectrum.flatten(), label=f'Head {i+1}', fill=True)
        
        plt.xlabel('Frequency')
        plt.ylabel('Density')
        plt.title(f'Layer {layer_name} - Head Frequency Distribution')
        plt.legend(loc='upper right')
        plt.savefig(filename)
        plt.close()

    def calculate_head_variance(head_scores):
        # 각 head의 magnitude spectrum을 비교하여 variance 계산
        stacked_spectra = np.stack(head_scores, axis=0)  # (num_heads, H, W)
        variance = np.var(stacked_spectra, axis=0)  # (H, W)
        return np.mean(variance)  # 평균 variance 반환

    def dss(layer):
        # if layer._get_name() == 'PatchembedSuper':
        #     if layer.sampled_weight.grad is not None:
        #         return torch.abs(layer.sampled_weight.grad * layer.sampled_weight)
        #     else:
        #         return torch.zeros_like(layer.sampled_weight)
        if isinstance(layer, nn.Linear) and 'qkv' in layer._get_name() and layer.samples or isinstance(layer,
                                                                                                       nn.Linear) and layer.out_features == layer.in_features and layer.samples:
            if layer.samples['weight'].grad is not None:
                return torch.abs(
                    torch.norm(layer.samples['weight'].grad, 'nuc') * torch.norm(layer.samples['weight'], 'nuc'))
            else:
                return torch.zeros_like(layer.samples['weight'])
        # if isinstance(layer,
        #               nn.Linear) and 'qkv' not in layer._get_name() and layer.out_features != layer.in_features and layer.out_features != 1000 and layer.samples:
        #     if layer.samples['weight'].grad is not None:
        #         return torch.abs(layer.samples['weight'].grad * layer.samples['weight'])
        #     else:
        #         return torch.zeros_like(layer.samples['weight'])
        # elif isinstance(layer, torch.nn.Linear) and layer.out_features == 1000:
        #     if layer.weight.grad is not None:
        #         return torch.abs(layer.weight.grad * layer.weight)
        #     else:
        #         return torch.zeros_like(layer.weight)
        else:
            return torch.tensor(0).to(device)
        
    def spectral_norm(layer):
        # if layer._get_name() == 'PatchembedSuper':
        #     if layer.sampled_weight.grad is not None:
        #         return torch.abs(layer.sampled_weight.grad * layer.sampled_weight)
        #     else:
        #         return torch.zeros_like(layer.sampled_weight)
        if isinstance(layer, nn.Linear) and 'qkv' in layer._get_name() and layer.samples:
            # qkv의 가중치를 3등분하여 W_q와 W_k 추출
            W_qkv = layer.samples['weight']
            if W_qkv is not None and W_qkv.grad is not None:
                d = W_qkv.shape[0] // 3
                W_q = W_qkv[:d, :]
                W_k = W_qkv[d:2 * d, :]

                # W_q * W_k^T 연산
                W_qkT = torch.matmul(W_q, W_k.T)

                # W_q * W_k^T의 spectral norm 계산
                spectral_norm = torch.svd(W_qkT)[1].max().item()
                
                return torch.tensor(spectral_norm).to(device)
            else:
                return None
            # if layer.samples['weight'].grad is not None:
            # return torch.abs(
            #     torch.norm(layer.samples['weight'], 'nuc'))
            # else:
            #     return torch.zeros_like(layer.samples['weight'])
        # if isinstance(layer,
        #               nn.Linear) and 'qkv' not in layer._get_name() and layer.out_features != layer.in_features and layer.out_features != 1000 and layer.samples:
        #     if layer.samples['weight'].grad is not None:
        #         return torch.abs(layer.samples['weight'].grad * layer.samples['weight'])
        #     else:
        #         return torch.zeros_like(layer.samples['weight'])
        # elif isinstance(layer, torch.nn.Linear) and layer.out_features == 1000:
        #     if layer.weight.grad is not None:
        #         return torch.abs(layer.weight.grad * layer.weight)
        #     else:
        #         return torch.zeros_like(layer.weight)
        else:
            # return torch.tensor(0).to(device)
            return None
        
    def attn_map_sequence_diversity(layer):
        # print(dir(layer))
        # print(type(layer))

        # layer의 QKV 가중치 추출
        qkv_weight = layer.qkv.weight.detach().cpu().numpy()
        num_heads = layer.num_heads
        print('num_heads : ', num_heads)
        print('qkv_weight.shape[1] : ', qkv_weight.shape[1])
        head_dim = qkv_weight.shape[1] // num_heads
        
        print('head_dim : ', head_dim)

        # Q, K 추출
        q_weight, k_weight, _ = np.split(qkv_weight, 3, axis=0)

        head_scores = []

        for head_idx in range(num_heads):
            # 특정 head의 Q, K 추출
            q_head = q_weight[:, head_idx * head_dim:(head_idx + 1) * head_dim]
            k_head = k_weight[:, head_idx * head_dim:(head_idx + 1) * head_dim]

            # Attention 계산
            attn_scores = np.dot(q_head, k_head.T)
            attn_probs = np.exp(attn_scores) / np.sum(np.exp(attn_scores), axis=-1, keepdims=True)  # Softmax 적용
            
            print('attn_probs.shape : ', attn_probs.shape)
            # CLS 토큰에 대한 첫 번째 열을 제외한 나머지 패치에 대한 attention map 추출
            head_attn_map = attn_probs[1:, 0]  # (N-1,)
            print('head_attn_map shape : ', head_attn_map.shape)

            # 14x14 패치로 reshape (N=196, 패치 크기)
            head_attn_map_2d = head_attn_map.reshape(14, 14)

            # Fourier 변환 및 스펙트럼 계산
            fft_map = fftshift(fft2(head_attn_map_2d))
            magnitude_spectrum = np.abs(fft_map)
            head_scores.append(magnitude_spectrum)

        # Head 간의 variance를 계산하여 diversity를 측정
        variance_score = calculate_head_variance(head_scores)

        # 그래프 저장
        # plot_head_distribution(head_scores, "Layer Name", "output_head_distribution.png")

        return variance_score
        
        
    def lipschitz_constant(layer):
        if isinstance(layer, nn.Linear) and 'qkv' in layer._get_name() and layer.samples:
            # qkv의 가중치를 3등분하여 W_q와 W_k 추출
            W_qkv = layer.samples['weight']
            if W_qkv is not None and W_qkv.grad is not None:
                d = W_qkv.shape[0] // 3
                W_q = W_qkv[:d, :]
                W_k = W_qkv[d:2 * d, :]

                # backward를 통해 구한 gradient
                W_q_grad = W_qkv.grad[:d, :]
                W_k_grad = W_qkv.grad[d:2 * d, :]

                # gradient가 None이 아닌지 확인
                if W_q_grad is not None and W_k_grad is not None:
                    # W_q와 W_k의 Jacobian 행렬의 spectral norm 계산
                    spectral_norm_W_q = torch.svd(W_q_grad)[1].max().item()
                    spectral_norm_W_k = torch.svd(W_k_grad)[1].max().item()

                    # 두 Jacobian의 spectral norm 곱셈으로 Lipschitz constant 계산
                    lipschitz_constant = spectral_norm_W_q * spectral_norm_W_k
                    return torch.tensor(lipschitz_constant).to(W_qkv.device)
                else:
                    # return torch.tensor(0.0).to(W_qkv.device)
                    return None
            else:
                # return torch.tensor(0.0).to(W_qkv.device)
                return None
        else:
            # return torch.tensor(0.0).to(layer.samples['weight'].device)
            return None
        
    # grads_abs = get_layer_metric_array_dss(net, spectral_norm, mode, device, inputs)
    # grads_abs = get_layer_metric_array_dss(net, attn_map_sequence_diversity, mode, device, inputs)
    grads_abs = get_layer_metric_array_dss(net, dss, mode, device, inputs)
    
    nonlinearize(net, signs)

    return grads_abs


