import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.fft import fft2, fftshift

def analyze_layer_heads(attention_maps):
    num_heads = attention_maps.shape[1]
    # print("num_heads : ", num_heads)
    head_scores = []

    for head in range(num_heads):
        averaged_magnitude_spectrum = np.zeros_like(attention_maps[0, head].cpu().numpy())

        # print('attention_maps length:', len(attention_maps))
        for attn_map in attention_maps:
            # Fourier 변환
            fft_map = fftshift(fft2(attn_map[head].cpu().numpy()))
            magnitude_spectrum = np.abs(fft_map)
            averaged_magnitude_spectrum += magnitude_spectrum
        
        averaged_magnitude_spectrum /= attention_maps.shape[0]

        head_scores.append(averaged_magnitude_spectrum)

    return head_scores

def calculate_head_variance(head_scores): # head 간 다양성 측정 
    # 각 head의 magnitude spectrum을 비교하여 variance 계산
    stacked_spectra = np.stack(head_scores, axis=0)  # (num_heads, H, W)
    variance = np.var(stacked_spectra, axis=0)  # (H, W)
    return np.mean(variance)  # 평균 variance 반환

def calculate_layer_score(attn_maps):
    head_scores = analyze_layer_heads(attn_maps)

    # Head 간의 variance 계산
    variance_score = calculate_head_variance(head_scores)
    
    return variance_score

def get_some_data(train_dataloader, num_batches, device):
    traindata = []
    dataloader_iter = iter(train_dataloader)
    for _ in range(num_batches):
        traindata.append(next(dataloader_iter))
    inputs  = torch.cat([a for a,_ in traindata])
    targets = torch.cat([b for _,b in traindata])
    inputs = inputs.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)
    return inputs, targets

def get_some_data_grasp(train_dataloader, num_classes, samples_per_class, device):
    datas = [[] for _ in range(num_classes)]
    labels = [[] for _ in range(num_classes)]
    mark = dict()
    dataloader_iter = iter(train_dataloader)
    while True:
        inputs, targets = next(dataloader_iter)
        for idx in range(inputs.shape[0]):
            x, y = inputs[idx:idx+1], targets[idx:idx+1]
            category = y.item()
            if len(datas[category]) == samples_per_class:
                mark[category] = True
                continue
            datas[category].append(x)
            labels[category].append(y)
        if len(mark) == num_classes:
            break

    x = torch.cat([torch.cat(_, 0) for _ in datas]).to(device) 
    y = torch.cat([torch.cat(_) for _ in labels]).view(-1).to(device)
    return x, y

def get_layer_metric_array(net, metric, mode):
    metric_array = []

    for layer in net.modules():
        if mode == 'channel' and hasattr(layer, 'dont_ch_prune'):
            continue
        if layer._get_name() == 'PatchembedSuper':
            metric_array.append(metric(layer))
        if isinstance(layer, nn.Linear) and layer.samples:
            metric_array.append(metric(layer))

    return metric_array

def get_layer_metric_array_dss(net, metric, mode, device, inputs):
    metric_array = []
    # metric_array_tmp = []

    # @torch.no_grad()
    # def linearize(net):
    #     signs = {}
    #     for name, param in net.state_dict().items():
    #         signs[name] = torch.sign(param)
    #         param.abs_()
    #     return signs

    # @torch.no_grad()
    # def nonlinearize(net, signs):
    #     for name, param in net.state_dict().items():
    #         if 'weight_mask' not in name:
    #             param.mul_(signs[name])

    # # Linearize the network weights
    # signs = linearize(net)

    # # Ensure gradients are zeroed before starting
    # net.zero_grad()

    # # Generate the input tensor filled with ones
    # input_dim = list(inputs[0, :].shape)
    # inputs = torch.ones([1] + input_dim).float().to(device)

    # # Forward pass
    # output = net.forward(inputs)

    # # Backward pass
    # torch.sum(output).backward()

    # for i, block in enumerate(net.blocks):
    #     # qkv에서 W_q와 W_k를 추출
    #     W_qkv = block.attn.qkv.weight  # Assuming qkv is present in attn module

    #     # qkv의 가중치를 3등분하여 W_q, W_k 추출
    #     d = W_qkv.shape[0] // 3
    #     print("grad true ? :", W_qkv.grad is not None)
    #     W_q_grad = W_qkv.grad[:d, :]
    #     W_k_grad = W_qkv.grad[d:2 * d, :]

    #     # W_q와 W_k에 대한 gradient를 이용하여 spectral norm을 계산
    #     if W_q_grad is not None and W_k_grad is not None:
    #         # Compute the spectral norm of the Jacobians
    #         spectral_norm_W_q = torch.svd(W_q_grad)[1].max().item()
    #         spectral_norm_W_k = torch.svd(W_k_grad)[1].max().item()

    #         # 두 Jacobian의 spectral norm 곱셈으로 Lipschitz constant 근사
    #         lipschitz_constant = spectral_norm_W_q * spectral_norm_W_k
    #         metric_array.append(lipschitz_constant)
    #     else:
    #         metric_array_tmp.append(0)  # gradient가 None일 경우, 0 추가 (예외 처리)

    # # Nonlinearize the network weights (revert changes)
    # nonlinearize(net, signs)

    # print('metric_array:', metric_array_tmp)    
    # print('length:', len(metric_array_tmp))
    
    # ######## attn map sequence diversity START
    
    # diversity_score_final = 0
    # layer_scores = []

    # for i, block in enumerate(net.blocks):
    #     attn_maps = block.attn.get_attention_maps()
        
    #     # print('attn_maps shape : ', attn_maps.shape)
    #     # if (i == 0):
    #     #     print(attn_maps)
    #     if attn_maps is not None:
    #         # diversity_score = average_pairwise_similarity(attn_maps)
    #         # diversity_score = average_pairwise_diversity_kl(attn_maps)
            
    #         # diversity_score_final += diversity_score
    #         layer_score = calculate_layer_score(attn_maps)
    #         layer_score *= 10000000
    #         layer_scores.append(layer_score)
    #         diversity_score_final += layer_score
    #         # print('diversity_score : ', diversity_score)
    #         # print(f"Layer {i} Attention Maps: {attn_maps.shape}")
    #         # super net layer지만 subnet layer에 해당안하면 Nonetype
    # # diversity_score_final *=  10000000 # 걍 내맘대로 곱함
    # print('diversity_score_final:', diversity_score_final)
    # print('layer_scores:', layer_scores)
    # metric_array.append(torch.tensor(diversity_score_final).to(device))
    
    # float_metric_array = [t.item() for t in metric_array]

    # # 결과 출력
    # print("metric :", float_metric_array)
    
    # ######## attn map sequence diversity END
    
    
    ######## original START

    for layer in net.modules():
        if mode == 'channel' and hasattr(layer, 'dont_ch_prune'):
            continue
        # if isinstance(layer, nn.Linear) and 'qkv' in layer._get_name() and layer.samples:
        # print(layer._get_name())
        # if 'Attention' in layer._get_name():
        #     val = metric(layer)
        #     if val is not None:
        #         metric_array.append(val)
        if isinstance(layer, nn.Linear) and layer.samples:
            metric_array.append(metric(layer))
        if isinstance(layer, torch.nn.Linear) and layer.out_features == 1000:
            metric_array.append(metric(layer))
    print("metric_length : ", len(metric_array))
    # 텐서를 float 값으로 변환하여 리스트에 저장
    # float_metric_array = [t.item() for t in metric_array]

    # 결과 출력
    # print("metric :", float_metric_array)
    
    ####### original END
    
    return metric_array

def reshape_elements(elements, shapes, device):
    def broadcast_val(elements, shapes):
        ret_grads = []
        for e,sh in zip(elements, shapes):
            ret_grads.append(torch.stack([torch.Tensor(sh).fill_(v) for v in e], dim=0).to(device))
        return ret_grads
    if type(elements[0]) == list:
        outer = []
        for e,sh in zip(elements, shapes):
            outer.append(broadcast_val(e,sh))
        return outer
    else:
        return broadcast_val(elements, shapes)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

