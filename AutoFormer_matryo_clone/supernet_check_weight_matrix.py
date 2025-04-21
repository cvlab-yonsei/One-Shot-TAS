import torch
import torch.nn as nn

from model.supernet_transformer import Vision_TransformerSuper
from model_matryo.supernet_transformer import Vision_TransformerSuper as Vision_TransformerSuper_Matryo
from timm.utils.model import unwrap_model
import matplotlib.pyplot as plt
import os

import sys
import warnings

# UserWarning 무시
warnings.filterwarnings("ignore", category=UserWarning)

sys.stdout = open('check_parameter_checkpoint_original_check_only192_original_optimizer-epoch480-matryo_load_matryo_216-0.log', 'w')
sys.stderr = sys.stdout

# 미리 학습된 모델 불러오기
# model_path = './GroundTruthGenerator/OUTPUT_save/config1_18/config18/checkpoint.pth'  # 여기에 .pth 파일 경로를 입력하세요


model_path = '/OUTPUT_PATH/checkpoint_original_check_only192_original_optimizer-epoch480-matryo_load_matryo_216-0.pth'  # 여기에 .pth 파일 경로를 입력하세요
# model_path = '/OUTPUT_PATH/checkpoint_original_check_only192_original_optimizer-epoch480-24.pth'  # 여기에 .pth 파일 경로를 입력하세요
# model_path = 'supernet-tiny.pth'

# net = Vision_TransformerSuper(img_size=224,
#                                     patch_size=16,
#                                     embed_dim=256, depth=14,
#                                     num_heads=4,mlp_ratio=4.,
#                                     qkv_bias=True, drop_rate=0.0,
#                                     drop_path_rate=0.1,
#                                     gp=True,
#                                     num_classes=1000,
#                                     max_relative_position=14,
#                                     relative_position=True,
#                                     change_qkv=True, abs_pos=not False)
# Vision_TransformerSuper 모델 정의 (해당 코드를 제공받은 코드로 대체하세요)
choices = {
            'num_heads': [3, 4],
            'mlp_ratio': [3.5, 4.0],
            'embed_dim': [192, 216, 240],
            'depth': [12, 13, 14]
        }

net = Vision_TransformerSuper_Matryo(img_size=224,
                                    patch_size=16,
                                    embed_dim=448, depth=14,
                                    num_heads=7, mlp_ratio=4.,
                                    qkv_bias=True, drop_rate=0.0,
                                    drop_path_rate=0.1,
                                    gp=True,
                                    num_classes=1000,
                                    max_relative_position=14,
                                    relative_position=True,
                                    change_qkv=True, abs_pos=not False, choices=choices)

# 체크포인트 파일 로드
checkpoint = torch.load(model_path, map_location='cpu')
print("resume from checkpoint: {}".format(model_path))

# 현재 체크포인트 파일에서 네트워크의 상태를 로드하기 위한 매핑 작업
new_checkpoint = {}
for key in checkpoint.keys():
    new_key = key.replace('blocks.blocks', 'blocks')  # 키 매핑 (예시)
    new_checkpoint[new_key] = checkpoint[key]

# 매핑된 상태 딕셔너리를 이용해 모델에 로드
net.load_state_dict(new_checkpoint, strict=False)

net.eval()  # 모델을 평가 모드로 전환

# 출력 디렉토리 설정
output_dir = './layer_weight_heatmaps/supernet-checkpoint_original_check_only192_original_optimizer-epoch480-matryo_load_matryo_216-0'
# output_dir = './layer_weight_heatmaps/supernet-checkpoint_original_check_only192_original_optimizer-epoch480-24'
# output_dir = './layer_weight_heatmaps/supernet-tiny-github'


os.makedirs(output_dir, exist_ok=True)

# net의 파라미터 수 계산 및 출력
total_params = sum(p.numel() for p in net.parameters())
print(f'Total number of parameters in the network: {total_params}')

def plot_heatmap(weight, filename):
    plt.figure(figsize=(10, 8))
    plt.imshow(weight, aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title(filename)
    plt.savefig(filename)
    plt.close()

# # 모든 weight matrix 히트맵 플롯 및 저장
# for name, layer in net.named_modules():
#     if hasattr(layer, 'weight') and layer.weight is not None:
#         weight = layer.weight.detach().cpu().numpy()
#         if weight.ndim == 1:
#             weight = weight.reshape(1, -1)  # 1차원 텐서를 2차원으로 변환
#         elif weight.ndim > 2:
#             weight = weight.reshape(weight.shape[0], -1)
#         filename = os.path.join(output_dir, f"{name.replace('.', '_')}_weight.png")
#         plot_heatmap(weight, filename)

#     if hasattr(layer, 'bias') and layer.bias is not None:
#         bias = layer.bias.detach().cpu().numpy().reshape(1, -1)  # 1차원 텐서를 2차원으로 변환
#         filename = os.path.join(output_dir, f"{name.replace('.', '_')}_bias.png")
#         plot_heatmap(bias, filename)

# # 'samples' 속성을 가진 레이어가 있다면 그 속성도 플롯 및 저장
# for name, layer in net.named_modules():
#     if hasattr(layer, 'samples'):
#         if 'weight' in layer.samples and layer.samples['weight'] is not None:
#             weight = layer.samples['weight'].detach().cpu().numpy()
#             if weight.ndim == 1:
#                 weight = weight.reshape(1, -1)  # 1차원 텐서를 2차원으로 변환
#             elif weight.ndim > 2:
#                 weight = weight.reshape(weight.shape[0], -1)
#             filename = os.path.join(output_dir, f"{name.replace('.', '_')}_samples_weight.png")
#             plot_heatmap(weight, filename)

# 1. 샘플 config 먼저 설정 (필수)
config = {
    'layer_num': 14,
    'mlp_ratio': [4.0] * 14,
    'num_heads': [4] * 14,
    'embed_dim': [240] * 14
}
# 1. checkpoint 로드 먼저
net.load_state_dict(new_checkpoint, strict=False)

# 2. 샘플 config 설정
model_module = unwrap_model(net)
model_module.set_sample_config(config=config)

# 3. 수동으로 sample_parameters()를 resample=True로 강제 호출
for m in net.modules():
    if hasattr(m, 'sample_parameters'):
        m.sample_parameters(resample=True)

# 4. forward 수행 → samples가 실제로 사용됨
dummy_input = torch.randn(1, 3, 224, 224)
net(dummy_input)

# 3. 이후 각 레이어에서 samples['weight'], ['bias']를 시각화
for name, layer in net.named_modules():
    if hasattr(layer, 'samples'):
        if 'weight' in layer.samples and layer.samples['weight'] is not None:
            weight = layer.samples['weight'].detach().cpu().numpy()
            if weight.ndim == 1:
                weight = weight.reshape(1, -1)
            elif weight.ndim > 2:
                weight = weight.reshape(weight.shape[0], -1)
            filename = os.path.join(output_dir, f"{name.replace('.', '_')}_samples_weight.png")
            plot_heatmap(weight, filename)

        if 'bias' in layer.samples and layer.samples['bias'] is not None:
            bias = layer.samples['bias'].detach().cpu().numpy().reshape(1, -1)
            filename = os.path.join(output_dir, f"{name.replace('.', '_')}_samples_bias.png")
            plot_heatmap(bias, filename)
