import torch
import matplotlib.pyplot as plt
from timm.utils.model import unwrap_model
import os
from model_matryo.supernet_transformer import Vision_TransformerSuper as Vision_TransformerSuper_Matryo
import sys
import warnings

# UserWarning 무시
warnings.filterwarnings("ignore", category=UserWarning)

sys.stdout = open('check_parameter_checkpoint_original_check_only192_original_optimizer-epoch480-matryo_load_matryo_216-0.log', 'w')
sys.stderr = sys.stdout


# ==== 1. 모델 정의 ====
choices = {
    'num_heads': [3, 4],
    'mlp_ratio': [3.5, 4.0],
    'embed_dim': [192, 216, 240],
    'depth': [12, 13, 14]
}
model = Vision_TransformerSuper_Matryo(img_size=224,
                                    patch_size=16,
                                    embed_dim=256, depth=14,
                                    num_heads=4,mlp_ratio=4.,
                                    qkv_bias=True, drop_rate=0.0,
                                    drop_path_rate=0.1,
                                    gp=True,
                                    num_classes=1000,
                                    max_relative_position=14,
                                    relative_position=True,
                                    change_qkv=True, abs_pos=not False,
                                    choices=choices
                                    )

# ==== 2. 체크포인트 로드 ====
ckpt_path = '/OUTPUT_PATH/checkpoint_original_check_only192_original_optimizer-epoch480-matryo_load_matryo_216-0.pth'
ckpt = torch.load(ckpt_path, map_location='cpu')
model.load_state_dict(ckpt['model'], strict=False)

# ==== 3. 저장 경로 설정 ====
output_dir = './layer_weight_heatmaps/supernet-checkpoint_original_check_only192_original_optimizer-epoch480-matryo_load_matryo_216-0'
os.makedirs(output_dir, exist_ok=True)

# ==== 4. 시각화 함수 ====
def plot_heatmap(tensor, filename):
    arr = tensor.detach().cpu().numpy()
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    plt.figure(figsize=(10, 8))
    plt.imshow(arr, aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title(filename)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

config = {
    'layer_num': 14,
    'mlp_ratio': [4.0] * 14,
    'num_heads': [4] * 14,
    'embed_dim': [240] * 14
}

model_module = unwrap_model(model)
model_module.set_sample_config(config=config)

# 4. forward 수행 → samples가 실제로 사용됨
dummy_input = torch.randn(1, 3, 224, 224)
model(dummy_input)

# ==== 5. samples['weight'], ['bias']를 저장 ====
for name, module in model.named_modules():
    if hasattr(module, 'samples'):
        if 'weight' in module.samples and module.samples['weight'] is not None:
            print("weight!")
            w = module.samples['weight']
            plot_heatmap(w, os.path.join(output_dir, f'{name.replace(".", "_")}_samples_weight.png'))

        if 'bias' in module.samples and module.samples['bias'] is not None:
            print("bias!")
            b = module.samples['bias']
            plot_heatmap(b, os.path.join(output_dir, f'{name.replace(".", "_")}_samples_bias.png'))
