import argparse
import torch
import matplotlib.pyplot as plt
from timm.utils.model import unwrap_model
import os
from model_matryo.supernet_transformer import Vision_TransformerSuper as Vision_TransformerSuper_Matryo
from model.supernet_transformer import Vision_TransformerSuper
import sys
import warnings
import numpy as np


def get_args_parser():
    parser = argparse.ArgumentParser('AutoFormer training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    # config file
    parser.add_argument('--cfg',help='experiment configure file name',required=True,type=str)

    # custom parameters
    parser.add_argument('--platform', default='pai', type=str, choices=['itp', 'pai', 'aml'],
                        help='Name of model to train')
    parser.add_argument('--teacher_model', default='', type=str,
                        help='Name of teacher model to train')
    parser.add_argument('--relative_position', action='store_true')
    parser.add_argument('--gp', action='store_true')
    parser.add_argument('--change_qkv', action='store_true')
    parser.add_argument('--max_relative_position', type=int, default=14, help='max distance in relative position embedding')

    # Model parameters
    parser.add_argument('--model', default='', type=str, metavar='MODEL',
                        help='Name of model to train')
    # AutoFormer config
    parser.add_argument('--mode', type=str, default='super', choices=['super', 'retrain'], help='mode of AutoFormer')
    parser.add_argument('--input-size', default=224, type=int)
    parser.add_argument('--patch_size', default=16, type=int)

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.0, metavar='PCT',
                        help='Drop path rate (default: 0.1)') # 0.1 -> 0.0
    parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                        help='Drop block rate (default: None)')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    # parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')
    parser.add_argument('--rpe_type', type=str, default='bias', choices=['bias', 'direct'])
    parser.add_argument('--post_norm', action='store_true')
    parser.add_argument('--no_abs_pos', action='store_true')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--lr-power', type=float, default=1.0,
                        help='power of the polynomial lr scheduler')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')


    parser.set_defaults(repeated_aug=True)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Dataset parameters
    parser.add_argument('--data-path', default='./data/imagenet/', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default='./',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='tcp://localhost:2040', help='url used to set up distributed training')
    parser.add_argument('--save_checkpoint_path', default='', help='save checkpoint to the path')
    parser.add_argument('--save_log_path', default='', help='save log file to the path')
    parser.add_argument('--interval', default=1, type=int, help='interval of reusing top-k subnet searched by the sn indicator')

    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--no-amp', action='store_false', dest='amp')
    parser.set_defaults(amp=True)


    return parser

def main(args):

    # UserWarning 무시
    warnings.filterwarnings("ignore", category=UserWarning)

    sys.stdout = open("check_layer_output.log", 'w')
    sys.stderr = sys.stdout

    print(args)

    # ===== 모델 정의 =====
    choices = {
        'num_heads': [3, 4],
        'mlp_ratio': [3.5, 4.0],
        'embed_dim': [192, 216, 240],
        'depth': [12, 13, 14]
    }
    model = Vision_TransformerSuper_Matryo(
        img_size=224, patch_size=16, embed_dim=256, depth=14,
        num_heads=4, mlp_ratio=4., qkv_bias=True, drop_rate=0.0,
        drop_path_rate=0.1, gp=True, num_classes=1000,
        max_relative_position=14, relative_position=True,
        change_qkv=True, abs_pos=True, choices=choices
    )
    # model = Vision_TransformerSuper(
    #     img_size=224, patch_size=16, embed_dim=256, depth=14,
    #     num_heads=4, mlp_ratio=4., qkv_bias=True, drop_rate=0.0,
    #     drop_path_rate=0.1, gp=True, num_classes=1000,
    #     max_relative_position=14, relative_position=True,
    #     change_qkv=True, abs_pos=True
    #     # , choices=choices
    # )

    # 체크포인트 로드
    # ckpt_path = '/OUTPUT_PATH/checkpoint_original_check_only192_original_optimizer-epoch480-24.pth'  # 여기에 .pth 파일 경로를 입력하세요
    # ckpt_path = '/OUTPUT_PATH/checkpoint-original-25.pth'
    # ckpt_path = '/OUTPUT_PATH/checkpoint_original_check_only192_original_optimizer-epoch480-matryo_load_matryo_216_param-fc-wb-gaus-no-share-1e-4-decay005-0.pth'
    # ckpt_path = '/OUTPUT_PATH/checkpoint_original_check_only192-no-share-2e-4-decay0001-BC-3.pth'
    ckpt_path = '/OUTPUT_PATH/checkpoint-lr0002-feature-align-all-2.pth'
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt['model'], strict=False)

    model.eval()

    # 저장 경로 설정
    output_dir = './layer_feature_map_heatmaps_A_BC_feature_align/'
    os.makedirs(output_dir, exist_ok=True)

    # ===== Hook =====
    feature_outputs = {}

    def get_hook(name):
        def hook(module, input, output):
            feature_outputs[name] = output.detach()
        return hook

    # 필요한 모듈에 Hook 걸기
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv2d):
            hooks.append(module.register_forward_hook(get_hook(name)))

    # ===== 데이터 한 장 가져오기 =====
    from lib.datasets import build_dataset

    dataset_val, _ = build_dataset(is_train=False, args=args)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=1, shuffle=False, num_workers=4
    )

    dummy_input, _ = next(iter(data_loader_val))  # input만 가져옴
    dummy_input = dummy_input.to('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

    ######
    model_module = unwrap_model(model)

    # Vision_TransformerSuper에 sample config 설정
    sample_config = {
        'layer_num': 14,
        'embed_dim': [240] * 14,  # 최댓값
        'mlp_ratio': [4.0] * 14,
        'num_heads': [4] * 14,
    }
    model_module.set_sample_config(config=sample_config)

    # + 추가로 patch_embed_super 내부에도 수동 세팅
    if hasattr(model_module, 'patch_embed_super'):
        model_module.patch_embed_super.set_sample_config(sample_embed_dim=240)


    # forward
    with torch.no_grad():
        model(dummy_input)

    # ===== 시각화 함수 =====
    def plot_heatmap(tensor, filename):
        arr = tensor.cpu().numpy()
        if arr.ndim == 4:  # (B, C, H, W)면
            arr = arr[0]  # 첫 배치만
            arr = arr.mean(axis=0)  # 채널 평균
        elif arr.ndim == 3:
            arr = arr.mean(axis=0)
        elif arr.ndim == 2:
            pass  # 그대로
        else:
            print(f"Skip: {filename} shape {arr.shape}")
            return
        plt.figure(figsize=(10, 8))
        plt.imshow(arr, aspect='auto', cmap='viridis')
        plt.colorbar()
        plt.title(filename)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    # =====================

    # ===== 꺾은선 그래프 그리는 함수 추가 =====
    def plot_linegraph(tensor, filename):
        arr = tensor.cpu().numpy()

        if arr.ndim == 4:  # (B, C, H, W)
            arr = arr[0]  # 첫 배치만
            arr = arr.mean(axis=0)  # 채널 평균 → (H, W)
        elif arr.ndim == 3:  # (B, S, F)
            arr = arr[0]  # 배치 첫 개
            arr = arr.mean(axis=0)  # 시퀀스 평균 → (F,)
        elif arr.ndim == 2:
            pass  # 그대로
        else:
            print(f"Skip (unsupported shape): {filename} shape {arr.shape}")
            return

        # 핵심 수정: 열 방향 평균 → x축 = W 그대로 유지
        if arr.ndim == 2:
            line = arr.mean(axis=0)  # axis=0 → 각 column(세로줄)의 평균 → x축 유지
        elif arr.ndim == 1:
            line = arr  # 1D면 그대로
        else:
            print(f"Skip (unsupported dimension): {filename} shape {arr.shape}")
            return

        # 그래프 그리기
        plt.figure(figsize=(10, 4))
        plt.plot(line)
        plt.title(filename)
        plt.xlabel('X-axis (position index)')
        plt.ylabel('Mean over Y (vertical avg)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    # ==================================

    def plot_histogram(tensor, filename, bin_size=5):
        arr = tensor.cpu().numpy()

        if arr.ndim == 4:
            arr = arr[0]
            arr = arr.mean(axis=0)
        elif arr.ndim == 3:
            arr = arr[0]
            arr = arr.mean(axis=0)
        elif arr.ndim == 2:
            pass
        else:
            print(f"Skip (unsupported shape): {filename} shape {arr.shape}")
            return

        if arr.ndim == 2:
            line = arr.mean(axis=0)
        elif arr.ndim == 1:
            line = arr
        else:
            print(f"Skip (unsupported dimension): {filename} shape {arr.shape}")
            return

        W = line.shape[0]
        num_bins = W // bin_size
        binned = line[:num_bins * bin_size].reshape(num_bins, bin_size).mean(axis=1)

        # ✅ shift & remember min
        min_val = binned.min()
        binned_shifted = binned - min_val

        # ✅ draw shifted histogram
        plt.figure(figsize=(10, 4))
        plt.bar(np.arange(num_bins) * bin_size, binned_shifted, width=bin_size, align='edge')
        plt.title(filename)
        plt.xlabel(f'X-axis (grouped every {bin_size})')
        plt.ylabel('Feature Value')

        # ✅ relabel y-axis with original values
        yticks = plt.yticks()[0]
        ytick_labels = [f'{tick + min_val:.2f}' for tick in yticks]
        plt.yticks(yticks, ytick_labels)

        plt.tight_layout()
        plt.savefig(filename)
        plt.close()


    # ===== 저장 =====
    for name, feat in feature_outputs.items():
        save_path = os.path.join(output_dir, f"{name.replace('.', '_')}_output.png")

        print(f"[Saving Feature Map] {name} | shape: {feat.shape}")
        
        plot_heatmap(feat, save_path)

        # blocks_1 레이어에 대해서만 꺾은선 그래프 추가로 저장
        if name.startswith('blocks.1'):
            linegraph_path = os.path.join(output_dir, f"{name.replace('.', '_')}_linegraph.png")
            print(f"[Saving Line Graph] {name} | shape: {feat.shape}")
            plot_linegraph(feat, linegraph_path)

            histogram_path = os.path.join(output_dir, f"{name.replace('.', '_')}_histogram.png")
            plot_histogram(feat, histogram_path)

    # ===== Hook 해제 =====
    for h in hooks:
        h.remove()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('AutoFormer training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)