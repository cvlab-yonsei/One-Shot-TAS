#!/bin/bash

# UTF-8 환경 변수 설정 (UnicodeEncodeError 방지)
export PYTHONIOENCODING=utf-8
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

# python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train_small_sn.py --data-path '/data' --gp \
# --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-S.yaml --epochs 500 --warmup-epochs 20 \
# --resume '/OUTPUT_PATH/checkpoint_small_original_450.pth' --output /OUTPUT_PATH --batch-size 128 \
# --save_checkpoint_path 'checkpoint-sn-small-450ep-' --save_log_path './log/supernet_sn_small_450.log' --interval 1

# python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train_small_sn.py --data-path '/data' --gp \
# --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-S.yaml --epochs 500 --warmup-epochs 20 \
# --resume '/OUTPUT_PATH/checkpoint-sn-small-450ep-23.pth' --output /OUTPUT_PATH --batch-size 128 \
# --save_checkpoint_path 'checkpoint-sn-small-450ep-ing-' --save_log_path './log/supernet_sn_small_450_ing.log' --interval 1

# python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train_small_sn.py --data-path '/data' --gp \
# --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-S.yaml --epochs 500 --warmup-epochs 20 \
# --resume '/OUTPUT_PATH/checkpoint_small_original_450.pth' --output /OUTPUT_PATH --batch-size 128 \
# --save_checkpoint_path 'checkpoint-sn-small-450ep-droppath01-' --save_log_path './log/supernet_sn_small_450_droppath01.log' --interval 1

python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution.py --data-path '/data' --gp \
--change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-S.yaml --resume '/OUTPUT_PATH/checkpoint-sn-small-450ep-droppath01-25.pth' \
--min-param-limits 22 --param-limits 23 \
--log-file-path './log/search_sn-small-450ep-droppath01_6M.log'
