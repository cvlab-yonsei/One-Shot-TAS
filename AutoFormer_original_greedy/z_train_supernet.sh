#!/bin/bash

# UTF-8 환경 변수 설정 (UnicodeEncodeError 방지)
export PYTHONIOENCODING=utf-8
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

python -m torch.distributed.launch --nproc_per_node=8 --use_env z_supernet_train.py --data-path '/data' --gp \
--change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-S.yaml --epochs 500 --warmup-epochs 20 \
--output /OUTPUT_PATH --batch-size 128 \
--save_checkpoint_path 'checkpoint-z_original_auto_s_prenassmallaug' --save_log_path './log/supernet_z_original_auto_s_prenassmallaug.log' --interval 1

python3 -m torch.distributed.launch --nproc_per_node=8 --use_env z_evolution.py --data-path '/data' --gp \
--change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-S.yaml --resume '/OUTPUT_PATH/checkpoint-z_original_auto_s_prenassmallaug-25.pth' \
--min-param-limits 5 --param-limits 23 \
--log-file-path './log/search_z_original_auto_s_prenassmallaug_23M.log'
