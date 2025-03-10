#!/bin/bash

# UTF-8 환경 변수 설정 (UnicodeEncodeError 방지)
export PYTHONIOENCODING=utf-8
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_observation.py --data-path '/data' --gp \
--change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 500 --warmup-epochs 20 \
--resume '/OUTPUT_PATH/checkpoint-original-24.pth' --output /OUTPUT_PATH --batch-size 128 \
--save_checkpoint_path 'checkpoint-tiny-observation2-' --save_log_path './log/supernet_tiny_observation2.log' --interval 1
