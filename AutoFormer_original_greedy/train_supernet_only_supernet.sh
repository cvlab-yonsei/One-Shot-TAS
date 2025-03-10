#!/bin/bash

# UTF-8 환경 변수 설정 (UnicodeEncodeError 방지)
export PYTHONIOENCODING=utf-8
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

# python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train_only_supernet.py --data-path '/data' --gp \
# --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 500 --warmup-epochs 20 \
# --output /OUTPUT_PATH --batch-size 128 \
# --save_checkpoint_path 'checkpoint-tiny-only-supernet-maximum240-' --save_log_path './log/supernet_tiny-only-supernet-maximum240.log' --interval 1

# python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution_only_supernet.py --data-path '/data' --gp \
# --change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --resume '/OUTPUT_PATH/checkpoint-tiny-only-supernet-maximum240-21.pth' \
# --min-param-limits 5 --param-limits 13 \
# --log-file-path './log/search_tiny-only-supernet240-minimum_pop1050.log'

# python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution_only_supernet.py --data-path '/data' --gp \
# --change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --resume '/OUTPUT_PATH/checkpoint-tiny-only-supernet-minimum-21.pth' \
# --min-param-limits 5 --param-limits 13 \
# --log-file-path './log/search_tiny-only-supernet192-minimum_pop1050.log'

# --min-param-limits 5 --param-limits 6 \


python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution_only_supernet.py --data-path '/data' --gp \
--change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --resume '/OUTPUT_PATH/checkpoint-tiny-only-supernet-minimum-21.pth' \
--min-param-limits 6 --param-limits 7 \
--log-file-path './log/search_tiny-only-supernet192-minimum_pop1050_7M.log'

python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution_only_supernet.py --data-path '/data' --gp \
--change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --resume '/OUTPUT_PATH/checkpoint-tiny-only-supernet-minimum-21.pth' \
--min-param-limits 7 --param-limits 8 \
--log-file-path './log/search_tiny-only-supernet192-minimum_pop1050_8M.log'

python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution_only_supernet.py --data-path '/data' --gp \
--change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --resume '/OUTPUT_PATH/checkpoint-tiny-only-supernet-minimum-21.pth' \
--min-param-limits 8 --param-limits 9 \
--log-file-path './log/search_tiny-only-supernet192-minimum_pop1050_9M.log'

python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution_only_supernet.py --data-path '/data' --gp \
--change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --resume '/OUTPUT_PATH/checkpoint-tiny-only-supernet-minimum-21.pth' \
--min-param-limits 9 --param-limits 10 \
--log-file-path './log/search_tiny-only-supernet192-minimum_pop1050_10M.log'

# python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train_only_supernet.py --data-path '/data' --gp \
# --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 500 --warmup-epochs 20 \
# --output /OUTPUT_PATH --batch-size 128 \
# --save_checkpoint_path 'checkpoint-tiny-only-supernet-minimum-' --save_log_path './log/supernet_tiny-only-supernet-minimum.log' --interval 1

# python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution_only_supernet.py --data-path '/data' --gp \
# --change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --resume '/OUTPUT_PATH/checkpoint-tiny-only-supernet-minimum-21.pth' \
# --min-param-limits 5 --param-limits 6 \
# --log-file-path './log/search_tiny-only-supernet-minimum_6M.log'
