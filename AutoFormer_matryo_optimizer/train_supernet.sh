# python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train.py --data-path '/data' --gp \
# --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 500 --warmup-epochs 20 \
# --output /OUTPUT_PATH --batch-size 128 \
# --save_checkpoint_path 'checkpoint-matryo-curriculum-small-to-big-' --save_log_path './log/supernet_matryo-curriculum-small-to-big.log'

# python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution.py --data-path '/data' --gp \
# --change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --resume '/OUTPUT_PATH/checkpoint-matryo-curriculum-small-to-big-25.pth' \
# --min-param-limits 5 --param-limits 6 \
# --log-file-path './log/search_tiny-curriculum-small-to-big_6M.log'

# python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train.py --data-path '/data' --gp \
# --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 500 --warmup-epochs 20 \
# --output /OUTPUT_PATH --batch-size 128 \
# --save_checkpoint_path 'checkpoint-matryo-optimizer-config-full-change-' --save_log_path './log/supernet_matryo-optimizer-full-change-p-to-f-module-config.log'

# python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train.py --data-path '/data' --gp \
# --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 500 --warmup-epochs 20 \
# --resume '/OUTPUT_PATH/checkpoint-matryo-optimizer-config-full-change-19.pth' --output /OUTPUT_PATH --batch-size 128 \
# --save_checkpoint_path 'checkpoint-matryo-optimizer-config-full-change-400-relinear-' --save_log_path './log/supernet_matryo-optimizer-full-change-p-to-f-module-config-400-relinear.log'

# python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution.py --data-path '/data' --gp \
# --change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --resume '/OUTPUT_PATH/checkpoint-matryo-optimizer-config-full-change-400-relinear-25.pth' \
# --min-param-limits 6 --param-limits 7 \
# --log-file-path './log/search-matryo-optimizer-config-full-change-400-relinear-7M.log'

# python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution.py --data-path '/data' --gp \
# --change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --resume '/OUTPUT_PATH/checkpoint-matryo-optimizer-config-full-change-only-minimum-19.pth' \
# --min-param-limits 5 --param-limits 6 \
# --log-file-path './log/search-checkpoint-matryo-optimizer-config-full-change-only-minimum-19-6M.log'

# python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution.py --data-path '/data' --gp \
# --change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --resume '/OUTPUT_PATH/checkpoint-matryo-optimizer-config-full-change-only-minimum-21-133.pth' \
# --min-param-limits 5 --param-limits 6 \
# --log-file-path './log/search-checkpoint-matryo-optimizer-config-full-change-only-minimum-21-133-6M.log'


python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution.py --data-path '/data' --gp \
--change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --resume '/OUTPUT_PATH/checkpoint-sn-not-original-0-prenas-aug19.pth' \
--min-param-limits 5 --param-limits 6 \
--log-file-path './log/search-checkpoint-sn-not-original-0-prenas-aug19-6M.log'

python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution.py --data-path '/data' --gp \
--change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --resume '/OUTPUT_PATH/checkpoint-sn-not-original-0-prenas-aug21.pth' \
--min-param-limits 5 --param-limits 6 \
--log-file-path './log/checkpoint-sn-not-original-0-prenas-aug21-6M.log'

python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution.py --data-path '/data' --gp \
--change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --resume '/OUTPUT_PATH/checkpoint-sn-not-original-0-prenas-aug19.pth' \
--min-param-limits 7 --param-limits 8 \
--log-file-path './log/search-checkpoint-sn-not-original-0-prenas-aug19-8M.log'

python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution.py --data-path '/data' --gp \
--change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --resume '/OUTPUT_PATH/checkpoint-sn-not-original-0-prenas-aug21.pth' \
--min-param-limits 7 --param-limits 8 \
--log-file-path './log/checkpoint-sn-not-original-0-prenas-aug21-8M.log'


# python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train.py --data-path '/data' --gp \
# --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 500 --warmup-epochs 20 \
# --resume '/OUTPUT_PATH/checkpoint-original-14.pth' --output /OUTPUT_PATH --batch-size 128 \
# --save_checkpoint_path 'checkpoint-matryo-optimizer-300-' --save_log_path './log/supernet_matryo-optimizer-300.log'
