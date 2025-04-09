# python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train.py --data-path '/dataset/ILSVRC2012' --gp \
# --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 500 --warmup-epochs 20 \
# --resume '/OUTPUT_PATH/checkpoint-matryo-module-only-mlp-14.pth' --output /OUTPUT_PATH --batch-size 128 \
# --save_checkpoint_path 'checkpoint-matryo-module-only-mlp3-' --save_log_path './log/supernet_matryo-module-only-mlp3.log'

python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train.py --data-path '/dataset/ILSVRC2012' --gp \
--change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 500 --warmup-epochs 20 \
--output /OUTPUT_PATH --batch-size 128 \
--save_checkpoint_path 'checkpoint-matryo-module-full-depth-random-grouped-' --save_log_path './log/supernet_matryo-module-full-depth-random-grouped.log'

# python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train.py --data-path '/dataset/ILSVRC2012' --gp \
# --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 500 --warmup-epochs 20 \
# --resume '/OUTPUT_PATH/checkpoint-matryo-module-only-mlp4-depth-random-19.pth' --output /OUTPUT_PATH --batch-size 128 \
# --save_checkpoint_path 'checkpoint-matryo-module-only-mlp5-depth-random-400-' --save_log_path './log/supernet_matryo-module-only-mlp5-depth-random-400.log'

# python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution.py --data-path '/data' --gp \
# --change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --resume '/OUTPUT_PATH/checkpoint-sn-440-interval-1-top(data_aug)-25.pth' \
# --min-param-limits 5 --param-limits 6 \
# --log-file-path './log/search_sn_not_original_440-interval-1-top(data_aug)_6M.log'

# python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution.py --data-path '/dataset/ILSVRC2012' --gp \
# --change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --resume '/OUTPUT_PATH/checkpoint-matryo-module-only-mlp5-depth-random-400-22.pth' \
# --min-param-limits 5 --param-limits 6 \
# --log-file-path './log/search-matryo-optimizer-config-full-change-460-relinear-6M.log'