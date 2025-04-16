# python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train_only_supernet.py --data-path '/dataset/ILSVRC2012' --gp \
# --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 500 --warmup-epochs 20 \
# --output /OUTPUT_PATH --batch-size 128 \
# --save_checkpoint_path 'checkpoint-original-only192-' --save_log_path './log/supernet_original_only192.log'

# python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train.py --data-path '/data' --gp \
# --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 500 --warmup-epochs 20 \
# --output /OUTPUT_PATH --batch-size 128 \
# --save_checkpoint_path 'checkpoint_matryo_check_only192_original_optimizer-' --save_log_path './log/supernet_matryo_check_only192_original_optimizer.log'

# python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train.py --data-path '/data' --gp \
# --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 400 --warmup-epochs 16 \
# --output /OUTPUT_PATH --batch-size 128 \
# --save_checkpoint_path 'checkpoint_original_check_only192_original_optimizer-epoch400-' --save_log_path './log/supernet_original_check_only192_original_optimizer-epoch400.log'

python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution.py --data-path '/dataset/ILSVRC2012' --gp \
--change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --resume '/OUTPUT_PATH/checkpoint_original_check_only192_original_optimizer-epoch400-after-matryo-4.pth' \
--min-param-limits 5 --param-limits 6 \
--log-file-path './log/search-original_check_only192_original_optimizer-epoch400-after-matryo-4-6M.log'
