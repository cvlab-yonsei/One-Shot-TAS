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

# python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution.py --data-path '/data' --gp \
# --change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --resume '/OUTPUT_PATH/checkpoint_original_check_only192_original_optimizer-epoch400-19.pth' \
# --min-param-limits 5 --param-limits 6 \
# --log-file-path './log/search-checkpoint_original_check_only192_original_optimizer-epoch400-19-6M.log'


# python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train.py --data-path '/dataset/ILSVRC2012' --gp \
# --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 100 --warmup-epochs 4 \
# --resume '/OUTPUT_PATH/checkpoint_original_check_only192_original_optimizer-epoch400-20.pth' --output /OUTPUT_PATH --batch-size 128 \
# --save_checkpoint_path 'checkpoint_original_check_only192_original_optimizer-epoch400-after-matryo-exp_super_change-all-freeze-random-depth-' --save_log_path './log/supernet_original_check_only192_original_optimizer-epoch400-after-matryo-exp-super-change-all-freeze-random-depth.log'


# python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train.py --data-path '/dataset/ILSVRC2012' --gp \
# --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 20 --warmup-epochs 0 \
# --resume '/OUTPUT_PATH/checkpoint_original_check_only192_original_optimizer-epoch480-24.pth' --output /OUTPUT_PATH --batch-size 128 \
# --save_checkpoint_path 'checkpoint_original_check_only192_original_optimizer-epoch480-matryo-lr-1e-4-cheet-layernorm-3-droppath0-' --save_log_path './log/supernet_original_check_only192_original_optimizer-epoch480-matryo-lr-1e-4-cheet-layernorm-3-droppath0.log'

# python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train_load_matryo.py --data-path '/dataset/ILSVRC2012' --gp \
# --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 20 --warmup-epochs 0 \
# --resume '/OUTPUT_PATH/checkpoint_original_check_only192_original_optimizer-epoch480-matryo-lr-1e-4-cheet-layernorm-3-droppath0-0.pth' --output /OUTPUT_PATH --batch-size 128 \
# --save_checkpoint_path 'checkpoint_original_check_only192_original_optimizer-epoch480-matryo-lr-1e-4-cheet-layernorm-3-droppath0-0-240-single-optimizer-basecode-' --save_log_path './log/supernet_original_check_only192_original_optimizer-epoch480-matryo-lr-1e-4-cheet-layernorm-3-droppath0-0-240-single-optimizer-basecode.log'

# python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train.py --data-path '/dataset/ILSVRC2012' --gp \
# --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 20 --warmup-epochs 0 \
# --resume '/OUTPUT_PATH/checkpoint_original_check_only192_original_optimizer-epoch400-480-finetune-1e-5-24.pth' --output /OUTPUT_PATH --batch-size 128 \
# --save_checkpoint_path 'checkpoint_original_check_only192_original_optimizer-epoch400-480-finetune-1e-5-cheet-layernorm-' --save_log_path './log/supernet_original_check_only192_original_optimizer-epoch400-480-finetune-1e-5-cheet-layernorm.log'


# python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train_load_matryo_216.py --data-path '/dataset/ILSVRC2012' --gp \
# --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 20 --warmup-epochs 0 \
# --resume '/OUTPUT_PATH/checkpoint_original_check_only192_original_optimizer-epoch480-24.pth' --output /OUTPUT_PATH --batch-size 128 \
# --save_checkpoint_path 'checkpoint_original_check_only192_original_optimizer-epoch480-matryo_load_matryo_216-' --save_log_path './log/supernet_original_check_only192_original_optimizer-epoch480-matryo_load_matryo_216.log'

# python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train_load_matryo_216.py --data-path '/dataset/ILSVRC2012' --gp \
# --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 20 --warmup-epochs 0 \
# --resume '/OUTPUT_PATH/checkpoint_original_check_only192_original_optimizer-epoch480-24.pth' --output /OUTPUT_PATH --batch-size 128 \
# --save_checkpoint_path 'checkpoint_original_check_only192_original_optimizer-epoch480-matryo_load_matryo_216_reinit-' --save_log_path './log/supernet_original_check_only192_original_optimizer-epoch480-matryo_load_matryo_216_reinit.log'

python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train_load_matryo_216.py --data-path '/dataset/ILSVRC2012' --gp \
--change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 20 --warmup-epochs 0 \
--resume '/OUTPUT_PATH/checkpoint_original_check_only192_original_optimizer-epoch480-24.pth' --output /OUTPUT_PATH --batch-size 128 \
--save_checkpoint_path 'checkpoint_original_check_only192_original_optimizer-epoch480-matryo_load_matryo_216_decay0001-' --save_log_path './log/supernet_original_check_only192_original_optimizer-epoch480-matryo_load_matryo_216_decay0001.log'


# python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train_load_matryo_240.py --data-path '/dataset/ILSVRC2012' --gp \
# --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 20 --warmup-epochs 0 \
# --resume '/OUTPUT_PATH/checkpoint_original_check_only192_original_optimizer-epoch480-matryo_load_matryo_216-2.pth' --output /OUTPUT_PATH --batch-size 128 \
# --save_checkpoint_path 'checkpoint_original_check_only192_original_optimizer-epoch480-matryo_load_matryo_240-' --save_log_path './log/supernet_original_check_only192_original_optimizer-epoch480-matryo_load_matryo_240.log'
