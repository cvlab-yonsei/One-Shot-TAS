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


# python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train.py --data-path '/data' --gp \
# --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 100 --warmup-epochs 4 \
# --resume '/OUTPUT_PATH/checkpoint_original_check_only192_original_optimizer-epoch400-20.pth' --output /OUTPUT_PATH --batch-size 128 \
# --save_checkpoint_path 'checkpoint_original_check_only192_original_optimizer-epoch400-after-matryo-exp_super_change-all-freeze-include-layernorm-random-depth-' --save_log_path './log/supernet_original_check_only192_original_optimizer-epoch400-after-matryo-exp-super-change-all-freeze-include-layernorm-random-depth.log'

# python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train_load_matryo_216_240.py --data-path '/data' --gp \
# --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 40 --warmup-epochs 0 \
# --resume '/OUTPUT_PATH/checkpoint-original-only192216-training-23.pth' --output /OUTPUT_PATH --batch-size 128 \
# --save_checkpoint_path 'checkpoint-original-only192216-training-460-216-1-240-3-lr00005-allfreeze-' --save_log_path './log/supernet_original_only192216_training_460_216_1_240_3_lr00005_allfreeze.log'

# python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train_load_matryo_216_240.py --data-path '/data' --gp \
# --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 40 --warmup-epochs 0 \
# --resume '/OUTPUT_PATH/checkpoint-original-only192216-training-23.pth' --output /OUTPUT_PATH --batch-size 128 \
# --save_checkpoint_path 'checkpoint-original-only192216-training-460-216-1-240-1-lr00005-allfreeze-' --save_log_path './log/supernet_original_only192216_training_460_216_1_240_1_lr00005_allfreeze.log'

# python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train_load_matryo_216_240_2.py --data-path '/data' --gp \
# --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 40 --warmup-epochs 0 \
# --resume '/OUTPUT_PATH/checkpoint-original-only192216-training-23.pth' --output /OUTPUT_PATH --batch-size 128 \
# --save_checkpoint_path 'checkpoint-original-only192216-training-460-216-0-240-1-lr00005-allfreeze-' --save_log_path './log/supernet_original_only192216_training_460_216_0_240_1_lr00005_allfreeze.log'

# python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train_load_matryo_216_240.py --data-path '/data' --gp \
# --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 40 --warmup-epochs 0 \
# --resume '/OUTPUT_PATH/checkpoint-original-only192216-training-23.pth' --output /OUTPUT_PATH --batch-size 128 \
# --save_checkpoint_path 'checkpoint-original-only192216-training-460-216-1-0-240-1-1-lr00005-allfreeze-' --save_log_path './log/supernet_original_only192216_training_460_216_1-0_240_1-1_lr00005_allfreeze.log'

# python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train_load_matryo_216_240.py --data-path '/data' --gp \
# --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 40 --warmup-epochs 0 \
# --resume '/OUTPUT_PATH/checkpoint-original-only192216-training-100-460-23.pth' --output /OUTPUT_PATH --batch-size 128 \
# --save_checkpoint_path 'checkpoint-original-only192216-training-100-460-216-1-0-240-1-1-10-lr00005-allfreeze-' --save_log_path './log/supernet_original_only192216_training_100_460_216_1-0_240_1-1-10_lr00005_allfreeze.log'

# python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train_load_matryo_216_240.py --data-path '/data' --gp \
# --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 40 --warmup-epochs 0 \
# --resume '/OUTPUT_PATH/checkpoint-original-only192216-training-100-460-23.pth' --output /OUTPUT_PATH --batch-size 128 \
# --save_checkpoint_path 'checkpoint-original-only192216-training-100-460-216-1-0-240-1-1-10-not-freeze-lr00005-allfreeze-' --save_log_path './log/supernet_original_only192216_training_100_460_216_1-0_240_1-1-10-not-freeze_lr00005_allfreeze.log'

# python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train_load_matryo_216_240.py --data-path '/data' --gp \
# --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 40 --warmup-epochs 0 \
# --resume '/OUTPUT_PATH/checkpoint-original-only192216-training-100-460-23.pth' --output /OUTPUT_PATH --batch-size 128 \
# --save_checkpoint_path 'checkpoint-original-only192216-training-100-460-216-1-240-1-lr00005-allfreeze-' --save_log_path './log/supernet_original_only192216_training_100_460_216_1_240_1_lr00005_allfreeze.log'

# python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train_load_matryo_216_240.py --data-path '/data' --gp \
# --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 40 --warmup-epochs 0 \
# --resume '/OUTPUT_PATH/checkpoint-original-only192216-training-100-460-23.pth' --output /OUTPUT_PATH --batch-size 128 \
# --save_checkpoint_path 'checkpoint-original-only192216-training-100-460-216-0-240-1-lr00005-allfreeze-' --save_log_path './log/supernet_original_only192216_training_100_460_216_0_240_1_lr00005_allfreeze.log'

# python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train_load_matryo_216_240_2.py --data-path '/data' --gp \
# --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 40 --warmup-epochs 0 \
# --resume '/OUTPUT_PATH/checkpoint-original-only192216-training-100-460-23.pth' --output /OUTPUT_PATH --batch-size 128 \
# --save_checkpoint_path 'checkpoint-original-only192216-training-100-460-216-1-240-3-lr00005-allfreeze-' --save_log_path './log/supernet_original_only192216_training_100_460_216_1_240_3_lr00005_allfreeze.log'

python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train_load_matryo_216_240.py --data-path '/data' --gp \
--change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 20 --warmup-epochs 0 \
--resume '/OUTPUT_PATH/checkpoint-original-only192216-training-400-500-25.pth' --output /OUTPUT_PATH --batch-size 128 \
--save_checkpoint_path 'checkpoint-original-only192216-training-400-500-216-1-240-1-lr00005-allfreeze-' --save_log_path './log/supernet_original_only192216_training_400_500_216_1_240_1_lr00005_allfreeze.log'

