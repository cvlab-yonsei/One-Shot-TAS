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

# python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train_load_matryo_216.py --data-path '/dataset/ILSVRC2012' --gp \
# --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 20 --warmup-epochs 0 \
# --resume '/OUTPUT_PATH/checkpoint_original_check_only192_original_optimizer-epoch480-24.pth' --output /OUTPUT_PATH --batch-size 128 \
# --save_checkpoint_path 'checkpoint_original_check_only192_original_optimizer-epoch480-matryo_load_matryo_216_decay0001-' --save_log_path './log/supernet_original_check_only192_original_optimizer-epoch480-matryo_load_matryo_216_decay0001.log'

# python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train_load_matryo_216.py --data-path '/dataset/ILSVRC2012' --gp \
# --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 20 --warmup-epochs 0 \
# --resume '/OUTPUT_PATH/checkpoint_original_check_only192_original_optimizer-epoch480-24.pth' --output /OUTPUT_PATH --batch-size 128 \
# --save_checkpoint_path 'checkpoint_original_check_only192_original_optimizer-epoch480-matryo_load_matryo_216_all_optim-' --save_log_path './log/supernet_original_check_only192_original_optimizer-epoch480-matryo_load_matryo_216_all_optim.log'

# python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train_load_matryo_216.py --data-path '/dataset/ILSVRC2012' --gp \
# --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 20 --warmup-epochs 0 \
# --resume '/OUTPUT_PATH/checkpoint_original_check_only192_original_optimizer-epoch480-24.pth' --output /OUTPUT_PATH --batch-size 128 \
# --save_checkpoint_path 'checkpoint_original_check_only192_original_optimizer-epoch480-matryo_load_matryo_216_param-fc-gaus-' --save_log_path './log/supernet_original_check_only192_original_optimizer-epoch480-matryo_load_matryo_216_param_fc_gaus.log'

# python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train_load_matryo_216.py --data-path '/dataset/ILSVRC2012' --gp \
# --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 20 --warmup-epochs 0 \
# --resume '/OUTPUT_PATH/checkpoint_original_check_only192_original_optimizer-epoch480-24.pth' --output /OUTPUT_PATH --batch-size 128 \
# --save_checkpoint_path 'checkpoint_original_check_only192_original_optimizer-epoch480-matryo_load_matryo_216_param-fc-onlyallbias-gaus-no-share-1e-4-decay005-' --save_log_path './log/supernet_original_check_only192_original_optimizer-epoch480-matryo_load_matryo_216_param_fc_onlyallbias_gaus_no_share_1e_4_decay005.log'

# python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train_load_matryo_240.py --data-path '/dataset/ILSVRC2012' --gp \
# --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 20 --warmup-epochs 0 \
# --resume '/OUTPUT_PATH/checkpoint_original_check_only192_original_optimizer-epoch480-matryo_load_matryo_216_decay0001-2.pth' --output /OUTPUT_PATH --batch-size 128 \
# --save_checkpoint_path 'checkpoint_original_check_only192_original_optimizer-epoch480-matryo_load_matryo_216_decay0001-fcshare-2-240-' --save_log_path './log/supernet_original_check_only192_original_optimizer-epoch480-matryo_load_matryo_216_decay0001-fcshare-2-240.log'

# python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train_load_matryo_240.py --data-path '/dataset/ILSVRC2012' --gp \
# --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 20 --warmup-epochs 0 \
# --resume '/OUTPUT_PATH/checkpoint_original_check_only192_original_optimizer-epoch480-matryo_load_matryo_216_decay0001-2.pth' --output /OUTPUT_PATH --batch-size 128 \
# --save_checkpoint_path 'checkpoint_original_check_only192_original_optimizer-epoch480-matryo_load_matryo_216_decay0001-fcnoshare-2-240-' --save_log_path './log/supernet_original_check_only192_original_optimizer-epoch480-matryo_load_matryo_216_decay0001-fcnoshare-2-240.log'

# python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train_load_matryo_216.py --data-path '/dataset/ILSVRC2012' --gp \
# --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 20 --warmup-epochs 0 \
# --resume '/OUTPUT_PATH/checkpoint_original_check_only192_original_optimizer-epoch400-20.pth' --output /OUTPUT_PATH --batch-size 128 \
# --save_checkpoint_path 'checkpoint_original_check_only192_original_optimizer-epoch480-matryo_load_matryo_216_param-fc-no-share-2e-4-decay0001-' --save_log_path './log/supernet_original_check_only192_original_optimizer-epoch480-matryo_load_matryo_216_param-fc-no-share-2e-4-decay0001.log'

# python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train_load_matryo_240.py --data-path '/dataset/ILSVRC2012' --gp \
# --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 20 --warmup-epochs 0 \
# --resume '/OUTPUT_PATH/checkpoint_original_check_only192_original_optimizer-epoch480-matryo_load_matryo_216_param-fc-no-share-2e-4-decay0001-2.pth' --output /OUTPUT_PATH --batch-size 128 \
# --save_checkpoint_path 'checkpoint_original_check_only192_original_optimizer-epoch480-matryo_load_matryo_216_param-fc-no-share-2e-4-decay0001-2-240-' --save_log_path './log/supernet_original_check_only192_original_optimizer-epoch480-matryo_load_matryo_216_param-fc-no-share-2e-4-decay0001-2-240.log'

# python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train_load_matryo_216.py --data-path '/dataset/ILSVRC2012' --gp \
# --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 40 --warmup-epochs 0 \
# --resume '/OUTPUT_PATH/checkpoint_original_check_only192_original_optimizer-epoch480-24.pth' --output /OUTPUT_PATH --batch-size 128 \
# --save_checkpoint_path 'checkpoint_original_check_only192-no-share-2e-4-decay0001-BC-' --save_log_path './log/supernet_original_check_only192-no-share-2e-4-decay0001-BC.log'

# python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train_load_matryo_216.py --data-path '/dataset/ILSVRC2012' --gp \
# --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 40 --warmup-epochs 0 \
# --resume '/OUTPUT_PATH/checkpoint_original_check_only192_original_optimizer-epoch480-24.pth' --output /OUTPUT_PATH --batch-size 128 \
# --save_checkpoint_path 'checkpoint_original_check_only192-no-share-2e-4-decay0001-BC-realshare-' --save_log_path './log/supernet_original_check_only192-no-share-2e-4-decay0001-BC-realshare.log'

# python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train_load_matryo_216.py --data-path '/dataset/ILSVRC2012' --gp \
# --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 20 --warmup-epochs 0 \
# --resume '/OUTPUT_PATH/checkpoint_original_check_only192_original_optimizer-epoch480-24.pth' --output /OUTPUT_PATH --batch-size 128 \
# --save_checkpoint_path 'checkpoint-soft-freeze-216-decay-same-' --save_log_path './log/supernet_soft_freeze_216_decay_same.log'

# python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train_load_matryo_216_240.py --data-path '/dataset/ILSVRC2012' --gp \
# --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 40 --warmup-epochs 0 \
# --resume '/OUTPUT_PATH/checkpoint_original_check_only192_original_optimizer-epoch480-24.pth' --output /OUTPUT_PATH --batch-size 128 \
# --save_checkpoint_path 'checkpoint-soft-freeze-216-240-gaussian-' --save_log_path './log/supernet_soft_freeze_216_240_gaussian.log'

# python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train_load_matryo_216_240.py --data-path '/dataset/ILSVRC2012' --gp \
# --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 40 --warmup-epochs 0 \
# --resume '/OUTPUT_PATH/checkpoint_original_check_only192_original_optimizer-epoch480-24.pth' --output /OUTPUT_PATH --batch-size 128 \
# --save_checkpoint_path 'checkpoint-soft-freeze-216-240-A-bias-learnable-' --save_log_path './log/supernet_soft_freeze_216_240_gaussian_A_bias_learnable.log'

# python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train_load_matryo_216_240.py --data-path '/dataset/ILSVRC2012' --gp \
# --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 40 --warmup-epochs 0 \
# --resume '/OUTPUT_PATH/checkpoint_original_check_only192_original_optimizer-epoch480-24.pth' --output /OUTPUT_PATH --batch-size 128 \
# --save_checkpoint_path 'checkpoint-soft-freezeX-216-240-A-bias-learnable-head-share-lr001-' --save_log_path './log/supernet_soft_freezeX_216_240_A_bias_learnable_head_share_lr001.log'

# python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train_load_matryo_216_240.py --data-path '/dataset/ILSVRC2012' --gp \
# --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 40 --warmup-epochs 0 \
# --resume '/OUTPUT_PATH/checkpoint_original_check_only192_original_optimizer-epoch480-24.pth' --output /OUTPUT_PATH --batch-size 128 \
# --save_checkpoint_path 'checkpoint-soft-freezeX-216-240-A-bias-learnable-head-share-lr0001-' --save_log_path './log/supernet_soft_freezeX_216_240_A_bias_learnable_head_share_lr0001.log'

# python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train_load_matryo_216_240.py --data-path '/dataset/ILSVRC2012' --gp \
# --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 40 --warmup-epochs 3 \
# --resume '/OUTPUT_PATH/checkpoint_original_check_only192_original_optimizer-epoch480-24.pth' --output /OUTPUT_PATH --batch-size 128 \
# --save_checkpoint_path 'checkpoint-soft-freezeX-216-240-lr001-regterm-' --save_log_path './log/supernet_soft_freezeX_216_240_lr001_regterm.log'

# python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train_load_matryo_216_240.py --data-path '/dataset/ILSVRC2012' --gp \
# --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 40 --warmup-epochs 3 \
# --resume '/OUTPUT_PATH/checkpoint_original_check_only192_original_optimizer-epoch480-24.pth' --output /OUTPUT_PATH --batch-size 128 \
# --save_checkpoint_path 'checkpoint-soft-freezeX-216-240-lr0001-regterm-0005-' --save_log_path './log/supernet_soft_freezeX_216_240_lr0001_regterm_0005.log'

python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train_load_matryo_216_240.py --data-path '/dataset/ILSVRC2012' --gp \
--change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 40 --warmup-epochs 0 \
--resume '/OUTPUT_PATH/checkpoint_original_check_only192_original_optimizer-epoch480-24.pth' --output /OUTPUT_PATH --batch-size 128 \
--save_checkpoint_path 'checkpoint-lr000001-feature-align-linear-' --save_log_path './log/supernet_lr000001_feature_align_linear.log'


# python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution_original.py --data-path '/dataset/ILSVRC2012' --gp \
# --change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --resume '/OUTPUT_PATH/checkpoint_original_check_only192_original_optimizer-epoch480-24.pth' \
# --min-param-limits 5 --param-limits 6 \
# --log-file-path './log/search-checkpoint_original_check_only192_original_optimizer-epoch480-24-best-6M.log'




# python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train_load_matryo_240.py --data-path '/dataset/ILSVRC2012' --gp \
# --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 20 --warmup-epochs 0 \
# --resume '/OUTPUT_PATH/checkpoint_original_check_only192_original_optimizer-epoch480-matryo_load_matryo_216-2.pth' --output /OUTPUT_PATH --batch-size 128 \
# --save_checkpoint_path 'checkpoint_original_check_only192_original_optimizer-epoch480-matryo_load_matryo_240-' --save_log_path './log/supernet_original_check_only192_original_optimizer-epoch480-matryo_load_matryo_240.log'
