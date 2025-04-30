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

# python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution.py --data-path '/dataset/ILSVRC2012' --gp \
# --change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --resume '/OUTPUT_PATH/checkpoint_original_check_only192_original_optimizer-epoch400-after-matryo-4.pth' \
# --min-param-limits 5 --param-limits 6 \
# --log-file-path './log/search-original_check_only192_original_optimizer-epoch400-after-matryo-4-6M.log'


# python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution.py --data-path '/dataset/ILSVRC2012' --gp \
# --change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --resume '/OUTPUT_PATH/checkpoint-original-25.pth' \
# --min-param-limits 7 --param-limits 8 \
# --log-file-path './log/search-original_tiny-epoch400-25-8M.log'

# python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution.py --data-path '/dataset/ILSVRC2012' --gp \
# --change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --resume '/OUTPUT_PATH/checkpoint_original_check_only192_original_optimizer-epoch480-matryo-0.pth' \
# --min-param-limits 7 --param-limits 8 \
# --log-file-path './log/search-checkpoint_original_check_only192_original_optimizer-epoch480-matryo-0-8M.log'

# python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution.py --data-path '/dataset/ILSVRC2012' --gp \
# --change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --resume '/OUTPUT_PATH/checkpoint_original_check_only192_original_optimizer-epoch480-matryo-lr-1e-4-cheet-1.pth' \
# --min-param-limits 5 --param-limits 6 \
# --log-file-path './log/search-checkpoint_original_check_only192_original_optimizer-epoch480-matryo-lr-1e-4-cheet-1-6M.log'

# python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution.py --data-path '/dataset/ILSVRC2012' --gp \
# --change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --resume '/OUTPUT_PATH/checkpoint_original_check_only192_original_optimizer-epoch400-480-finetune-1e-5-22.pth' \
# --min-param-limits 5 --param-limits 6 \
# --log-file-path './log/search-checkpoint_original_check_only192_original_optimizer-epoch400-480-finetune-1e-5-22-6M.log'


# python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution.py --data-path '/dataset/ILSVRC2012' --gp \
# --change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --resume '/OUTPUT_PATH/checkpoint_original_check_only192_original_optimizer-epoch480-24.pth' \
# --min-param-limits 5 --param-limits 6 \
# --log-file-path './log/search-checkpoint_original_check_only192_original_optimizer-epoch480-24-6M.log'

# # 480 후 216 10에폭 한 뒤. 240은 x
# python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution.py --data-path '/dataset/ILSVRC2012' --gp \
# --change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --resume '/OUTPUT_PATH/checkpoint_original_check_only192_original_optimizer-epoch480-matryo-lr-1e-4-cheet-layernorm-3-droppath0-0.pth' \
# --min-param-limits 9 --param-limits 10 \
# --log-file-path './log/search-checkpoint_original_check_only192_original_optimizer-epoch480-matryo-lr-1e-4-cheet-layernorm-3-droppath0-0-10M.log'

# # 480 후 216 10에폭 한 뒤 240 10 에폭.
# python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution.py --data-path '/dataset/ILSVRC2012' --gp \
# --change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --resume '/OUTPUT_PATH/checkpoint_original_check_only192_original_optimizer-epoch480-matryo-lr-1e-4-cheet-layernorm-3-droppath0-0-240-single-optimizer-basecode-0.pth' \
# --min-param-limits 9 --param-limits 10 \
# --log-file-path './log/search-checkpoint_original_check_only192_original_optimizer-epoch480-matryo-lr-1e-4-cheet-layernorm-3-droppath0-0-240-single-optimizer-basecode-0-10M.log'

# # 480 후 216 10에폭 한 뒤 240 10 에폭.
# python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution.py --data-path '/dataset/ILSVRC2012' --gp \
# --change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --resume '/OUTPUT_PATH/checkpoint_original_check_only192_original_optimizer-epoch480-matryo-lr-1e-4-cheet-layernorm-3-droppath0-0-240-single-optimizer-basecode-1.pth' \
# --min-param-limits 9 --param-limits 10 \
# --log-file-path './log/search-checkpoint_original_check_only192_original_optimizer-epoch480-matryo-lr-1e-4-cheet-layernorm-3-droppath0-0-240-single-optimizer-basecode-1-10M.log'


# # tiny original
# python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution_original.py --data-path '/dataset/ILSVRC2012' --gp \
# --change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --resume '/OUTPUT_PATH/checkpoint-original-25.pth' \
# --min-param-limits 9 --param-limits 10 \
# --log-file-path './log/search-checkpoint-original-25-10M.log'

# # tiny original
# python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution_original.py --data-path '/dataset/ILSVRC2012' --gp \
# --change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --resume '/OUTPUT_PATH/checkpoint-original-23.pth' \
# --min-param-limits 7 --param-limits 8 \
# --log-file-path './log/search-checkpoint-original-23(480)-8M.log'

# python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution_original.py --data-path '/dataset/ILSVRC2012' --gp \
# --change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --resume '/OUTPUT_PATH/checkpoint-original-22.pth' \
# --min-param-limits 7 --param-limits 8 \
# --log-file-path './log/search-checkpoint-original-22(460)-8M.log'

# python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution_original.py --data-path '/dataset/ILSVRC2012' --gp \
# --change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --resume '/OUTPUT_PATH/checkpoint-original-22.pth' \
# --min-param-limits 9 --param-limits 10 \
# --log-file-path './log/search-checkpoint-original-22(460)-10M.log'

# python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution.py --data-path '/dataset/ILSVRC2012' --gp \
# --change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --resume '/OUTPUT_PATH/checkpoint_original_check_only192_original_optimizer-epoch480-matryo_load_matryo_216_decay0001-2.pth' \
# --min-param-limits 7 --param-limits 8 \
# --log-file-path './log/search-checkpoint_original_check_only192_original_optimizer-epoch480-matryo_load_matryo_216_decay0001-2-8M.log'


# python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution_original.py --data-path '/dataset/ILSVRC2012' --gp \
# --change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --resume '/OUTPUT_PATH/checkpoint-original-25.pth' \
# --min-param-limits 5 --param-limits 6 \
# --log-file-path './log/search-checkpoint-original-25(500)-6M.log'

# python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution.py --data-path '/dataset/ILSVRC2012' --gp \
# --change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --resume '/OUTPUT_PATH/checkpoint_original_check_only192_original_optimizer-epoch480-matryo_load_matryo_216_decay0001-fcnoshare-2-240-2.pth' \
# --min-param-limits 9 --param-limits 10 \
# --log-file-path './log/search-checkpoint_original_check_only192_original_optimizer-epoch480-matryo_load_matryo_216_decay0001-fcnoshare-2-240-2-10M.log'

# python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution_original.py --data-path '/dataset/ILSVRC2012' --gp \
# --change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --resume '/OUTPUT_PATH/checkpoint_original_check_only192_original_optimizer-epoch400-20.pth' \
# --min-param-limits 5 --param-limits 6 \
# --log-file-path './log/checkpoint_original_check_only192_original_optimizer-epoch400-20-6M.log'

# python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution.py --data-path '/dataset/ILSVRC2012' --gp \
# --change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --resume '/OUTPUT_PATH/checkpoint_original_check_only192_original_optimizer-epoch480-matryo_load_matryo_216_param-fc-no-share-2e-4-decay0001-2.pth' \
# --min-param-limits 7 --param-limits 8 \
# --log-file-path './log/search-checkpoint_original_check_only192_original_optimizer-epoch480-matryo_load_matryo_216_param-fc-no-share-2e-4-decay0001-2-8M.log'

# python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution.py --data-path '/dataset/ILSVRC2012' --gp \
# --change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --resume '/OUTPUT_PATH/checkpoint_original_check_only192_original_optimizer-epoch480-matryo_load_matryo_216_param-fc-no-share-2e-4-decay0001-2-240-2.pth' \
# --min-param-limits 9 --param-limits 10 \
# --log-file-path './log/search-checkpoint_original_check_only192_original_optimizer-epoch400-matryo_load_matryo_216_param-fc-no-share-2e-4-decay0001-2-240-2-10M.log'

# python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution_original.py --data-path '/dataset/ILSVRC2012' --gp \
# --change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --resume '/OUTPUT_PATH/checkpoint-original-21.pth' \
# --min-param-limits 9 --param-limits 10 \
# --log-file-path './log/search-checkpoint-original-21(440)-10M.log'

# python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution_original.py --data-path '/dataset/ILSVRC2012' --gp \
# --change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --resume '/OUTPUT_PATH/checkpoint-original-22.pth' \
# --min-param-limits 5 --param-limits 6 \
# --log-file-path './log/search-checkpoint-original-22(460)-6M.log'

# python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution_original.py --data-path '/dataset/ILSVRC2012' --gp \
# --change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --resume '/OUTPUT_PATH/checkpoint-original-19.pth' \
# --min-param-limits 9 --param-limits 10 \
# --log-file-path './log/search-checkpoint-original-19(400)-10M.log'


# python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution.py --data-path '/dataset/ILSVRC2012' --gp \
# --change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --resume '/OUTPUT_PATH/checkpoint_original_check_only192-no-share-2e-4-decay0001-BC-4.pth' \
# --min-param-limits 7 --param-limits 8 \
# --log-file-path './log/search-checkpoint_original_check_only192-no-share-2e-4-decay0001-BC-4-8M.log'

# python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution.py --data-path '/dataset/ILSVRC2012' --gp \
# --change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --resume '/OUTPUT_PATH/checkpoint_original_check_only192-no-share-2e-4-decay0001-BC-4.pth' \
# --min-param-limits 9 --param-limits 10 \
# --log-file-path './log/search-checkpoint_original_check_only192-no-share-2e-4-decay0001-BC-4-10M.log'

# python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train_load_matryo_216_240.py --data-path '/dataset/ILSVRC2012' --gp \
# --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 40 --warmup-epochs 0 \
# --resume '/OUTPUT_PATH/checkpoint_original_check_only192_original_optimizer-epoch480-24.pth' --output /OUTPUT_PATH --batch-size 128 \
# --save_checkpoint_path 'checkpoint-soft-freeze-216-240-gaussian-' --save_log_path './log/supernet_soft_freeze_216_240_gaussian.log'

python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution.py --data-path '/dataset/ILSVRC2012' --gp \
--change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --resume '/OUTPUT_PATH/checkpoint_original_check_only192-no-share-2e-4-decay0001-BC-4.pth' \
--min-param-limits 7 --param-limits 8 \
--log-file-path './log/search-checkpoint_original_check_only192-no-share-2e-4-decay0001-BC-4-8M-4.log'
