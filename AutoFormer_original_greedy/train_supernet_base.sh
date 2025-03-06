# python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train_base.py --data-path '/dataset/ILSVRC2012' --gp \
# --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-B.yaml --epochs 500 --warmup-epochs 20 \
# --output /OUTPUT_PATH --batch-size 128 \
# --save_checkpoint_path 'checkpoint-original-base-' --save_log_path './log/supernet_original_base.log' --interval 1

# python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train_base_sn.py --data-path '/dataset/ILSVRC2012' --gp \
# --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-B.yaml --epochs 500 --warmup-epochs 20 \
# --resume '/OUTPUT_PATH/checkpoint-original-base-22.pth' --output /OUTPUT_PATH --batch-size 128 \
# --save_checkpoint_path 'checkpoint-greeze-base-460-' --save_log_path './log/supernet_greeze_460_base.log' --interval 1

# python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution_init_mix.py --data-path '/data' --gp \
# --change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --resume '/OUTPUT_PATH/checkpoint-original-base-25.pth' \
# --min-param-limits 42 --param-limits 54 \
# --log-file-path './log/search_original_base_6M.log'

python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train_base_sn.py --data-path '/dataset/ILSVRC2012' --gp \
--change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-B.yaml --epochs 500 --warmup-epochs 20 \
--resume '/OUTPUT_PATH/checkpoint-original-base-22.pth' --output /OUTPUT_PATH --batch-size 128 \
--save_checkpoint_path 'checkpoint-greeze-base-460-droppath01-' --save_log_path './log/supernet_greeze_460_droppath01_base.log' --interval 1

