# python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train.py --data-path '/data' --gp \
# --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 500 --warmup-epochs 20 \
# --output /OUTPUT_PATH --batch-size 128 \
# --save_checkpoint_path 'checkpoint-matryo-curriculum-small-to-big-' --save_log_path './log/supernet_matryo-curriculum-small-to-big.log'

python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution.py --data-path '/data' --gp \
--change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --resume '/OUTPUT_PATH/checkpoint-matryo-curriculum-small-to-big-25.pth' \
--min-param-limits 5 --param-limits 6 \
--log-file-path './log/search_tiny-curriculum-small-to-big_6M.log'

