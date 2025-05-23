
# python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train_feature_map.py --data-path '/data' --gp \
# --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 501 --warmup-epochs 0 \
# --resume '/OUTPUT_PATH/checkpoint-tiny-only-supernet-minimum-21.pth' --output /OUTPUT_PATH --batch-size 128 \
# --save_log_path './log/check_feature_map.log'

python3 -m torch.distributed.launch --nproc_per_node=8 --use_env observe_supernet.py --data-path '/data' --gp \
--change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --resume '/OUTPUT_PATH/checkpoint-tiny-only-supernet-minimum-21.pth' \
--min-param-limits 1 --param-limits 12