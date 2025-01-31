python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train_base.py --data-path '/data' --gp \
--change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-B.yaml --epochs 500 --warmup-epochs 20 \
--output /OUTPUT_PATH --batch-size 128 \
--save_checkpoint_path 'checkpoint-original-base-' --save_log_path './log/supernet_original_base.log' --interval 1
