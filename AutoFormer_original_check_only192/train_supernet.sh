python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train_only_supernet.py --data-path '/dataset/ILSVRC2012' --gp \
--change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 500 --warmup-epochs 20 \
--output /OUTPUT_PATH --batch-size 128 \
--save_checkpoint_path 'checkpoint-original-only192-' --save_log_path './log/supernet_original_only192.log'
