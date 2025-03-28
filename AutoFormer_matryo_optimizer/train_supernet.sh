# python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train.py --data-path '/dataset/ILSVRC2012' --gp \
# --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 500 --warmup-epochs 20 \
# --output /OUTPUT_PATH --batch-size 128 \
# --save_checkpoint_path 'checkpoint-matryo-' --save_log_path './log/supernet_matryo.log'

# python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution.py --data-path '/dataset/ILSVRC2012' --gp \
# --change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --resume '/OUTPUT_PATH/checkpoint-matryo-25.pth' \
# --min-param-limits 5 --param-limits 6 \
# --log-file-path './log/search_tiny-matryo-not-curriculum_6M.log'

python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train.py --data-path '/dataset/ILSVRC2012' --gp \
--change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 500 --warmup-epochs 20 \
--output /OUTPUT_PATH --batch-size 128 \
--save_checkpoint_path 'checkpoint-matryo-module-' --save_log_path './log/supernet_matryo-module.log'