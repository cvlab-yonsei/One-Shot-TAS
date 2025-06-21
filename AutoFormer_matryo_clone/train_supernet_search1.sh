# python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution_base.py --data-path '/dataset/ILSVRC2012' --gp \
# --change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-B.yaml --resume '/OUTPUT_PATH/checkpoint-original-base-200-500-12-123-25.pth' \
# --min-param-limits 46 --param-limits 48 \
# --log-file-path './log/search-checkpoint-original-base-25-48M.log'


python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution_base.py --data-path '/dataset/ILSVRC2012' --gp \
--change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-B.yaml --resume '/OUTPUT_PATH/checkpoint-original-base-25.pth' \
--min-param-limits 46 --param-limits 48 \
--log-file-path './log/search-checkpoint-original-base-25-48M.log'
