# python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution_base.py --data-path '/dataset/ILSVRC2012' --gp \
# --change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-B.yaml --resume '/OUTPUT_PATH/checkpoint-original-base-200-500-12-123-25.pth' \
# --min-param-limits 46 --param-limits 48 \
# --log-file-path './log/search-checkpoint-original-base-200-500-12-123-25-48M.log'

# python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution_base.py --data-path '/dataset/ILSVRC2012' --gp \
# --change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-B.yaml --resume '/OUTPUT_PATH/checkpoint-original-base-200-500-12-123-25.pth' \
# --min-param-limits 62 --param-limits 64 \
# --log-file-path './log/search-checkpoint-original-base-200-500-12-123-25-64M.log'

# python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution_base.py --data-path '/dataset/ILSVRC2012' --gp \
# --change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-B.yaml --resume '/OUTPUT_PATH/checkpoint-original-base-25.pth' \
# --min-param-limits 46 --param-limits 48 \
# --log-file-path './log/search-checkpoint-original-base-25-48M.log'

# python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution_base.py --data-path '/dataset/ILSVRC2012' --gp \
# --change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-B.yaml --resume '/OUTPUT_PATH/checkpoint-original-base-25.pth' \
# --min-param-limits 62 --param-limits 64 \
# --log-file-path './log/search-checkpoint-original-base-25-64M.log'

python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution_base.py --data-path '/dataset/ILSVRC2012' --gp \
--change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-B.yaml --resume '/OUTPUT_PATH/autoformer_B_prenas_aug.pth' \
--min-param-limits 46 --param-limits 48 \
--log-file-path './log/search-autoformer_B_prenas_aug-48M.log'

python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution_base.py --data-path '/dataset/ILSVRC2012' --gp \
--change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-B.yaml --resume '/OUTPUT_PATH/autoformer_B_prenas_aug.pth' \
--min-param-limits 62 --param-limits 64 \
--log-file-path './log/search-autoformer_B_prenas_aug-64M.log'

python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution_base.py --data-path '/dataset/ILSVRC2012' --gp \
--change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-B.yaml --resume '/OUTPUT_PATH/checkpoint-original-base-200-500-12-123-1-1-lr000005-allfreeze-tuple-0-5ep.pth' \
--min-param-limits 62 --param-limits 64 \
--log-file-path './log/search-checkpoint-original-base-200-500-12-123-1-1-lr000005-allfreeze-tuple-0-5ep-64M.log'