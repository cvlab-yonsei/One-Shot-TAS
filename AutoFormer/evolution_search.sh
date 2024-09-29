#!/bin/bash

# 첫 번째 작업 실행
python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution.py --data-path '/data' --gp \
--change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --resume './experiments/supernet/checkpoint-25.pth' \
--min-param-limits 1 --param-limits 100 --config-list-path './greedyTAS/m(2500)_path_epoch100.pkl' \
--log-file-path './greedyTAS/m(2500)_path_epoch100-subnet.log'


# #!/bin/bash
# python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution.py --data-path '/data' --gp \
# --change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --resume './experiments/supernet/checkpoint-25.pth' \
# --min-param-limits 1 --param-limits 100
# # --data-set EVO_IMNET


