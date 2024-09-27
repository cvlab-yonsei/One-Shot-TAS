#!/bin/bash
python3 -m torch.distributed.launch --nproc_per_node=8 --use_env observe_supernet.py --data-path '/data' --gp \
--change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --resume './experiments/supernet/supernet-tiny.pth' \
--min-param-limits 1 --param-limits 7
# --data-set EVO_IMNET


