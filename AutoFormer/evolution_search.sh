#!/bin/bash

# 첫 번째 작업 실행
python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution.py --data-path '/data' --gp \
--change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --resume './experiments/supernet/checkpoint-25.pth' \
--min-param-limits 1 --param-limits 100 --config-list-path './greedyTAS/greedyTAS-epoch20-test/autoformer-greedyTAS(bottom)-20epoch.pkl' \
--log-file-path './greedyTAS/greedyTAS-epoch20-test/autoformer-greedyTAS(bottom)-20epoch-subnet.log'

# 첫 번째 작업이 성공적으로 완료되면 두 번째 작업 실행
if [ $? -eq 0 ]; then
    python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution.py --data-path '/data' --gp \
    --change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --resume './experiments/supernet/checkpoint-25.pth' \
    --min-param-limits 1 --param-limits 100 --config-list-path './greedyTAS/greedyTAS-epoch20-test/autoformer-greedyTAS(dss)-20epoch.pkl' \
    --log-file-path './greedyTAS/greedyTAS-epoch20-test/autoformer-greedyTAS(dss)-20epoch-subnet.log'
else
    echo "첫 번째 작업이 실패했습니다. 두 번째 작업을 실행하지 않습니다."
fi
# --data-set EVO_IMNET


# #!/bin/bash
# python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution.py --data-path '/data' --gp \
# --change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --resume './experiments/supernet/checkpoint-25.pth' \
# --min-param-limits 1 --param-limits 100
# # --data-set EVO_IMNET


