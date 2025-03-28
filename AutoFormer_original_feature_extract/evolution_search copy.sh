#!/bin/bash

# 첫 번째 작업 실행
python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution.py --data-path '/data' --gp \
--change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --resume '/OUTPUT_PATH/checkpoint-original-25.pth' \
--min-param-limits 5 --param-limits 6 \
--log-file-path './log/search_original_tiny_6M.log'

python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution.py --data-path '/data' --gp \
--change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --resume '/OUTPUT_PATH/checkpoint-sn-25.pth' \
--min-param-limits 5 --param-limits 6 \
--log-file-path './log/search_sn_tiny_6M.log'

# 첫 번째 작업 실행
python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution.py --data-path '/data' --gp \
--change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --resume '/OUTPUT_PATH/checkpoint-original-25.pth' \
--min-param-limits 6 --param-limits 7 \
--log-file-path './log/search_original_tiny_7M.log'

python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution.py --data-path '/data' --gp \
--change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --resume '/OUTPUT_PATH/checkpoint-sn-25.pth' \
--min-param-limits 6 --param-limits 7 \
--log-file-path './log/search_sn_tiny_7M.log'

# 첫 번째 작업 실행
python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution.py --data-path '/data' --gp \
--change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --resume '/OUTPUT_PATH/checkpoint-original-25.pth' \
--min-param-limits 7 --param-limits 8 \
--log-file-path './log/search_original_tiny_8M.log'

python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution.py --data-path '/data' --gp \
--change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --resume '/OUTPUT_PATH/checkpoint-sn-25.pth' \
--min-param-limits 7 --param-limits 8 \
--log-file-path './log/search_sn_tiny_8M.log'

# 첫 번째 작업 실행
python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution.py --data-path '/data' --gp \
--change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --resume '/OUTPUT_PATH/checkpoint-original-25.pth' \
--min-param-limits 8 --param-limits 9 \
--log-file-path './log/search_original_tiny_9M.log'

python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution.py --data-path '/data' --gp \
--change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --resume '/OUTPUT_PATH/checkpoint-sn-25.pth' \
--min-param-limits 8 --param-limits 9 \
--log-file-path './log/search_sn_tiny_9M.log'

# 첫 번째 작업 실행
python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution.py --data-path '/data' --gp \
--change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --resume '/OUTPUT_PATH/checkpoint-original-25.pth' \
--min-param-limits 9 --param-limits 10 \
--log-file-path './log/search_original_tiny_10M.log'

python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution.py --data-path '/data' --gp \
--change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --resume '/OUTPUT_PATH/checkpoint-sn-25.pth' \
--min-param-limits 9 --param-limits 10 \
--log-file-path './log/search_sn_tiny_10M.log'



# # 첫 번째 작업이 성공적으로 완료되면 두 번째 작업 실행
# if [ $? -eq 0 ]; then
#     python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution.py --data-path '/data' --gp \
#     --change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --resume './experiments/supernet/checkpoint-25.pth' \
#     --min-param-limits 1 --param-limits 100 --config-list-path './greedyTAS/greedyTAS-epoch20-test/autoformer-greedyTAS(dss)-20epoch.pkl' \
#     --log-file-path './greedyTAS/greedyTAS-epoch20-test/autoformer-greedyTAS(dss)-20epoch-subnet.log'
# else
#     echo "첫 번째 작업이 실패했습니다. 두 번째 작업을 실행하지 않습니다."
# fi
# # --data-set EVO_IMNET


# #!/bin/bash
# python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution.py --data-path '/data' --gp \
# --change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --resume './experiments/supernet/checkpoint-25.pth' \
# --min-param-limits 1 --param-limits 100
# # --data-set EVO_IMNET


