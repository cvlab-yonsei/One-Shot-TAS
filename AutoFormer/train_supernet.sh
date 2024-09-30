#!/bin/bash

python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train.py --data-path '/data' --gp \
--change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 500 --warmup-epochs 20 \
--log-file-path './greedyTAS/greedyTAS-epoch100-top-k(1024).log' \
--resume './greedyTAS/checkpoint-4.pth' --output /OUTPUT_PATH --batch-size 128 

python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train.py --data-path '/data' --gp \
--change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 500 --warmup-epochs 20 \
--log-file-path './greedyTAS/greedyTAS-epoch200-top-k(1024).log' \
--resume './greedyTAS/checkpoint-9.pth' --output /OUTPUT_PATH --batch-size 128 

python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train.py --data-path '/data' --gp \
--change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 500 --warmup-epochs 20 \
--log-file-path './greedyTAS/greedyTAS-epoch300-top-k(1024).log' \
--resume './greedyTAS/checkpoint-14.pth' --output /OUTPUT_PATH --batch-size 128 

python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train.py --data-path '/data' --gp \
--change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 500 --warmup-epochs 20 \
--log-file-path './greedyTAS/greedyTAS-epoch400-top-k(1024).log' \
--resume './greedyTAS/checkpoint-19.pth' --output /OUTPUT_PATH --batch-size 128 

python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train.py --data-path '/data' --gp \
--change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 500 --warmup-epochs 20 \
--log-file-path './greedyTAS/greedyTAS-epoch500-top-k(1024).log' \
--resume './greedyTAS/checkpoint-24.pth' --output /OUTPUT_PATH --batch-size 128 

python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train.py --data-path '/data' --gp \
--change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 500 --warmup-epochs 20 \
--log-file-path './greedyTAS/greedyTAS-epoch0-top-k(1024).log' \
--output /OUTPUT_PATH --batch-size 128 

python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train.py --data-path '/data' --gp \
--change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 500 --warmup-epochs 20 \
--log-file-path './greedyTAS/greedyTAS-epoch20-top-k(1024).log' \
--resume './greedyTAS/checkpoint-0.pth' --output /OUTPUT_PATH --batch-size 128 


# --resume './greedyTAS/greedyTAS-epoch100-test/checkpoint-4.pth'
# --resume './greedyTAS/checkpoint-09121607.pth' 
# --resume './experiments/supernet/autoformer_t_500ep.pth'
# --resume './greedyTAS/greedyTAS-epoch20-test/checkpoint-0.pth'
# --resume './greedyTAS/greedyTAS-epoch59/checkpoint.pth'
# --resume './greedyTAS/checkpoint-24.pth' 