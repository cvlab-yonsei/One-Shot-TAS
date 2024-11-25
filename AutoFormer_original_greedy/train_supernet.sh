# python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train_original.py --data-path '/data' --gp \
# --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 500 --warmup-epochs 20 \
# --resume '/OUTPUT_PATH/checkpoint-original-19.pth' --output /OUTPUT_PATH --batch-size 128 

##################

# python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train_sn_not_original.py --data-path '/data' --gp \
# --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 500 --warmup-epochs 20 \
# --resume '/OUTPUT_PATH/checkpoint-original-19.pth' --output /OUTPUT_PATH --batch-size 128 

python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train_sn1.py --data-path '/data' --gp \
--change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 500 --warmup-epochs 20 \
--resume '/OUTPUT_PATH/checkpoint-original-19.pth' --output /OUTPUT_PATH --batch-size 128 \
--save_checkpoint_path 'checkpoint-sn-400-interval-5-top-' --save_log_path './log/supernet_greedy_spectral_norm_400ep_interval_5_topk.log' --interval 5

python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train_sn1.py --data-path '/data' --gp \
--change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 500 --warmup-epochs 20 \
--resume '/OUTPUT_PATH/checkpoint-original-19.pth' --output /OUTPUT_PATH --batch-size 128 \
--save_checkpoint_path 'checkpoint-sn-400-interval-1-top-' --save_log_path './log/supernet_greedy_spectral_norm_400ep_interval_1_topk.log' --interval 1

python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train_sn1.py --data-path '/data' --gp \
--change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 500 --warmup-epochs 20 \
--resume '/OUTPUT_PATH/checkpoint-original-14.pth' --output /OUTPUT_PATH --batch-size 128 \
--save_checkpoint_path 'checkpoint-sn-300-interval-1-top-' --save_log_path './log/supernet_greedy_spectral_norm_300ep_interval_1_topk.log' --interval 1

# top-k
# python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train_sn1.py --data-path '/data' --gp \
# --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 500 --warmup-epochs 20 \
# --resume '/OUTPUT_PATH/checkpoint-original-19.pth' --output /OUTPUT_PATH --batch-size 128 


# bottom-k
# python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train_sn2.py --data-path '/data' --gp \
# --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 500 --warmup-epochs 20 \
# --resume '/OUTPUT_PATH/checkpoint-original-19.pth' --output /OUTPUT_PATH --batch-size 128 