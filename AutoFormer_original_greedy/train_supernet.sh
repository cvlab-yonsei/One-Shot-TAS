# python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train_original.py --data-path '/data' --gp \
# --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 500 --warmup-epochs 20 \
# --resume '/OUTPUT_PATH/checkpoint-original-19.pth' --output /OUTPUT_PATH --batch-size 128 

##################

# python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train_sn_not_original.py --data-path '/data' --gp \
# --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 500 --warmup-epochs 20 \
# --output /OUTPUT_PATH --batch-size 128 \


# python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train_sn1.py --data-path '/data' --gp \
# --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 500 --warmup-epochs 20 \
# --resume '/OUTPUT_PATH/checkpoint-original-19.pth' --output /OUTPUT_PATH --batch-size 128 \
# --save_checkpoint_path 'checkpoint-sn-400-interval-5-top-' --save_log_path './log/supernet_greedy_spectral_norm_400ep_interval_5_topk.log' --interval 5

# python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train_sn1.py --data-path '/data' --gp \
# --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 500 --warmup-epochs 20 \
# --resume '/OUTPUT_PATH/checkpoint-sn-not-original-0-prenas-aug21.pth' --output /OUTPUT_PATH --batch-size 128 \
# --save_checkpoint_path 'checkpoint-sn-440-interval-1-top(data_aug)-' --save_log_path './log/supernet_greedy_spectral_norm_440ep_interval_1_topk(data_aug).log' --interval 1

# python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution.py --data-path '/data' --gp \
# --change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --resume '/OUTPUT_PATH/checkpoint-sn-440-interval-1-top(data_aug)-25.pth' \
# --min-param-limits 5 --param-limits 6 \
# --log-file-path './log/search_sn_not_original_440-interval-1-top(data_aug)_6M.log'

# python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train_sn1.py --data-path '/data' --gp \
# --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 500 --warmup-epochs 20 \
# --resume '/OUTPUT_PATH/checkpoint-sn-not-original-0-prenas-aug22.pth' --output /OUTPUT_PATH --batch-size 128 \
# --save_checkpoint_path 'checkpoint-sn-460-interval-1-top(data_aug)-' --save_log_path './log/supernet_greedy_spectral_norm_460ep_interval_1_topk(data_aug).log' --interval 1

# python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution.py --data-path '/data' --gp \
# --change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --resume '/OUTPUT_PATH/checkpoint-sn-460-interval-1-top(data_aug)-25.pth' \
# --min-param-limits 5 --param-limits 6 \
# --log-file-path './log/search_sn_not_original_460-interval-1-top(data_aug)_6M.log'

# python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution.py --data-path '/data' --gp \
# --change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --resume '/OUTPUT_PATH/checkpoint-sn-400-interval-1-top(data_aug)-25.pth' \
# --min-param-limits 6 --param-limits 7 \
# --log-file-path './log/search_sn_not_original_400-interval-1-top(data_aug)_7M.log'

# python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution.py --data-path '/data' --gp \
# --change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --resume '/OUTPUT_PATH/checkpoint-sn-not-original-0-prenas-aug25.pth' \
# --min-param-limits 6 --param-limits 7 \
# --log-file-path './log/search_sn_not_original_500-top(data_aug)_7M.log'

# python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train_sn1.py --data-path '/data' --gp \
# --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 500 --warmup-epochs 20 \
# --resume '/OUTPUT_PATH/checkpoint-sn-not-original-0-prenas-aug23.pth' --output /OUTPUT_PATH --batch-size 128 \
# --save_checkpoint_path 'checkpoint-sn-480-interval-1-top(data_aug)-' --save_log_path './log/supernet_greedy_spectral_norm_480ep_interval_1_topk(data_aug).log' --interval 1

# python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution.py --data-path '/data' --gp \
# --change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --resume '/OUTPUT_PATH/checkpoint-sn-480-interval-1-top(data_aug)-25.pth' \
# --min-param-limits 5 --param-limits 6 \
# --log-file-path './log/search_sn_not_original_480-interval-1-top(data_aug)_6M.log'

# ######

# python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train_sn1.py --data-path '/data' --gp \
# --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 500 --warmup-epochs 20 \
# --resume '/OUTPUT_PATH/checkpoint-sn-not-original-0-prenas-aug19.pth' --output /OUTPUT_PATH --batch-size 128 \
# --save_checkpoint_path 'checkpoint-sn-400-interval-1-top(data_aug_pool)-' --save_log_path './log/supernet_greedy_spectral_norm_400ep_interval_1_topk(data_aug_pool).log' --interval 1

# python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution.py --data-path '/data' --gp \
# --change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --resume '/OUTPUT_PATH/checkpoint-sn-400-interval-1-top(data_aug_pool)-25.pth' \
# --min-param-limits 5 --param-limits 6 \
# --log-file-path './log/search_sn_not_original_400-interval-1-top(data_aug_pool)_6M.log'

# ######

# #####

# python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train_sn1.py --data-path '/data' --gp \
# --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 500 --warmup-epochs 20 \
# --resume '/OUTPUT_PATH/checkpoint-sn-not-original-0-prenas-aug22.pth' --output /OUTPUT_PATH --batch-size 128 \
# --save_checkpoint_path 'checkpoint-sn-460-interval-1-top(data_aug_pool)-' --save_log_path './log/supernet_greedy_spectral_norm_460ep_interval_1_topk(data_aug_pool).log' --interval 1

# python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution.py --data-path '/data' --gp \
# --change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --resume '/OUTPUT_PATH/checkpoint-sn-460-interval-1-top(data_aug_pool)-25.pth' \
# --min-param-limits 5 --param-limits 6 \
# --log-file-path './log/search_sn_not_original_460-interval-1-top(data_aug_pool)_6M.log'

####

####

# python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train_sn1.py --data-path '/data' --gp \
# --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 500 --warmup-epochs 20 \
# --resume '/OUTPUT_PATH/checkpoint-sn-not-original-0-prenas-aug19.pth' --output /OUTPUT_PATH --batch-size 128 \
# --save_checkpoint_path 'checkpoint-sn-400-interval-1-top(data_aug_pool)-' --save_log_path './log/supernet_greedy_spectral_norm_400ep_interval_1_topk(data_aug_pool).log' --interval 1

# python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution.py --data-path '/data' --gp \
# --change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --resume '/OUTPUT_PATH/checkpoint-sn-400-interval-1-top(data_aug_pool)-22.pth' \
# --min-param-limits 5 --param-limits 6 \
# --log-file-path './log/search_sn_not_original_400-460-interval-1-top(data_aug_pool)_6M.log'

####

# python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train_sn1.py --data-path '/data' --gp \
# --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 500 --warmup-epochs 20 \
# --resume '/OUTPUT_PATH/checkpoint-sn-not-original-0-prenas-aug22.pth' --output /OUTPUT_PATH --batch-size 128 \
# --save_checkpoint_path 'checkpoint-sn-460-interval-1-top(data_aug_pool_no_duplicate)-' --save_log_path './log/supernet_greedy_spectral_norm_460ep_interval_1_topk(data_aug_pool_no_duplicate).log' --interval 1

# python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train_sn1.py --data-path '/data' --gp \
# --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 500 --warmup-epochs 20 \
# --resume '/OUTPUT_PATH/checkpoint-sn-not-original-0-prenas-aug22.pth' --output /OUTPUT_PATH --batch-size 128 \
# --save_checkpoint_path 'checkpoint-sn-460-interval-1-top(data_aug_pool_no_duplicate_full_0.8_weight_decay_0.05)-' --save_log_path './log/supernet_greedy_spectral_norm_460ep_interval_1_topk(data_aug_pool_no_duplicate_full_0.8_weight_decay_0.05).log' --interval 1

# # python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution_random_init.py --data-path '/data' --gp \
# # --change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --resume '/OUTPUT_PATH/checkpoint-sn-not-original-0-prenas-aug18.pth' \
# # --min-param-limits 5 --param-limits 6 \
# # --log-file-path './log/search_sn_not_original_0-380-interval-1-top(data_aug_pool_init_random)_6M.log'

# python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train_sn1.py --data-path '/data' --gp \
# --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 500 --warmup-epochs 20 \
# --resume '/OUTPUT_PATH/checkpoint-sn-not-original-0-prenas-aug22.pth' --output /OUTPUT_PATH --batch-size 128 \
# --save_checkpoint_path 'checkpoint-sn-460-interval-1-top(data_aug_pool_no_duplicate_full_0.8_linear_weight_decay_0.05_cutmix)-' --save_log_path './log/supernet_greedy_spectral_norm_460ep_interval_1_topk(data_aug_pool_no_duplicate_full_0.8_linear_weight_decay_0.05_cutmix).log' --interval 1


# python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution.py --data-path '/data' --gp \
# --change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --resume '/OUTPUT_PATH/checkpoint-sn-460-interval-1-top(data_aug_pool_no_duplicate_full_0.8_linear_weight_decay_0.05_cutmix)-25.pth' \
# --min-param-limits 5 --param-limits 6 \
# --log-file-path './log/search_sn_460-interval-1-top(data_aug_pool_no_duplicate_real_linear_0.8_weight_decay_0.05_cutmix)_6M.log'


# python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution2.py --data-path '/data' --gp \
# --change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --resume '/OUTPUT_PATH/checkpoint-sn-460-interval-1-top(data_aug_pool_no_duplicate_full_0.8_weight_decay_0.05)-25.pth' \
# --min-param-limits 5 --param-limits 6 \
# --log-file-path './log/search_sn_460-interval-1-top(data_aug_pool_no_duplicate_linear_0.8_weight_decay_0.05)_6M.log'

# python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train_sn1.py --data-path '/data' --gp \
# --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 500 --warmup-epochs 20 \
# --resume '/OUTPUT_PATH/checkpoint-sn-not-original-0-prenas-aug22.pth' --output /OUTPUT_PATH --batch-size 128 \
# --save_checkpoint_path 'checkpoint-sn-460-interval-1-top(data_aug_pool_no_duplicate_full_0.8_linear_weight_decay_0.05)-' --save_log_path './log/supernet_greedy_spectral_norm_460ep_interval_1_topk(data_aug_pool_no_duplicate_full_0.8_linear_weight_decay_0.05).log' --interval 1

python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train_sn1.py --data-path '/data' --gp \
--change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 500 --warmup-epochs 20 \
--resume '/OUTPUT_PATH/checkpoint-sn-not-original-0-prenas-aug19.pth' --output /OUTPUT_PATH --batch-size 128 \
--save_checkpoint_path 'checkpoint-sn-midtraining3-random-400-interval-1-top(data_aug_pool_no_duplicate_full_0.8_linear)-' --save_log_path './log/supernet_midtraining3-random_greedy_spectral_norm_400ep_interval_1_topk(data_aug_pool_no_duplicate_full_0.8_linear).log' --interval 1


python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution.py --data-path '/data' --gp \
--change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --resume '/OUTPUT_PATH/checkpoint-sn-midtraining3-random-400-interval-1-top(data_aug_pool_no_duplicate_full_0.8_linear)-25.pth' \
--min-param-limits 5 --param-limits 6 \
--log-file-path './log/search_sn_midtraining3-random-400-interval-1-top(data_aug_pool_no_duplicate_real_linear_0.8)_6M.log'

# python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution.py --data-path '/data' --gp \
# --change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --resume '/OUTPUT_PATH/checkpoint-sn-460-interval-1-top(data_aug_pool_no_duplicate_full_0.8_linear_weight_decay_0.05)-25.pth' \
# --min-param-limits 5 --param-limits 6 \
# --log-file-path './log/search_sn_460-interval-1-top(data_aug_pool_no_duplicate_real_linear_0.8_weight_decay_0.05)_6M.log'


# 여기 하기 전에 evolution.py에서 pkl 파일 바꾸기.
# python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution2.py --data-path '/data' --gp \
# --change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --resume '/OUTPUT_PATH/checkpoint-sn-300-interval-1-top(data_aug_pool_no_duplicate_full_0.8)-25.pth' \
# --min-param-limits 5 --param-limits 6 \
# --log-file-path './log/search_sn_300-500-interval-1-top(data_aug_pool_no_duplicate_linear_0.8)_6M.log'

# python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution2.py --data-path '/data' --gp \
# --change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --resume '/OUTPUT_PATH/checkpoint-sn-300-interval-1-top(data_aug_pool_no_duplicate_full_0.8)-25.pth' \
# --min-param-limits 6 --param-limits 7 \
# --log-file-path './log/search_sn_300-500-interval-1-top(data_aug_pool_no_duplicate_linear_0.8)_7M.log'

# python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution.py --data-path '/data' --gp \
# --change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --resume '/OUTPUT_PATH/checkpoint-sn-460-interval-1-top(data_aug_pool_no_duplicate)-25.pth' \
# --min-param-limits 5 --param-limits 6 \
# --log-file-path './log/search_sn_not_original_460-interval-1-top(data_aug_pool_init_random)_6M.log'

# python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train_sn1.py --data-path '/data' --gp \
# --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 500 --warmup-epochs 20 \
# --resume '/OUTPUT_PATH/checkpoint-sn-not-original-0-prenas-aug22.pth' --output /OUTPUT_PATH --batch-size 128 \
# --save_checkpoint_path 'checkpoint-sn-460-interval-1-top(data_aug_pool_no_duplicate_full_0.8)-' --save_log_path './log/supernet_greedy_spectral_norm_460ep_interval_1_topk(data_aug_pool_no_duplicate_full_0.8).log' --interval 1

# python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution.py --data-path '/data' --gp \
# --change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --resume '/OUTPUT_PATH/checkpoint-sn-460-interval-1-top(data_aug_pool_no_duplicate_full_0.8)-25.pth' \
# --min-param-limits 5 --param-limits 6 \
# --log-file-path './log/search_sn_not_original_460-interval-1-top(data_aug_pool_init_random_full_0.8)_6M.log'

# python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution.py --data-path '/data' --gp \
# --change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --resume '/OUTPUT_PATH/checkpoint-sn-480-interval-1-top(data_aug)-25.pth' \
# --min-param-limits 6 --param-limits 7 \
# --log-file-path './log/search_sn_not_original_480-interval-1-top(data_aug)_7M.log'

# python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train_sn1.py --data-path '/data' --gp \
# --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 500 --warmup-epochs 20 \
# --resume '/OUTPUT_PATH/checkpoint-sn-not-original-0-prenas-aug14.pth' --output /OUTPUT_PATH --batch-size 128 \
# --save_checkpoint_path 'checkpoint-sn-300-interval-1-top(data_aug)-' --save_log_path './log/supernet_greedy_spectral_norm_300ep_interval_1_topk(data_aug).log' --interval 1

# python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution.py --data-path '/data' --gp \
# --change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --resume '/OUTPUT_PATH/checkpoint-sn-300-interval-1-top(data_aug)-25.pth' \
# --min-param-limits 5 --param-limits 6 \
# --log-file-path './log/search_sn_not_original_300-interval-1-top(data_aug)_6M.log'


# python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train_sn1.py --data-path '/data' --gp \
# --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 500 --warmup-epochs 20 \
# --resume '/OUTPUT_PATH/checkpoint-original-14.pth' --output /OUTPUT_PATH --batch-size 128 \
# --save_checkpoint_path 'checkpoint-sn-300-interval-1-top-' --save_log_path './log/supernet_greedy_spectral_norm_300ep_interval_1_topk.log' --interval 1

# top-k
# python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train_sn1.py --data-path '/data' --gp \
# --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 500 --warmup-epochs 20 \
# --resume '/OUTPUT_PATH/checkpoint-original-19.pth' --output /OUTPUT_PATH --batch-size 128 


# bottom-k
# python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train_sn2.py --data-path '/data' --gp \
# --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 500 --warmup-epochs 20 \
# --resume '/OUTPUT_PATH/checkpoint-original-19.pth' --output /OUTPUT_PATH --batch-size 128 