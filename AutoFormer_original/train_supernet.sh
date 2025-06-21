# python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train.py --data-path '/dataset/ILSVRC2012' --gp \
# --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 460 --warmup-epochs 18 \
# --output /OUTPUT_PATH --batch-size 128 

# python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train.py --data-path '/dataset/ILSVRC2012' --gp \
# --resume '/OUTPUT_PATH/checkpoint-original-only192216-training-300-460-23.pth' --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 500 --warmup-epochs 0 \
# --output /OUTPUT_PATH --batch-size 128 
# -> checkpoint_paths = [output_dir / ('checkpoint-original-only192216-training-300-460-500-11-' + str((epoch+1)//20) + '.pth')]


# python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train_small_prenas_aug.py --data-path '/dataset/ILSVRC2012' --gp \
# --resume '/OUTPUT_PATH/checkpoint-original-small-400-500-12-123-11.pth' --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-S.yaml --epochs 500 --warmup-epochs 20 \
# --output /OUTPUT_PATH --batch-size 128 

# python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train_base_prenas_aug.py --data-path '/dataset/ILSVRC2012' --gp \
# --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-B.yaml --epochs 500 --warmup-epochs 20 \
# --output /OUTPUT_PATH --batch-size 128 

# python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train_base_prenas_aug.py --data-path '/dataset/ILSVRC2012' --gp \
# --resume '/OUTPUT_PATH/checkpoint-original-base-200-500-12-123-16.pth' --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-B.yaml --epochs 500 --warmup-epochs 20 \
# --output /OUTPUT_PATH --batch-size 128 

python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train_small_prenas_aug.py --data-path '/dataset/ILSVRC2012' --gp \
--resume '/OUTPUT_PATH/checkpoint-original-small-400-500-12-123-9.pth' --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-S.yaml --epochs 500 --warmup-epochs 20 \
--output /OUTPUT_PATH --batch-size 128 

# python3 -m torch.distributed.launch --nproc_per_node=8 --use_env evolution.py --data-path '/dataset/ILSVRC2012' --gp \
# --change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-S.yaml --resume '/OUTPUT_PATH/checkpoint-original-small-400-500-12-123-25.pth' \
# --min-param-limits 22 --param-limits 23 \
# --log-file-path './log/search-checkpoint-original-small-400-500-12-123-25-23M.log'