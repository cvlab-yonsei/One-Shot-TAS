# python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train.py --data-path '/dataset/ILSVRC2012' --gp \
# --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 460 --warmup-epochs 18 \
# --output /OUTPUT_PATH --batch-size 128 

# python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train.py --data-path '/dataset/ILSVRC2012' --gp \
# --resume '/OUTPUT_PATH/checkpoint-original-only192216-training-300-460-23.pth' --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 500 --warmup-epochs 0 \
# --output /OUTPUT_PATH --batch-size 128 
# -> checkpoint_paths = [output_dir / ('checkpoint-original-only192216-training-300-460-500-11-' + str((epoch+1)//20) + '.pth')]


python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train.py --data-path '/dataset/ILSVRC2012' --gp \
--resume '/OUTPUT_PATH/checkpoint-original-only192216-training-300-460-4.pth' --change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 460 --warmup-epochs 18 \
--output /OUTPUT_PATH --batch-size 128 