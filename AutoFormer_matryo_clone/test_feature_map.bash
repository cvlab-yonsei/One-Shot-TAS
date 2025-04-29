python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_check_feature_map.py --data-path '/dataset/ILSVRC2012' --gp \
--change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 40 --warmup-epochs 3 \
--resume '/OUTPUT_PATH/checkpoint_original_check_only192_original_optimizer-epoch480-24.pth' --output /OUTPUT_PATH --batch-size 128 --input-size 224
