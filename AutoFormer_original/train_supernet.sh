python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train.py --data-path '/dataset/ILSVRC2012' --gp \
--change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 460 --warmup-epochs 18 \
--output /OUTPUT_PATH --batch-size 128 