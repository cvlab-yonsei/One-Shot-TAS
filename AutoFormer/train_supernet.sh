python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train.py --data-path '/data' --gp \
--change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --epochs 500 --warmup-epochs 20 \
--resume './greedyTAS/checkpoint-0.pth' --output /OUTPUT_PATH --batch-size 128 
# --resume './greedyTAS/greedyTAS-epoch100-test/checkpoint-4.pth'
# --resume './greedyTAS/checkpoint-09121607.pth' 
# --resume './experiments/supernet/autoformer_t_500ep.pth'
# --resume './greedyTAS/greedyTAS-epoch20-test/checkpoint-0.pth'
# --resume './greedyTAS/greedyTAS-epoch59/checkpoint.pth'
# --resume './greedyTAS/checkpoint-24.pth' 