# One-Shot-TAS

## How to Start

```
apt install python3.8-venv

cd ./AutoFormer

python3 -m venv {your_venv_name}

source {your_venv_name}/bin/activate

pip install -r requirements.txt


# if 'Pillow' error is occured...

sudo apt-get install python3-dev

pip install wheel

apt-get update

apt-get install build-essential

apt-get install libjpeg-dev

apt-get install libpng-dev libtiff-dev

pip install pillow==6.1.0

`pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html`

```

## Data Preparation 
You need to first download the [ImageNet-2012](http://www.image-net.org/) to the folder `./data/imagenet` and move the validation set to the subfolder `./data/imagenet/val`. To move the validation set, you cloud use the following script: <https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh>

The directory structure is the standard layout as following.
```
/path/to/imagenet/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class/2
      img4.jpeg
```

## Quick Start
We provide *Supernet Train, Search, Test* code of AutoFormer as follows.

### Supernet Train 

To train the supernet-T/S/B, we provided the corresponding supernet configuration files in `/experiments/supernet/`. For example, to train the supernet-B, you can run the following command. The default output path is `./`, you can specify the path with argument `--output`.

```buildoutcfg
python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train.py --data-path /PATH/TO/IMAGENT --gp \
--change_qk --relative_position --mode super --dist-eval --cfg ./experiments/supernet/supernet-B.yaml --epochs 500 --warmup-epochs 20 \
--output /OUTPUT_PATH --batch-size 128
```

### Search
We run our evolution search on part of the ImageNet training dataset and use the validation set of ImageNet as the test set for fair comparison. To generate the subImagenet in `/PATH/TO/IMAGENET`, you could simply run:
```buildoutcfg
python ./lib/subImageNet.py --data-path /PATH/TO/IMAGENT
```
 

After obtaining the subImageNet and training of the supernet. We could perform the evolution search using below command. Please remember to config the specific constraint in this evolution search using `--min-param-limits` and `--param-limits`: 
```buildoutcfg
python -m torch.distributed.launch --nproc_per_node=8 --use_env evolution.py --data-path /PATH/TO/IMAGENT --gp \
--change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-B.yaml --resume /PATH/TO/CHECKPOINT \
--min-param-limits YOUR/CONFIG --param-limits YOUR/CONFIG --data-set EVO_IMNET
```

### Test
To test our trained models, you need to put the downloaded model in `/PATH/TO/CHECKPOINT`. After that you could use the following command to test the model (Please change your config file and model checkpoint according to different models. Here we use the AutoFormer-B as an example).
```buildoutcfg
python -m torch.distributed.launch --nproc_per_node=8 --use_env supernet_train.py --data-path /PATH/TO/IMAGENT --gp \
--change_qk --relative_position --mode retrain --dist-eval --cfg ./experiments/subnet/AutoFormer-B.yaml --resume /PATH/TO/CHECKPOINT --eval 
```

## Acknowledgements

The codes are inspired by [Autoformer](https://github.com/microsoft/Cream/tree/main/AutoFormer), [tf-tas](https://github.com/decemberzhou/TF_TAS).

