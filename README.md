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