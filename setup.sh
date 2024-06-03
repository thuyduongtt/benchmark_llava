#!/bin/bash

# Please set up conda env with the name "llava" and python 3.10 before running this script

if [ $CONDA_DEFAULT_ENV != 'llava' ]
then
  echo "Please set up the conda environment with name 'llava' and python 3.10"
  exit
fi

git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA

# conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .

CKPT="llava-v1.5-13b"

# clone the repo but skip all LFS files (to avoid being stuck at "Filtering content")
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/liuhaotian/${CKPT}

# download LFS files
cd $CKPT

# If this script doesn't work, download these files manually and put into checkpoint folder (e.g. llava-v1.5-13b)
wget -q https://huggingface.co/liuhaotian/${CKPT}/resolve/main/mm_projector.bin -O mm_projector.bin
wget -q https://huggingface.co/liuhaotian/${CKPT}/resolve/main/pytorch_model-00001-of-00003.bin -O pytorch_model-00001-of-00003.bin
wget -q https://huggingface.co/liuhaotian/${CKPT}/resolve/main/pytorch_model-00002-of-00003.bin -O pytorch_model-00002-of-00003.bin
wget -q https://huggingface.co/liuhaotian/${CKPT}/resolve/main/pytorch_model-00003-of-00003.bin -O pytorch_model-00003-of-00003.bin
wget -q https://huggingface.co/liuhaotian/${CKPT}/resolve/main/tokenizer.model -O tokenizer.model

# download eval.zip
#cd ../playground/data
#mkdir eval
#cd eval
#pip install --upgrade --no-cache-dir gdown
#ZIP_ID='1atZSBBrAX54yYpxtVVW33zFvcnaHeFPy'
#ZIP_NAME='eval.zip'
#gdown $ZIP_ID -O $ZIP_NAME
#unzip -q $ZIP_NAME
#rm $ZIP_NAME



