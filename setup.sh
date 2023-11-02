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

# clone the repo but skip all LFS files (to avoid being stuck at "Filtering content")
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/liuhaotian/llava-v1.5-13b

# download LFS files
cd llava-v1.5-13b
curl -O https://huggingface.co/liuhaotian/llava-v1.5-13b/resolve/main/mm_projector.bin
curl -O https://huggingface.co/liuhaotian/llava-v1.5-13b/resolve/main/pytorch_model-00001-of-00003.bin
curl -O https://huggingface.co/liuhaotian/llava-v1.5-13b/resolve/main/pytorch_model-00002-of-00003.bin
curl -O https://huggingface.co/liuhaotian/llava-v1.5-13b/resolve/main/pytorch_model-00003-of-00003.bin
curl -O https://huggingface.co/liuhaotian/llava-v1.5-13b/resolve/main/tokenizer.model

# download eval.zip
cd ../playground/data
pip install --upgrade --no-cache-dir gdown
ZIP_ID='1atZSBBrAX54yYpxtVVW33zFvcnaHeFPy'
ZIP_NAME='eval.zip'
gdown $ZIP_ID -O $ZIP_NAME
unzip -q $ZIP_NAME
rm $ZIP_NAME



