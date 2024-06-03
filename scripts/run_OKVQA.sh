#!/bin/bash

DS_NAME="OKVQA"
DS_DIR="../dataset/${DS_NAME}"
IMG_DIR="../dataset/COCO/val2014"

DS_NAME=$DS_NAME DS_DIR=$DS_DIR IMG_DIR=$IMG_DIR CUDA_VISIBLE_DEVICES=0 ./benchmark.sh
