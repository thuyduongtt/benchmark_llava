#!/bin/bash

DS_NAME="ReasonVQA"
DS_VERSION="unbalanced"
DS_DIR="../dataset/${DS_VERSION}"

DS_NAME=$DS_NAME DS_DIR=$DS_DIR IMG_DIR=$DS_DIR CUDA_VISIBLE_DEVICES=0 ./benchmark.sh
