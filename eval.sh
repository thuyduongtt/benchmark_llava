#!/bin/bash

# convert dataset into jsonl format
python convert.py --path_to_ds ${DS_NAME} --output_jsonl_name ${DS_NAME}

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT="llava-v1.5-13b"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path LLaVA/llava-v1.5-13b \
        --question-file ${DS_NAME}.jsonl \
        --image-folder ${DS_NAME} \
        --answers-file answers/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=answers/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat answers/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done


