#!/bin/bash

eval "$(conda shell.bash hook)"
# export CUDA_VISIBLE_DEVICES=0
conda activate pytorch-from-scratch

export ACCELERATE_LOG_LEVEL="INFO"

# DEBUG
accelerate launch --gpu_ids 0, --num_processes 1 bert/train.py \
--output-dir /media/bryan/ssd01/expr/bert_from_scratch/debug01 \
--initial-dataset-cache-path /media/bryan/ssd01/expr/bert_from_scratch/debug01/initial_dataset_cache \
--max-dataset-cache-path /media/bryan/ssd01/expr/bert_from_scratch/debug01/max_dataset_cache \
--debug-dataset-sample-limit 1000000 \
--max-train-iters 100 --lr-warmup-iters 100

# --limit-val-iters 20


# pre-layer-norm configuration
# accelerate launch bert/train.py \
# --output-dir /media/bryan/ssd01/expr/bert_from_scratch/debug01 \
# --pre-layer-norm --lr 3e-4 --lr-warmup-iters 0