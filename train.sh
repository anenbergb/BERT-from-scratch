#!/bin/bash

eval "$(conda shell.bash hook)"
# export CUDA_VISIBLE_DEVICES=0
conda activate pytorch-from-scratch

export ACCELERATE_LOG_LEVEL="INFO"

# DEBUG
accelerate launch --gpu_ids 0, --num_processes 1 bert/train.py \
--output-dir /media/bryan/ssd01/expr/bert_from_scratch/debug01 \
--train-batch-size 32 --val-batch-size 32 \
--epochs 10 --lr-warmup-epochs 5 --limit-train-iters 10  --limit-val-iters 20