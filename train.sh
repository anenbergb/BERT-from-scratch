#!/bin/bash

eval "$(conda shell.bash hook)"
# export CUDA_VISIBLE_DEVICES=0
conda activate pytorch-from-scratch

export ACCELERATE_LOG_LEVEL="INFO"

# DEBUG
# accelerate launch --gpu_ids 0, --num_processes 1 bert/train.py \
# --output-dir /media/bryan/ssd01/expr/bert_from_scratch/debug01 \
# --initial-dataset-cache-path /media/bryan/ssd01/expr/bert_from_scratch/debug01/initial_dataset_cache \
# --max-dataset-cache-path /media/bryan/ssd01/expr/bert_from_scratch/debug01/max_dataset_cache \
# --debug-dataset-sample-limit 1000000 \
# --max-train-iters 50000 --lr-warmup-iters 5000

# rm -rf /media/bryan/ssd01/expr/bert_from_scratch/debug02
# accelerate launch --gpu_ids 0, --num_processes 1 bert/train.py \
# --output-dir /media/bryan/ssd01/expr/bert_from_scratch/debug02 \
# --resume-from-checkpoint /media/bryan/ssd01/expr/bert_from_scratch/debug01/checkpoints/checkpoint_10 \
# --debug-dataset-sample-limit 10000 \
# --lr 0 --max-train-iters 50 --evaluation-iters 10

# rm -rf /media/bryan/ssd01/expr/bert_from_scratch/long01
# accelerate launch --gpu_ids 0, --num_processes 1 bert/train.py \
# --output-dir /media/bryan/ssd01/expr/bert_from_scratch/long01 \
# --initial-dataset-cache-path /media/bryan/ssd01/expr/bert_from_scratch/dataset_cache_seq128_seed0 \
# --max-dataset-cache-path /media/bryan/ssd01/expr/bert_from_scratch/dataset_cache_seq512_seed0

# DEBUGGING
# rm -rf /media/bryan/ssd01/expr/bert_from_scratch/debug_seq128
# accelerate launch --gpu_ids 0, --num_processes 1 bert/train.py \
# --output-dir /media/bryan/ssd01/expr/bert_from_scratch/debug_seq128 \
# --train-only-with-initial-seq-len --val-batch-size 256 --evaluation-iters 200  \
# --pre-layer-norm --lr 3e-4 \
# --debug-dataset-sample-limit 100000
# --initial-dataset-cache-path /media/bryan/ssd01/expr/bert_from_scratch/dataset_cache_seq128_seed0 \

accelerate launch --gpu_ids 0, --num_processes 1 bert/train.py \
--output-dir /media/bryan/ssd01/expr/bert_from_scratch/1M_seq128 \
--initial-dataset-cache-path /media/bryan/ssd01/expr/bert_from_scratch/dataset_cache_seq128_seed0 \
--train-only-with-initial-seq-len --val-batch-size 256 \
--pre-layer-norm --lr 3e-4