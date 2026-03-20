#!/bin/bash
# SFT：单张 RGB 图 -> 结构化生成 stage / shape / orientation / task_completion
# 先生成数据集（若未生成）
# python make_cloth_dataset.py

DATASET_DIR="/mnt/oss/zbh/zbh/__backup/REPO/ms-swift/datasets/processed/cloth_debug"
# GPU: 9G
DATASET_NAME=$(basename "${DATASET_DIR}")
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model "Qwen/Qwen3-VL-4B-Instruct" \
    --tuner_type lora \
    --dataset "${DATASET_DIR}/train.jsonl" "${DATASET_DIR}/val.jsonl" \
    --torch_dtype bfloat16 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --learning_rate 1e-4 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 8 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir "outputs/${DATASET_NAME}_sft" \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --remove_unused_columns false
