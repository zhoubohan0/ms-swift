#!/bin/bash
# CHORD：SFT + GRPO 混合，单卡 48G，结构化输出 reward 用 cloth_json_match
# 先执行: python make_cloth_dataset.py
# 数据集需带 solution 列，reward 插件解析模型 JSON 与 solution 比较

DATASET_DIR="/mnt/oss/zbh/zbh/__backup/REPO/ms-swift/datasets/processed/cloth_debug"

DATASET_NAME=$(basename "${DATASET_DIR}")
CUDA_VISIBLE_DEVICES=0 \
NPROC_PER_NODE=1 \
swift rlhf \
    --rlhf_type grpo \
    --model "Qwen/Qwen3-VL-4B-Instruct" \
    --dataset "${DATASET_DIR}/train_grpo.jsonl" \
    --external_plugins "/mnt/oss/zbh/zbh/__backup/REPO/ms-swift/bash/grpo_reward_plugin.py" \
    --reward_funcs cloth_json_match \
    --load_from_cache_file true \
    --torch_dtype bfloat16 \
    --beta 0.01 \
    --steps_per_generation 2 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --chord_sft_per_device_train_batch_size 1 \
    --chord_sft_dataset "${DATASET_DIR}/train.jsonl" \
    --chord_enable_phi_function false \
    --chord_mu_warmup_steps 0 \
    --chord_mu_decay_steps 150 \
    --chord_mu_peak 0.85 \
    --chord_mu_valley 0.1 \
    --num_generations 4 \
    --tuner_type lora \
    --lora_rank 16 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.5 \
    --vllm_max_model_len 2048 \
    --max_completion_length 512 \
    --overlong_filter true \
    --save_steps 100 \
    --learning_rate 5e-6 \
    --save_total_limit 2 \
    --logging_steps 2 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 2 \
    --max_length 2048 \
    --output_dir "outputs/${DATASET_NAME}_chord" \
    --log_completions true \
    --remove_unused_columns false
