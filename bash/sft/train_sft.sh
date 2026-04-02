#!/usr/bin/env bash
# SFT：单张 RGB 图 -> 结构化生成 stage / shape / orientation / task_completion

set -euo pipefail

usage() {
    echo "Usage: $0 [-d dataset_name] [-g gpu_id]"
}

SCRIPT_DIR=$(dirname "$(realpath "$0")")
REPO_ROOT=$(dirname "$(dirname "$SCRIPT_DIR")")

dataset_name="stage_20260325"
gpu_id="0"

while getopts ":d:g:h" opt; do
    case "$opt" in
        d) dataset_name="$OPTARG" ;;
        g) gpu_id="$OPTARG" ;;
        h)
            usage
            exit 0
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            usage >&2
            exit 1
            ;;
        :)
            echo "Option -$OPTARG requires an argument." >&2
            usage >&2
            exit 1
            ;;
    esac
done

DATASET_DIR="${REPO_ROOT}/datasets/processed/${dataset_name}"
DATASET_NAME=$(basename "${DATASET_DIR}")

if [[ ! -f "${DATASET_DIR}/train.jsonl" || ! -f "${DATASET_DIR}/val.jsonl" ]]; then
    echo "Dataset files not found in ${DATASET_DIR}" >&2
    exit 1
fi

CUDA_VISIBLE_DEVICES="${gpu_id}" \
swift sft \
    --model "Qwen/Qwen3-VL-4B-Instruct" \
    --tuner_type lora \
    --dataset "${DATASET_DIR}/train.jsonl" "${DATASET_DIR}/val.jsonl" \
    --torch_dtype bfloat16 \
    --num_train_epochs 100 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
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
