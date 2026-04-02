#!/usr/bin/env bash
# GRPO：cloth_debug 多模态 JSON 结构化输出，奖励为与 solution 字段匹配的 cloth_json_match。
# 文档：README.md（强化学习 / GRPO）、docs/source/Instruction/GRPO/GetStarted/GRPO.md
# 依赖：与 README 一致需 vLLM（colocate 采样）；显存不足可改为去掉 --use_vllm 及相关 vllm_* 参数，由 Transformers 引擎采样（较慢）。
# TRL GRPO：全局 eval batch = GPU 数 × per_device_eval_batch_size，必须能被 num_generations 整除。
# 可选：在 SFT 权重上继续 GRPO 时，取消注释 --adapters 行（指向 SFT 的 checkpoint 目录）。

set -euo pipefail

usage() {
    echo "Usage: $0 [-d dataset_name] [-g gpu_id] [-a adapters_dir] [-r ref_adapters_dir] [-o output_dir]"
}

SCRIPT_DIR=$(dirname "$(realpath "$0")")
REPO_ROOT=$(dirname "$SCRIPT_DIR")

dataset_name="cloth_debug"
gpu_id="0"
adapters_dir="${REPO_ROOT}/outputs/cloth_debug_sft/v0-20260320-135729/checkpoint-3"
ref_adapters_dir="${REPO_ROOT}/outputs/cloth_debug_sft/v0-20260320-135729/checkpoint-3"
output_dir=""

while getopts ":d:g:a:r:o:h" opt; do
    case "$opt" in
        d) dataset_name="$OPTARG" ;;
        g) gpu_id="$OPTARG" ;;
        a) adapters_dir="$OPTARG" ;;
        r) ref_adapters_dir="$OPTARG" ;;
        o) output_dir="$OPTARG" ;;
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
REWARD_PLUGIN="${REPO_ROOT}/bash/grpo_cloth_reward_plugin.py"
OUTPUT_DIR="${output_dir:-${REPO_ROOT}/outputs/${DATASET_NAME}_grpo}"

if [[ ! -f "${DATASET_DIR}/train.jsonl" || ! -f "${DATASET_DIR}/val.jsonl" ]]; then
    echo "Dataset files not found in ${DATASET_DIR}" >&2
    exit 1
fi

CUDA_VISIBLE_DEVICES="${gpu_id}" \
MAX_PIXELS=1003520 \
VIDEO_MAX_PIXELS=50176 \
FPS_MAX_FRAMES=12 \
swift rlhf \
    --rlhf_type grpo \
    --model "Qwen/Qwen3-VL-4B-Instruct" \
    --tuner_type lora \
    --lora_rank 16 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --dataset "${DATASET_DIR}/train.jsonl" \
    --val_dataset "${DATASET_DIR}/val.jsonl" \
    --external_plugins "${REWARD_PLUGIN}" \
    --reward_funcs cloth_json_match \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.45 \
    --vllm_max_model_len 8192 \
    --sleep_level 1 \
    --max_length 2048 \
    --max_completion_length 512 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --save_steps 5 \
    --save_total_limit 2 \
    --logging_steps 1 \
    --eval_steps 5 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 2 \
    --dataset_num_proc 2 \
    --remove_unused_columns false \
    --num_generations 4 \
    --temperature 0.9 \
    --top_p 0.95 \
    --output_dir "${OUTPUT_DIR}" \
    --log_completions true \
    --adapters "${adapters_dir}" \
    --ref_adapters "${ref_adapters_dir}" \
# 可选：在 SFT LoRA 上继续 GRPO（需同时指定 policy 与 ref）：
