#!/usr/bin/env bash
# GRPO 训练完成后，在 val.jsonl 上批量推理（vLLM + LoRA merge）。
# 将 GRPO_CKPT 改为实际 run 目录下的 checkpoint-*（内含 adapter_config.json）。

set -euo pipefail

usage() {
  echo "Usage: $0 [-d dataset_name] [-c ckpt_dir] [-g gpu_id] [-b infer_backend] [-o out_jsonl]"
}

SCRIPT_DIR=$(dirname "$(realpath "$0")")
REPO_ROOT=$(dirname "$SCRIPT_DIR")

dataset_name="cloth_debug"
gpu_id="0"
infer_backend="vllm"
GRPO_CKPT="${REPO_ROOT}/outputs/cloth_debug_grpo/v0-REPLACE_ME/checkpoint-REPLACE_ME"
out_jsonl=""

while getopts ":d:c:g:b:o:h" opt; do
  case "$opt" in
    d) dataset_name="$OPTARG" ;;
    c) GRPO_CKPT="$OPTARG" ;;
    g) gpu_id="$OPTARG" ;;
    b) infer_backend="$OPTARG" ;;
    o) out_jsonl="$OPTARG" ;;
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

VAL_JSONL="${REPO_ROOT}/datasets/processed/${dataset_name}/val.jsonl"
OUT_JSONL="${out_jsonl:-$(dirname "${GRPO_CKPT}")/eval_grpo.jsonl}"

if [[ ! -f "${VAL_JSONL}" ]]; then
  echo "Validation dataset not found: ${VAL_JSONL}" >&2
  exit 1
fi

if [[ ! -d "${GRPO_CKPT}" ]]; then
  echo "Checkpoint directory not found: ${GRPO_CKPT}" >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES="${gpu_id}" \
MAX_PIXELS=1003520 \
VIDEO_MAX_PIXELS=50176 \
FPS_MAX_FRAMES=12 \
swift infer \
  --merge_lora true \
  --adapters "${GRPO_CKPT}" \
  --load_data_args false \
  --val_dataset "${VAL_JSONL}" \
  --template qwen3_vl \
  --max_length 2048 \
  --remove_unused_columns false \
  --infer_backend "${infer_backend}" \
  --vllm_max_model_len 8192 \
  --stream false \
  --temperature 0 \
  --max_new_tokens 2048 \
  --result_path "${OUT_JSONL}"
