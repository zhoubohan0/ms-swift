#!/usr/bin/env bash

set -euo pipefail

usage() {
  echo "Usage: $0 [-d dataset_name] [-c ckpt_dir] [-b infer_backend] [-g gpu_id] [-o out_jsonl]"
}

SCRIPT_DIR=$(dirname "$(realpath "$0")")
REPO_ROOT=$(dirname "$(dirname "$SCRIPT_DIR")")

dataset_name="stage_20260325"
CKPT_DIR="${REPO_ROOT}/outputs/stage_20260325_sft/v1-20260324-224944/checkpoint-30"
infer_backend="transformers"
gpu_id="1"
out_jsonl=""

while getopts ":d:c:b:g:o:h" opt; do
  case "$opt" in
    d) dataset_name="$OPTARG" ;;
    c) CKPT_DIR="$OPTARG" ;;
    b) infer_backend="$OPTARG" ;;
    g) gpu_id="$OPTARG" ;;
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

DATASET_DIR="${REPO_ROOT}/datasets/processed/${dataset_name}"
VAL_JSONL="${DATASET_DIR}/val.jsonl"
OUT_JSONL="${out_jsonl:-"${CKPT_DIR}-eval_${dataset_name}.jsonl"}"

if [[ ! -f "${VAL_JSONL}" ]]; then
  echo "Validation dataset not found: ${VAL_JSONL}" >&2
  exit 1
fi

if [[ ! -d "${CKPT_DIR}" ]]; then
  echo "Checkpoint directory not found: ${CKPT_DIR}" >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES="${gpu_id}" \
MAX_PIXELS=1003520 \
VIDEO_MAX_PIXELS=50176 \
FPS_MAX_FRAMES=12 \
swift infer \
  --merge_lora true \
  --adapters "${CKPT_DIR}" \
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

# 说明：checkpoint 的 args.json 里 val_dataset 常为 []，--load_data_args true 会覆盖 CLI 的--val_dataset 并进入交互模式，故此处用 load_data_args false 并手动对齐 template/max_length。
# 交互式 LoRA+vLLM 示例：--adapters .../checkpoint-3 --merge_lora true --infer_backend vllm --stream true
