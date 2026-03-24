#!/usr/bin/env bash

set -euo pipefail
ROOT="/mnt/oss/zbh/zbh/__backup/REPO/ms-swift"
CKPT_MERGED="${ROOT}/outputs/cloth_debug_sft/v0-20260320-135729/checkpoint-3-merged"
VAL_JSONL="${ROOT}/datasets/processed/cloth_debug/val.jsonl"
OUT_JSONL="${ROOT}/outputs/cloth_debug_sft/v0-20260320-135729/infer_val.jsonl"

export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

CUDA_VISIBLE_DEVICES=0 \
MAX_PIXELS=1003520 \
VIDEO_MAX_PIXELS=50176 \
FPS_MAX_FRAMES=12 \
swift infer \
  --model "${CKPT_MERGED}" \
  --load_data_args false \
  --val_dataset "${VAL_JSONL}" \
  --template qwen3_vl \
  --max_length 2048 \
  --remove_unused_columns false \
  --infer_backend transformers \
  --stream false \
  --temperature 0 \
  --max_new_tokens 2048 \
  --result_path "${OUT_JSONL}"

# 注意：若使用 --load_data_args true，checkpoint 的 args.json 里 val_dataset 常为 []，
# 会覆盖命令行传入的 --val_dataset，导致进入交互模式。此处显式关闭 load_data_args，
# 并手动对齐训练时的 template / max_length / remove_unused_columns。
# 若已安装 vLLM，可将 infer_backend 改为 vllm 并加上 vllm 相关参数，仿照下例：

# CUDA_VISIBLE_DEVICES=0 \
# swift infer \
#     --adapters outputs/cloth_debug_sft/v0-20260320-135729/checkpoint-3 \
#     --stream true \
#     --merge_lora true \
#     --infer_backend vllm \
#     --vllm_max_model_len 8192 \
#     --temperature 0 \
#     --max_new_tokens 2048