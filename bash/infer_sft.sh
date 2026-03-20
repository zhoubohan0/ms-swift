# GPU: 15G
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --adapters outputs/cloth_debug_sft/v0-20260320-135729/checkpoint-3 \
    --stream true \
    --merge_lora true \
    --infer_backend vllm \
    --vllm_max_model_len 8192 \
    --temperature 0 \
    --max_new_tokens 2048