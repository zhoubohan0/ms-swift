## data processing
bash bash/make_dataset.sh -d stage_20260325

## SFT
```bash
export ckpt_dir=outputs/stage_20260325_sft/v1-20260324-224944/checkpoint-30
bash bash/train_sft.sh -d stage_20260325 -g 0
bash bash/infer_sft.sh -d stage_20260325 -b transformers -g 1 -c $ckpt_dir
bash bash/eval_sft.sh -c $ckpt_dir
```

核心代码片段

| 阶段 | 文件 | 位置 | 可检查内容 |
|------|------|------|------------|
| **1. 模型加载** | `swift/pipelines/train/sft.py` | `_prepare_model_tokenizer()` 内，`get_model_processor` 返回后 | `self.model`, `self.processor`，设备、dtype、参数量 |
| **2. 数据加载** | `swift/pipelines/train/sft.py` | `_get_dataset()` 内，`load_dataset()` 返回后 | `train_dataset`, `val_dataset`，条数、列名、首条样本 |
| **2b. 数据加载（底层）** | `swift/dataset/loader.py` | `load_dataset()` 末尾 return 前 | `train_datasets`, `val_datasets`（未 concat 前） |
| **3. 样本 batch** | `swift/trainers/seq2seq_trainer.py` | `training_step()` 入口 | `inputs`：当前 batch 的 input_ids、attention_mask、labels、pixel_values 等 |
| **4. 模型推理（前向）** | `swift/trainers/seq2seq_trainer.py` | `compute_loss()` 内，`model(**inputs)` 返回后 | `outputs`（logits、hidden_states），`inputs` 已 pop 掉 compute_loss_func 等 |
| **5. 损失计算** | `swift/trainers/seq2seq_trainer.py` | `compute_loss()` 内，return 前 | `loss`（标量或 per-token），`outputs.loss`，labels 与 mask |
| **6. 反向传播** | `swift/trainers/seq2seq_trainer.py` | `training_step()` 末尾 return 前 | 该 step 的 backward 已完成，可看 `self.state.global_step`、梯度是否已更新 |
| **7. 验证集指标** | `swift/trainers/seq2seq_trainer.py` | `evaluate()` 内，`super().evaluate()` 返回后 | `res`：eval loss、accuracy 等指标 dict |
| **8. 训练日志存储** | `swift/trainers/mixin.py` | `log()` 内 | `logs`：当前步/epoch 的 loss、lr 等；会写入 tensorboard/wandb 等 |
| **8b. 训练日志落盘** | `swift/pipelines/train/sft.py` | `_save_trainer_state()` 内，写 `logging.jsonl` 前 | `self.train_msg`、`state.log_history`、`jsonl_path` |

`per_device_train_batch_size=2` 时输入形状：

| 键 | 形状 | 含义 |
|----|------|------|
| `input_ids` | `[2, 962]` | 文本 token 序列。2=batch_size，962=当前 batch 的**序列长度**（pad 到一致）。 |
| `labels` | `[2, 962]` | 与 `input_ids` 对齐的标签；-100 为不计算 loss 的位置（如 prompt、padding）。 |
| `attention_mask` | `[2, 962]` | 1=有效 token，0=padding；与 `input_ids` 一一对应。 |
| `pixel_values` | `[4080, 1536]` | **图像 patch 特征**（packed）。4080=本 batch 内**所有图像的 patch 总数**，1536=视觉编码器输出维度（Qwen-VL 常见）。约 2040 patch/图 → 2 张图。 |
| `image_grid_thw` | `[2, 3]` | 每张图的**网格 (T,H,W)**。2=batch 中图像数，3=(T, H, W)：时间/高/宽（patch 数）。单图时 T 多为 1。 |
| `position_ids` | `[3, 2, 962]` | **多维度位置 id**（如 Qwen-VL 的 MRope）。3=位置维度数（如 image_x, image_y, text），2=batch_size，962=序列长度。 |
| `text_position_ids` | `[2, 962]` | 仅**文本部分**的位置 id，用于 loss mask / 序列划分。2=batch，962=seq_len。 |


输出形状：
| 键 | 形状 | 含义 |
|----|------|------|


loss torch.Size([])
logits torch.Size([2, 962, 151936])