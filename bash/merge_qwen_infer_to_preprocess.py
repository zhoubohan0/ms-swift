#!/usr/bin/env python3
"""
Parse finetune eval JSONL results and merge into llm_preprocess_results JSON.
Fills the `llm` field in llm_stage_review.review for shape/orientation/completion/stage,
and overwrites stage_info.stage with the predicted stage.

Output: qwen_finetune_infer_all.json (same directory as source JSON).
"""
import json
import os
import argparse

FIELDS = ["shape", "orientation", "completion", "stage"]


def parse_jsonl(path):
    predictions = {}
    parse_errors = 0
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            image_path = record["images"][0]["path"]
            image_name = os.path.basename(image_path)
            try:
                response = json.loads(record["response"])
            except json.JSONDecodeError:
                parse_errors += 1
                print(f"  [WARN] Line {line_num}: failed to parse response JSON for {image_name}")
                continue
            predictions[image_name] = response
    if parse_errors:
        print(f"  [WARN] {parse_errors} lines had unparseable response JSON")
    return predictions


def merge(predictions, data):
    matched = 0
    unmatched_images = []

    for item_key, item_val in data["items"].items():
        if item_key not in predictions:
            unmatched_images.append(item_key)
            continue
        pred = predictions[item_key]
        review = item_val["llm_stage_review"]["review"]

        for field in FIELDS:
            pred_val = str(pred.get(field, ""))
            review[field]["llm"] = pred_val

        stage_val = pred.get("stage", "")
        try:
            item_val["stage_info"]["stage"] = int(stage_val)
        except (ValueError, TypeError):
            item_val["stage_info"]["stage"] = stage_val

        matched += 1

    return matched, unmatched_images


def main():
    parser = argparse.ArgumentParser(description="Merge finetune JSONL inference results into preprocess JSON")
    parser.add_argument("--jsonl_path", type=str,
                        default="/mnt/nas/zbh/__backup/REPO/ms-swift/outputs/stage_20260330_sft/v0-20260331-111104/checkpoint-1700-eval_real_20260401.jsonl")
    parser.add_argument("--src_json_path", type=str,
                        default="/mnt/nas/zbh/__backup/REPO/ms-swift/datasets/raw/real_20260401/llm_preprocess_results_all.json")
    args = parser.parse_args()

    predictions = parse_jsonl(args.jsonl_path)
    print(f"Parsed {len(predictions)} predictions from JSONL")

    with open(args.src_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data['items'])} items from source JSON")

    matched, unmatched = merge(predictions, data)
    print(f"Matched & updated: {matched} items")
    if unmatched:
        print(f"Unmatched (no prediction): {len(unmatched)} items")

    dst_json_path = os.path.join(os.path.dirname(args.src_json_path), "qwen_preprocess_all.json")
    with open(dst_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to: {dst_json_path}")

    print("\nSample (first 3 matched items):")
    count = 0
    for item_key, item_val in data["items"].items():
        if item_key in predictions:
            review = item_val["llm_stage_review"]["review"]
            stage = item_val["stage_info"]["stage"]
            print(f"  {item_key}: shape={review['shape']['llm']}, orientation={review['orientation']['llm']}, "
                  f"completion={review['completion']['llm']}, stage_review={review['stage']['llm']}, "
                  f"stage_info.stage={stage}")
            count += 1
            if count >= 3:
                break


if __name__ == "__main__":
    main()
