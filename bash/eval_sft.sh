#!/usr/bin/env bash
# 对 ms-swift 推理输出的 infer_*.jsonl：按 response vs labels 各字段输出 classification_report。
# 用法: ./eval.sh [eval.jsonl]
#       ./eval.sh -j /path/to/eval.jsonl [-o report.txt]

set -euo pipefail

SCRIPT_DIR=$(dirname "$(realpath "$0")")
REPO_ROOT=$(dirname "$SCRIPT_DIR")


usage() {
  echo "Usage: $0 [-c ckpt_dir]"
  echo "       $0 [ckpt_dir]   # 兼容：唯一参数视为 ckpt_dir"
}

CKPT_DIR=""
while getopts ":c:h" opt; do
  case "$opt" in
    c) CKPT_DIR="$OPTARG" ;;
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
shift $((OPTIND - 1)) || true
if [[ $# -ge 1 ]]; then
  CKPT_DIR="$1"
fi

if [[ -z "$CKPT_DIR" ]]; then
  echo "Missing ckpt_dir: use -c DIR or pass DIR as the first argument." >&2
  usage >&2
  exit 1
fi

RUN_DIR=$(dirname "$CKPT_DIR")
JSONL="${RUN_DIR}/eval.jsonl"
OUT_FILE="${RUN_DIR}/eval.txt"

if [[ ! -f "$JSONL" ]]; then
  echo "File not found: $JSONL" >&2
  exit 1
fi

run_py() {
  python3 - "$JSONL" <<'PY'
import json
import re
import sys
from pathlib import Path

from sklearn.metrics import classification_report, confusion_matrix

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")

KEYS = ("shape", "orientation", "completion", "stage")
LABEL_ORDER = {
    "shape": ["messy", "flatten", "folded"],
    "orientation": ["horizontal", "vertical", "unknown"],
    "completion": ["yes", "no"],
    "stage": ["1", "2", "3", "4", "5", "6"],
}


def unwrap_json_text(s: str) -> str:
    s = s.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", s, re.IGNORECASE)
    if m:
        s = m.group(1).strip()
    return s


def norm_stage(v):
    if v is None:
        return "__missing__"
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float) and v == int(v):
        return str(int(v))
    s = str(v).strip()
    if s.isdigit():
        return str(int(s))
    return s


def norm_field(key, v):
    if v is None:
        return "__missing__"
    if key == "stage":
        return norm_stage(v)
    return str(v).strip()


lines = [ln for ln in text.splitlines() if ln.strip()]
rows_ok = []
resp_parse_fail = 0
label_parse_fail = 0

for i, line in enumerate(lines):
    try:
        row = json.loads(line)
    except json.JSONDecodeError as e:
        print(f"[行 {i+1}] 整行 JSON 解析失败: {e}", file=sys.stderr)
        continue
    rs = row.get("response")
    ls = row.get("labels")
    if rs is None or ls is None:
        print(f"[行 {i+1}] 缺少 response 或 labels", file=sys.stderr)
        continue
    try:
        pred = json.loads(unwrap_json_text(rs))
    except (json.JSONDecodeError, TypeError) as e:
        resp_parse_fail += 1
        print(f"[行 {i+1}] response 解析失败: {e}", file=sys.stderr)
        continue
    try:
        gold = json.loads(unwrap_json_text(ls))
    except (json.JSONDecodeError, TypeError) as e:
        label_parse_fail += 1
        print(f"[行 {i+1}] labels 解析失败: {e}", file=sys.stderr)
        continue
    rows_ok.append((pred, gold))

n = len(rows_ok)
print(f"文件: {path}")
print(f"有效样本数: {n} (response 解析失败: {resp_parse_fail}, labels 解析失败: {label_parse_fail})")
print()

# 四字段全一致
exact = 0
for pred, gold in rows_ok:
    ok = True
    for k in KEYS:
        if norm_field(k, pred.get(k)) != norm_field(k, gold.get(k)):
            ok = False
            break
    if ok:
        exact += 1
if n:
    print(f"严格一致 (shape+orientation+completion+stage 全对): {exact}/{n} = {exact/n:.4f}")
else:
    print("严格一致: N/A (无有效样本)")
print()
print("=" * 72)

for key in KEYS:
    y_true = []
    y_pred = []
    for pred, gold in rows_ok:
        y_true.append(norm_field(key, gold.get(key)))
        y_pred.append(norm_field(key, pred.get(key)))

    labels = list(LABEL_ORDER[key])
    extras = sorted(set(y_true) | set(y_pred))
    for x in extras:
        if x not in labels and x != "__missing__":
            labels = labels + [x]
    if "__missing__" in y_true or "__missing__" in y_pred:
        if "__missing__" not in labels:
            labels.append("__missing__")

    print()
    print(f"### 字段: {key}")
    print(f"(labels 顺序优先: {LABEL_ORDER[key]})")
    print()
    if not n:
        print("(无样本，跳过)")
        continue
    print(classification_report(y_true, y_pred, labels=labels, digits=4, zero_division=0))
    print("Confusion matrix (行=真实 labels, 列=预测):")
    print("labels:", labels)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print(cm)
    print("-" * 72)

PY
}

run_py | tee "$OUT_FILE"
