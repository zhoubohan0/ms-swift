#!/usr/bin/env bash

set -euo pipefail

usage() {
    echo "Usage: $0 [-d dataset_name] [-t train_ratio]"
}

SCRIPT_DIR=$(dirname "$(realpath "$0")")
REPO_ROOT=$(dirname "$SCRIPT_DIR")

dataset_name="stage_20260325"
train_ratio=0.95
while getopts ":d:t:h" opt; do
    case "$opt" in
        d) dataset_name="$OPTARG" ;;
        t) train_ratio="$OPTARG" ;;
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

python "${REPO_ROOT}/datasets/make_cloth_dataset.py" -d "${dataset_name}" -t "${train_ratio}"