#!/usr/bin/env bash

set -euo pipefail

usage() {
    echo "Usage: $0 [-d dataset_name]"
}

SCRIPT_DIR=$(dirname "$(realpath "$0")")
REPO_ROOT=$(dirname "$SCRIPT_DIR")

dataset_name="stage_20260325"

while getopts ":d:h" opt; do
    case "$opt" in
        d) dataset_name="$OPTARG" ;;
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

python "${REPO_ROOT}/datasets/make_cloth_dataset.py" -d "${dataset_name}"