#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
bash scripts/check_dataset_size.sh ../../datasets/aflw2000
bash scripts/train.sh configs/default.yaml
bash scripts/eval.sh configs/default.yaml
bash scripts/report.sh configs/default.yaml
