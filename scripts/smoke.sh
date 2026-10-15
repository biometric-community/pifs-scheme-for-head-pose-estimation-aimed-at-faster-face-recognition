#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
python -c "import sheef; print('import ok', sheef.__version__)"
python -m sheef.train --help >/dev/null
python -m sheef.report --help >/dev/null
bash scripts/train.sh configs/smoke.yaml
bash scripts/eval.sh configs/smoke.yaml
bash scripts/report.sh configs/smoke.yaml
echo "SMOKE OK"
