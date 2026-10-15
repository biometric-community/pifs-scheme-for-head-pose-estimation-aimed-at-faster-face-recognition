#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MONO="$(cd "$ROOT/../../.." && pwd)"
cd "$ROOT"
PYTHON="${MONO}/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="${PYTHON:-python3}"
fi
"$PYTHON" -m pip install -r requirements.txt
echo "OK: deps installed into $($PYTHON -c 'import sys; print(sys.prefix)')"
