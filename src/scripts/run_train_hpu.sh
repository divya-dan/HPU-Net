#!/usr/bin/env bash
set -euo pipefail
PY="${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}"; PY="${PY:-$(command -v python)}"
export PYTHONPATH="$(pwd)/src"

exec "$PY" -m hpunet.train.train_hpu \
  --config configs/train_hpu_lidc.json \
  --project-root "$(pwd)" \
  --data-root "data/lidc_crops" \
  --max-steps 50 \
  --outdir "runs/hpu_smoke" \
  --save-name "hpu_last.pth"
