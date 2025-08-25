#!/usr/bin/env bash
set -euo pipefail
PY="${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}"; PY="${PY:-$(command -v python)}"
export PYTHONPATH="$(pwd)/src"

exec "$PY" -m hpunet.train.train_spu \
  --config configs/train_spu_lidc.json \
  --project-root "$(pwd)" \
  --data-root "data/lidc_crops" \
  --max-steps 2000 \
  --outdir "runs/spu_2000" \
  --save-name "spu_last.pth"
