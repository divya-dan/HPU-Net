#!/usr/bin/env bash
#SBATCH -J hpu_lidc
#SBATCH -o logs/hpu_lidc_%j.out
#SBATCH -e logs/hpu_lidc_%j.err
#SBATCH -p gpu              # <-- change to your partition/queue
#SBATCH --gres=gpu:1        # <-- number/type of GPUs
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 48:00:00

set -euo pipefail

# ---- ENV ----
# 1) venv (edit to your path)
VENV="${VENV:-$HOME/git/HPU-Net/.venv}"
source "$VENV/bin/activate"

# 2) project
PROJECT_ROOT="${PROJECT_ROOT:-$HOME/git/HPU-Net}"
cd "$PROJECT_ROOT"
mkdir -p logs runs

export PYTHONPATH="$PROJECT_ROOT/src"

# ---- RUN ----
OUTDIR="runs/hpu_lidc_$(date +%Y%m%d_%H%M%S)"
python -m hpunet.train.train_hpu \
  --config configs/train_hpu_lidc.json \
  --project-root "$PROJECT_ROOT" \
  --data-root "$PROJECT_ROOT/data/lidc_crops" \
  --max-steps 240000 \
  --outdir "$OUTDIR" \
  --save-name "hpu_last.pth"
