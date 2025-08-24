#!/usr/bin/env bash
#SBATCH -J spu_lidc
#SBATCH -o logs/spu_lidc_%j.out
#SBATCH -e logs/spu_lidc_%j.err
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 48:00:00

set -euo pipefail

VENV="${VENV:-$HOME/git/HPU-Net/.venv}"
source "$VENV/bin/activate"

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/git/HPU-Net}"
cd "$PROJECT_ROOT"
mkdir -p logs runs
export PYTHONPATH="$PROJECT_ROOT/src"

OUTDIR="runs/spu_lidc_$(date +%Y%m%d_%H%M%S)"
python -m hpunet.train.train_spu \
  --config configs/train_spu_lidc.json \
  --project-root "$PROJECT_ROOT" \
  --data-root "$PROJECT_ROOT/data/lidc_crops" \
  --max-steps 240000 \
  --outdir "$OUTDIR" \
  --save-name "spu_last.pth"
