#!/bin/bash -l
#SBATCH --job-name=hpu-a100
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --export=NONE

set -euo pipefail

# ---- USER PATHS ----
PROJECT_DIR="$HOME/projects/HPU-Net"
DATA_SRC="$WORK/data/iwi9140h-project/lidc_crops"
OUT_BASE="$WORK/runs/iwi9140h-project/hpu"

# ---- ENV ----
module purge
module load python/3.12-conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hpu

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1

mkdir -p "$OUT_BASE"

# Unique run folder
TS=$(date +'%Y%m%d_%H%M%S')
RUN_ID="run_${TS}_$SLURM_JOB_ID"
RUN_DIR="$OUT_BASE/$RUN_ID"
mkdir -p "$RUN_DIR" "$RUN_DIR/logs"

# ---- Stage data to fast node-local SSD ----
STAGE_DIR="$TMPDIR/hpu-${SLURM_JOB_ID}"
echo "[INFO] Staging data to $STAGE_DIR ..."
mkdir -p "$STAGE_DIR/data"
rsync -a --delete "$DATA_SRC/" "$STAGE_DIR/data/"

# ---- Run training ----
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

# Copy exact config for provenance
cp "$PROJECT_DIR/configs/train_hpu_lidc.json" "$RUN_DIR/"

echo "[INFO] Starting HPU training..."
srun python -u src/hpunet/train/train_hpu.py \
    --config "$RUN_DIR/train_hpu_lidc.json" \
    --project-root "$PROJECT_DIR" \
    --data-root "$STAGE_DIR/data" \
    --outdir "$RUN_DIR" \
    --max-steps 240000 |& tee -a "$RUN_DIR/logs/train.log"

echo "[INFO] Done. Artifacts are in: $RUN_DIR"

