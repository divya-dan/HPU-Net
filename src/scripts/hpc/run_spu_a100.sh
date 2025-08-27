#!/bin/bash -l
#SBATCH --job-name=spu-a100
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --export=NONE

set -euo pipefail

# ---- Paths ----
PROJECT_DIR="$HOME/projects/HPU-Net"
DATA_SRC="$WORK/data/iwi9140h-project/lidc_crops"
OUT_BASE="$WORK/runs/iwi9140h-project/spu"

# ---- Env ----
module purge
module load python/3.12-conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hpu

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1
# SAFE even if PYTHONPATH is unset:
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"

# ---- Run folder ----
mkdir -p "$OUT_BASE"
TS=$(date +'%Y%m%d_%H%M%S')
RUN_ID="run_${TS}_$SLURM_JOB_ID"
RUN_DIR="$OUT_BASE/$RUN_ID"
mkdir -p "$RUN_DIR/logs"

# ---- Stage data to node-local SSD ----
STAGE_DIR="$TMPDIR/spu-${SLURM_JOB_ID}"
mkdir -p "$STAGE_DIR/data"
rsync -a --delete "$DATA_SRC/" "$STAGE_DIR/data/"

# ---- Build CSV manifests (relative to STAGE_DIR) ----
python -u "$PROJECT_DIR/src/scripts/build_manifests.py" \
  --data-root "$STAGE_DIR/data" \
  --out "$STAGE_DIR/data" \
  --project-root "$STAGE_DIR" \
  --splits train val test |& tee -a "$RUN_DIR/logs/manifests.log"

# ---- Copy the exact config used ----
cp "$PROJECT_DIR/configs/train_spu_lidc.json" "$RUN_DIR/"

# ---- Train ----
echo "[INFO] Starting SPU training at $(date)"
srun python -u "$PROJECT_DIR/src/hpunet/train/train_spu.py" \
  --config "$RUN_DIR/train_spu_lidc.json" \
  --project-root "$STAGE_DIR" \
  --data-root "$STAGE_DIR/data" \
  --outdir "$RUN_DIR" \
  --max-steps 240000 |& tee -a "$RUN_DIR/logs/train.log"

echo "[INFO] Finished at $(date)"
echo "[INFO] Artifacts: $RUN_DIR"
