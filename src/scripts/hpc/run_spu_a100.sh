#!/bin/bash -l
#SBATCH --job-name=spu-a100
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --export=NONE

set -euo pipefail

# ---- USER PATHS (adapt if needed) ----
PROJECT_DIR="$HOME/projects/HPU-Net"
DATA_SRC="$WORK/data/iwi9140h-project/lidc_crops"
OUT_BASE="$WORK/runs/iwi9140h-project/spu"

# ---- ENV ----
module purge
module load python/3.12-conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hpu

# Keep CPU threads modest for DL unless you know better
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1

# Ensure out base exists
mkdir -p "$OUT_BASE"

# Unique run folder
TS=$(date +'%Y%m%d_%H%M%S')
RUN_ID="run_${TS}_$SLURM_JOB_ID"
RUN_DIR="$OUT_BASE/$RUN_ID"
mkdir -p "$RUN_DIR" "$RUN_DIR/logs"

# ---- Stage data to fast node-local SSD (auto cleaned after job) ----
STAGE_DIR="$TMPDIR/spu-${SLURM_JOB_ID}"
echo "[INFO] Staging data to $STAGE_DIR ..."
mkdir -p "$STAGE_DIR/data"
# rsync keeps directory structure and is resumable-ish
rsync -a --delete "$DATA_SRC/" "$STAGE_DIR/data/"

# ---- Run training ----
cd "$PROJECT_DIR"

# In case your code uses package-style imports, expose project root:
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

# Copy the exact config used into the run dir for provenance
cp "$PROJECT_DIR/configs/train_spu_lidc.json" "$RUN_DIR/"

echo "[INFO] Starting training..."
srun python -u src/hpunet/train/train_spu.py \
    --config "$RUN_DIR/train_spu_lidc.json" \
    --project-root "$PROJECT_DIR" \
    --data-root "$STAGE_DIR/data" \
    --outdir "$RUN_DIR" \
    --max-steps 240000 |& tee -a "$RUN_DIR/logs/train.log"

echo "[INFO] Done. Artifacts are in: $RUN_DIR"

