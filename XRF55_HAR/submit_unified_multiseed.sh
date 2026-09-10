#!/bin/bash
#SBATCH --job-name=xfi-unified-seeds
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=14:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --partition=research
#SBATCH --mem=64G
#SBATCH --array=0-1

set -euo pipefail

PROJECT_ROOT="${XFI_PROJECT_ROOT:-/mnt/weka/fgeikyan/rf-perception-papers}"
RUN_DIR="${XFI_RUN_DIR:-$PROJECT_ROOT/x-fi/XRF55_HAR}"
PYTHON="${XFI_PYTHON:-$PROJECT_ROOT/compass/.venv/bin/python}"
DATASET="${XFI_DATASET:-$PROJECT_ROOT/compass/data/XRF55}"
BACKBONES="${XFI_BACKBONES:-$PROJECT_ROOT/compass/XRF55_HAR/backbone_models}"
SEEDS=(1 2)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
OUTPUT="$RUN_DIR/unified_runs/seed${SEED}_job${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

cd "$RUN_DIR"
mkdir -p logs unified_runs
nvidia-smi --query-gpu=name,memory.total --format=csv
"$PYTHON" run.py \
    --dataset "$DATASET" \
    --backbone-root "$BACKBONES" \
    --backbone-source released \
    --output-dir "$OUTPUT" \
    --epochs 100 \
    --learning-rate 0.0001 \
    --train-batch-size 16 \
    --test-batch-size 32 \
    --workers 0 \
    --seed "$SEED"

"$PYTHON" validate_all.py \
    --dataset "$DATASET" \
    --backbone-root "$BACKBONES" \
    --backbone-source released \
    --pt_weights "$OUTPUT/checkpoint_final.pth"
