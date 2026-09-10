#!/bin/bash
#SBATCH --job-name=xfi-unified-full
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --partition=research
#SBATCH --mem=64G

set -euo pipefail

PROJECT_ROOT="${XFI_PROJECT_ROOT:-/mnt/weka/fgeikyan/rf-perception-papers}"
RUN_DIR="${XFI_RUN_DIR:-$PROJECT_ROOT/x-fi/XRF55_HAR}"
PYTHON="${XFI_PYTHON:-$PROJECT_ROOT/compass/.venv/bin/python}"
DATASET="${XFI_DATASET:-$PROJECT_ROOT/compass/data/XRF55}"
BACKBONES="${XFI_BACKBONES:-$PROJECT_ROOT/compass/XRF55_HAR/backbone_models}"
OUTPUT="$RUN_DIR/unified_runs/full_job${SLURM_JOB_ID}"

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
    --seed 3407

"$PYTHON" validate_all.py \
    --dataset "$DATASET" \
    --backbone-root "$BACKBONES" \
    --backbone-source released \
    --pt_weights "$OUTPUT/checkpoint_final.pth"
