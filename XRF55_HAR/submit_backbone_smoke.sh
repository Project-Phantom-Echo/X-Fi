#!/bin/bash
#SBATCH --job-name=xfi-bb-smoke
#SBATCH --output=logs/backbone_smoke_%j.out
#SBATCH --error=logs/backbone_smoke_%j.err
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --partition=research
#SBATCH --mem=64G

set -e

PROJECT_ROOT="/mnt/weka/fgeikyan/rf-perception-papers"
RUN_DIR="$PROJECT_ROOT/x-fi/code/XRF55_HAR"
PYTHON="$PROJECT_ROOT/compass/code/.venv/bin/python"
DATASET="$PROJECT_ROOT/compass/code/data/XRF55"
OUTPUT="$RUN_DIR/smoke_backbones/${SLURM_JOB_ID}"

cd "$RUN_DIR"
mkdir -p logs "$OUTPUT"

nvidia-smi --query-gpu=name,memory.total --format=csv
"$PYTHON" pretrain_backbones.py \
  --dataset "$DATASET" \
  --output-dir "$OUTPUT" \
  --modalities wifi rfid \
  --epochs 1 \
  --batch-size 4 \
  --workers 8 \
  --seed 3407 \
  --max-train-batches 2 \
  --max-test-batches 2
