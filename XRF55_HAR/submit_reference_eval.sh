#!/bin/bash
#SBATCH --job-name=xfi-bb-eval
#SBATCH --output=logs/reference_eval_%j.out
#SBATCH --error=logs/reference_eval_%j.err
#SBATCH --time=00:30:00
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
WEIGHTS="$PROJECT_ROOT/compass/code/XRF55_HAR/backbone_models"
OUTPUT="$RUN_DIR/reference_backbone_results.json"

cd "$RUN_DIR"
mkdir -p logs

nvidia-smi --query-gpu=name,memory.total --format=csv
"$PYTHON" evaluate_backbones.py \
  --dataset "$DATASET" \
  --weights-root "$WEIGHTS" \
  --batch-size 64 \
  --workers 8 \
  --output "$OUTPUT"
