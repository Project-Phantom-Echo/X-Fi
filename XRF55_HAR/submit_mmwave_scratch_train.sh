#!/bin/bash
#SBATCH --job-name=xfi-mmwave
#SBATCH --output=logs/mmwave_scratch_%j.out
#SBATCH --error=logs/mmwave_scratch_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h100:1
#SBATCH --partition=research
#SBATCH --mem=64G

set -e

PROJECT_ROOT="/mnt/weka/fgeikyan/rf-perception-papers"
RUN_DIR="$PROJECT_ROOT/x-fi/XRF55_HAR"
PYTHON="$PROJECT_ROOT/compass/.venv/bin/python"
DATASET="$PROJECT_ROOT/compass/data/XRF55"
OUTPUT="$RUN_DIR/reproduced_backbones/seed3407_mmwave_scratch_job${SLURM_JOB_ID}"

cd "$RUN_DIR"
mkdir -p logs "$OUTPUT"

nvidia-smi --query-gpu=name,memory.total --format=csv
"$PYTHON" pretrain_backbones.py \
  --dataset "$DATASET" \
  --output-dir "$OUTPUT" \
  --modalities mmwave \
  --mmwave-from-scratch \
  --workers 16 \
  --seed 3407 \
  --save-eval-checkpoints
