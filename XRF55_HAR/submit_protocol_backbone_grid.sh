#!/bin/bash
#SBATCH --job-name=xfi-new-grid
#SBATCH --export=ALL
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:h100:1
#SBATCH --partition=research
#SBATCH --mem=64G
set -euo pipefail
RUN_DIR="${XFI_RUN_DIR:-${SLURM_SUBMIT_DIR:?}}"
PYTHON="${XFI_PYTHON:-$RUN_DIR/../../compass/.venv/bin/python}"
CAMPAIGN="${XFI_CAMPAIGN:?set XFI_CAMPAIGN to the prepared campaign directory}"
TASK_ID=$(( ${SLURM_ARRAY_TASK_ID:?} + ${XFI_TASK_OFFSET:-0} ))
cd "$RUN_DIR"
exec "$PYTHON" -u protocol_backbone_grid.py --campaign "$CAMPAIGN" --task "$TASK_ID"
