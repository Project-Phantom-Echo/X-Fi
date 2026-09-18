#!/bin/bash
#SBATCH --job-name=xfi-final-test
#SBATCH --export=ALL
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:h100:1
#SBATCH --partition=research
#SBATCH --mem=64G
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=1
exec ../../compass/.venv/bin/python -u final_protocol_backbones.py --campaign "${XFI_FINAL_CAMPAIGN:?}" --task "${SLURM_ARRAY_TASK_ID:?}"
