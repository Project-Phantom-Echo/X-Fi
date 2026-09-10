#!/bin/bash
#SBATCH --job-name=xfi-selected-test
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:h100:1
#SBATCH --partition=research
#SBATCH --mem=64G
#SBATCH --array=0-26%9
set -euo pipefail
cd "${XFI_RUN_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
exec "${XFI_PYTHON:-python}" -u test_selected_backbones.py \
  --task "${SLURM_ARRAY_TASK_ID:?array required}" \
  --raw-root "${XRF55_RAW_ROOT:?set raw dataset path}" \
  --output-root "${XFI_TEST_OUTPUT:-$PWD/backbone_tests}" \
  --workers "${SLURM_CPUS_PER_TASK:-12}"
