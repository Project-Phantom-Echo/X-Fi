#!/bin/bash
# One protocol/modality/configuration/seed per task:
#   3 protocols x 3 modalities x 3 batches x 3 learning rates x 2 decays x
#   2 dropouts x 2 seeds = 648 tasks.
# Submit the complete search with: sbatch --array=0-647%24 submit_backbone_grid.sh
#
# The exact released point is present for every modality. Wi-Fi and RFID use
# batch/LR 32/1e-3 and 16/1e-3 respectively; mmWave uses 16/1e-4. All three use
# AdamW's released implicit decay 0.01 and the unchanged head (dropout 0).
#SBATCH --job-name=xfi-bb-grid
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

MODALITIES=(wifi rfid mmwave)
PROTOCOLS=(unseen_subjects_validation unseen_scenes unseen_scene_unseen_subject)
BATCH_SIZES=(16 32 64)
LEARNING_RATES=(1e-4 3e-4 1e-3)
WEIGHT_DECAYS=(0.01 0.3)
DROPOUTS=(0.0 0.3)
SEEDS=(1 2)

RUN_DIR="${XFI_RUN_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
PYTHON="${XFI_PYTHON:-python}"
RAW_ROOT="${XRF55_RAW_ROOT:?set XRF55_RAW_ROOT to the raw XRF55 dataset}"
PROTOCOL_CODE="$RUN_DIR/protocol_snapshot"
RUNNER_SHA256="9773caa152496c2f0233443dcdf5455fcef4e1c542e4255af9a649a642b059f8"
ID="${SLURM_ARRAY_TASK_ID:?submit_backbone_grid.sh must run as an array}"

SEED="${SEEDS[$((ID % 2))]}"
DROPOUT="${DROPOUTS[$((ID / 2 % 2))]}"
WEIGHT_DECAY="${WEIGHT_DECAYS[$((ID / 4 % 2))]}"
LEARNING_RATE="${LEARNING_RATES[$((ID / 8 % 3))]}"
BATCH_SIZE="${BATCH_SIZES[$((ID / 24 % 3))]}"
MODALITY="${MODALITIES[$((ID / 72 % 3))]}"
PROTOCOL="${PROTOCOLS[$((ID / 216 % 3))]}"
RUN_NAME="$PROTOCOL-$MODALITY-bs$BATCH_SIZE-lr$LEARNING_RATE-wd$WEIGHT_DECAY-do$DROPOUT-s$SEED"
EPOCHS="${XFI_GRID_EPOCHS:-100}"
if [[ "$EPOCHS" == 100 && -z "${XFI_GRID_MAX_TRAIN_BATCHES:-}" && -z "${XFI_GRID_MAX_VALIDATION_BATCHES:-}" ]]; then
  CAMPAIGN="$PROTOCOL"
else
  CAMPAIGN="smoke_job${SLURM_ARRAY_JOB_ID:-manual}/$PROTOCOL"
fi
OUTPUT="$RUN_DIR/backbone_grid/$CAMPAIGN/$MODALITY/$RUN_NAME"
EXTRA_ARGS=()
if [[ -n "${XFI_GRID_MAX_TRAIN_BATCHES:-}" ]]; then
  EXTRA_ARGS+=(--max-train-batches "$XFI_GRID_MAX_TRAIN_BATCHES")
fi
if [[ -n "${XFI_GRID_MAX_VALIDATION_BATCHES:-}" ]]; then
  EXTRA_ARGS+=(--max-validation-batches "$XFI_GRID_MAX_VALIDATION_BATCHES")
fi

cd "$RUN_DIR"
mkdir -p logs
if [[ -e "$OUTPUT/result.json" ]]; then
  echo "completed output exists: $OUTPUT" >&2
  exit 1
fi
mkdir -p "$OUTPUT"
read -r ACTUAL_RUNNER_SHA256 _ < <(sha256sum search_backbone.py)
if [[ "$ACTUAL_RUNNER_SHA256" != "$RUNNER_SHA256" ]]; then
  echo "search_backbone.py changed after submission; refusing mixed runs" >&2
  exit 1
fi
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} task=$ID run=$RUN_NAME"
nvidia-smi --query-gpu=name,memory.total --format=csv

exec "$PYTHON" -u search_backbone.py \
    --raw-root "$RAW_ROOT" \
    --protocol-code-dir "$PROTOCOL_CODE" \
    --output-dir "$OUTPUT" \
    --protocol "$PROTOCOL" \
    --modality "$MODALITY" \
    --batch-size "$BATCH_SIZE" \
    --learning-rate "$LEARNING_RATE" \
    --weight-decay "$WEIGHT_DECAY" \
    --dropout "$DROPOUT" \
    --seed "$SEED" \
    --epochs "$EPOCHS" \
    --workers 12 \
    "${EXTRA_ARGS[@]}"
