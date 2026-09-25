# X-Fi backbone hyperparameter search

> Historical campaign log. References below to “current”, queued jobs, and
> epoch-100 selection describe those original campaigns. The latest selection
> uses EMA validation accuracy (α=0.1); use the shared
> [hyperparameter report](https://github.com/Project-Phantom-Echo/wireless-sensing/blob/main/docs/Filyas-notes/models-for-xrf55/hparam-tuning.md)
> and [final results](https://github.com/Project-Phantom-Echo/wireless-sensing/blob/main/docs/Filyas-notes/models-for-xrf55/final-results.md)
> for current results.

We train each released X-Fi backbone architecture from random initialization on our three validation protocols, to
answer whether their weak cross-subject and cross-room results are a
hyperparameter choice or a property of the models.

The ResNet implementations are unchanged; the runner uses our fixed protocol
loader and adds optional dropout before the classifier. All weights train from
scratch: this is hyperparameter tuning, not fine-tuning released checkpoints.
Optimizer and modality-specific schedules follow the recovered notebook recipes.
Their batch/LR/decay/dropout settings are included as reference cells.

## Grid

| Axis | Values |
|---|---|
| Batch size | 16, 32, 64 |
| Learning rate | 1·10⁻⁴, 3·10⁻⁴, 1·10⁻³ |
| Weight decay | 0.01, 0.3 |
| Dropout | 0.0, 0.3 |

36 cells per modality and protocol, two seeds each: 324 cells and **648 runs**.
Every run trains 100 epochs and is scored once, at the fixed final epoch, on the
protocol's validation split. Selection is the argmax of the two-seed mean. The
test split is never opened during the search.

What is *not* swept is as important. Each modality keeps X-Fi's own learning
rate schedule — `MultiStepLR([20, 40], γ=0.1)` for RFID, none for Wi-Fi and
mmWave — so a Wi-Fi cell at `1·10⁻³` holds that rate for all 100 epochs while
the RFID cell at the same value spends epochs 41-100 at `1·10⁻⁵`. The learning
rate axis therefore does not mean quite the same thing across modalities.

## Their hyperparameters are in the grid

X-Fi sets these per modality, recovered from the deleted preparation notebooks
in commit `68b5fcf`. Weight decay `0.01` is AdamW's default, which their code
takes by passing none; dropout `0.0` is the absence of dropout.

| Modality | Batch | Learning rate | Weight decay | Dropout | In grid |
|---|---:|---:|---:|---:|:--:|
| Wi-Fi | 32 | 1·10⁻³ | 0.01 | 0.0 | yes |
| RFID | 16 | 1·10⁻³ | 0.01 | 0.0 | yes |
| mmWave | 16 | 1·10⁻⁴ | 0.01 | 0.0 | yes |

Each lands on a grid point exactly, so their configuration is scored under the
same protocol, seeds and stopping rule as every cell it is compared against.

## Results

Validation accuracy %, two-seed mean. Chance is `1.82`.

| Protocol | Model | Their cell | Rank | Tuned winner | Tuned |
|---|---|---:|---:|---|---:|
| `unseen_subjects` | mmWave | 59.14 | 11/36 | bs 64, lr 3·10⁻⁴, wd 0.01, do 0.0 | **61.63** |
| | RFID | 6.04 | 8/36 | bs 16, lr 1·10⁻³, wd 0.3, do 0.3 | 7.94 |
| | Wi-Fi | 1.56 | 33/36 | bs 16, lr 1·10⁻³, wd 0.3, do 0.3 | 3.78 |
| `unseen_scenes` | mmWave | 5.07 | 25/36 | bs 32, lr 1·10⁻³, wd 0.01, do 0.0 | 7.66 |
| | Wi-Fi | 2.59 | 14/36 | bs 16, lr 1·10⁻³, wd 0.3, do 0.0 | 4.18 |
| | RFID | 2.09 | 4/36 | bs 32, lr 1·10⁻⁴, wd 0.01, do 0.3 | 2.36 |
| `unseen_scene_unseen_subject` | mmWave | 8.65 | 22/36 | bs 64, lr 1·10⁻³, wd 0.01, do 0.0 | **10.26** |
| | Wi-Fi | 2.55 | 5/36 | bs 64, lr 1·10⁻⁴, wd 0.3, do 0.3 | 3.23 |
| | RFID | 1.89 | 14/36 | bs 16, lr 1·10⁻⁴, wd 0.01, do 0.0 | 2.58 |

## What the search shows

Within this grid and training recipe, Wi-Fi and RFID remain close to chance on
the scene validation protocols. This does not establish that no other recipe
could improve transfer. mmWave improves by 1.6–2.6 points over its reference
cells. The best three joint room/person-shift cells score 10.26, 10.24 and
10.21; the winner's individual seeds score 10.94 and 9.58. Such close rankings
should not be interpreted as statistically established differences.

## Reproducing

Run from `XRF55_HAR` with Python 3.11 and the environment in
`requirements-grid.txt` (PyTorch 2.1.1 CUDA 12.1 was used). Set data and
interpreter paths for your machine:

```bash
cd XRF55_HAR
export XRF55_RAW_ROOT=/path/to/raw/XRF55
export XFI_PYTHON=/path/to/environment/bin/python
mkdir -p logs
sbatch --array=0-647%24 submit_backbone_grid.sh
python summarize_backbone_grid.py backbone_grid
```

The generated `backbone_grid/` directory stays local and is ignored.
`backbone_validation_results.json` preserves the 648 full-run scalar results
and shared source hashes in one committed file; each row identifies its
original local `result.json`. Checkpoints, logs and test outputs also stay local.
To audit the published winners without local run directories:

```bash
python summarize_backbone_grid.py backbone_validation_results.json
``` The summary excludes
short/limited runs and rejects duplicate seeds or mixed source snapshots.
`protocol_snapshot/` contains the exact dataset and split modules used by the
grid; the runner verifies their hashes and the ResNet/training source hashes.
The launcher verifies the runner hash. Do not edit the frozen runner to rerun
the original campaign. Use a fresh output location if results already exist.

## Frozen test evaluation

`XRF55_HAR/selected_backbones.json` records the nine validation argmax cells,
their source records and test seeds **1, 2, 3**, matching our model experiments.
After freezing and publishing this selection:

```bash
cd XRF55_HAR
# Use the same environment variables as above.
sbatch submit_selected_tests.sh
```

This launches 27 trainings at fixed epoch 100, with no early stopping or test
selection. Subject testing retrains on people 01–21 and evaluates 22–30.
Scene protocols retain their grid training membership and evaluate Scenes 3/4.
The grid did not retain checkpoints, so those models must also be retrained.
The test runner reuses the grid's model, optimizer, schedule and training loop;
records source hashes, environment, selections, checkpoint hashes and metrics;
and refuses to overwrite an existing run directory.

## Proposed grid on the new subject/room splits

`XRF55_HAR/search_protocol_backbone.py` uses the frozen
`protocol_snapshot_v2` copy of the shared selectors and all 216 HTML subject
assignments. It also accepts `unseen_room_subject_rotation_1` through `_3`.
Train on the protocol's training membership; record validation every epoch and
rank grid configurations by their epoch-100 validation score.
Test files are not collected or loaded. The historical runner and snapshot remain
available for reproducing the previous campaign.

The proposed grid is batch 16/32/64 × LR 3e-5/1e-4/3e-4/1e-3 × weight decay
0.01/0.3 × dropout 0/0.3: 48 configurations, seeds 1/2, all three modalities
(288 jobs per selected split). Each campaign records every cell and source hashes,
including the subject catalog, and refuses source drift or completed-run overwrites.
Bounded smoke results are marked `smoke_test`, separate from complete grid results.

Prepare a campaign from `XRF55_HAR`, supplying the agreed protocol IDs explicitly:

```bash
../../compass/.venv/bin/python protocol_backbone_grid.py \
  --campaign backbone_grid/new-splits \
  --raw-root "$XRF55_RAW_ROOT" \
  --protocol unseen_subjects_001 \
  --protocol unseen_room_subject_rotation_1 \
  --protocol unseen_room_subject_rotation_2 \
  --protocol unseen_room_subject_rotation_3
```

This example has 1,152 jobs; it does not submit them. The cluster array limit is
1,001, so submit two independent 576-task arrays without a concurrency throttle:

```bash
export XFI_CAMPAIGN="$PWD/backbone_grid/new-splits"
XFI_TASK_OFFSET=0 sbatch --array=0-575 submit_protocol_backbone_grid.sh
XFI_TASK_OFFSET=576 sbatch --array=0-575 submit_protocol_backbone_grid.sh
```

Use the same selected subject IDs and rotations for every method. Split IDs are
independent of training seeds. Matching subject IDs across scenes remains the
split definition's provisional identity assumption. Previous runtime estimates
were for the old three protocols and do not cover this four-split example.

Verified 2026-09-15: four grid/catalog tests passed; actual RFID membership
counts matched subject split 1 and all room rotations. GPU smoke array 266106
passed Wi-Fi, RFID and mmWave forward/backward and validation (one batch each);
no test samples were collected or opened.

Launched 2026-09-15: arrays **266119** (cells 0–575) and **266120**
(cells 576–1,151), with no array throttle or inter-array dependency. Campaign:
`XRF55_HAR/backbone_grid/new-splits-20260915`. Subject split 1 plus all three room
rotations, 48 configurations × 3 modalities × 2 seeds × 4 splits = 1,152 jobs.
Rank on validation; retrain selected configurations to save backbones before
full X-Fi training, using the same split membership.

Near-chance stopping enabled before any grid task started: abort after epoch 30
if best training accuracy is ≤2.2%. Save all epoch training metrics and
`aborted_at_chance` status; skip validation/test, and never replace the seed.
The original campaign manifest is retained as `campaign.before-near-chance.json`.
Boundary, late-recovery and failure-recording tests passed.

Historical stopping-rule audit: **0/675 would stop; 675/675 would continue**
(648 completed grid runs + 27 selected-model retraining runs). The smallest
best logged training accuracy through epoch 30 was **34.81%**, above 2.2%.
Logs contain epochs 1/10/20/30; those observations alone establish preservation
for every completed run. Eighteen partial-training logs and 1,440 logs without
training metrics are excluded, not counted as stopping-rule failures.
This validates preservation on the previous completed runs, not effectiveness
at catching collapses on the new splits. [Per-run evidence](XRF55_HAR/backbone_near_chance_audit.json).

Per-epoch validation enabled before any grid task started. The new
`protocol_backbone_runner.py` saves `history.json` after each evaluated epoch
(training/validation loss, accuracy and sample counts); the result also embeds
that history. Validation preserves training RNG state. Selection remains fixed
at epoch 100; validation-based early stopping is not enabled. The near-chance
training guard still skips evaluation on the aborting epoch. Historical runs
cannot recover validation measurements that were never computed.

Validation checks: CPU end-to-end training test confirms all three epochs
appear in history and the reported selection score matches the final epoch;
near-chance and grid tests pass. GPU smoke array 266217 is queued.

Full X-Fi now supports the same numbered subject splits and room rotations via
`XRF55_HAR/run_protocol.py`, with validation loss/accuracy saved every epoch and
split-matched backbone provenance checks. See the sibling
`wireless-sensing/docs/Filyas-notes/models-for-xrf55/shared-split-training.md`
for commands and required saved backbone files. Existing backbone grid source
hashes remain unchanged.

## Current seed-1 campaign

The previous arrays 266119/266120 were already cancelled before resubmission.
The agreed 48-config grid now uses **seed 1 only**:

- **266779:** subject split 1; 144 tasks; `backbone_grid/seed1-subjects`.
- **266780:** room rotations 1/2/3; 432 tasks; `backbone_grid/seed1-rotations`.

No array throttle or dependency. Train up to 100 epochs, record validation every
epoch, rank by final validation accuracy. Retain the training-only near-chance
stop (best training accuracy ≤2.2% through epoch 30); validation-plateau stopping
is not enabled. The prior two-seed results/partial histories remain separate.
Prepare future single-seed campaigns with `protocol_backbone_grid.py --seeds 1`.

Environment-retrieval repair: GPU probe 267399 passed with plain `--export=ALL`
and campaign variables set in the submitting process environment. Replaced only
held tasks: **267401** (138 subject tasks) and **267402** (432 rotation tasks).
The three completed and three running tasks of 266779 were preserved. No seeds,
configurations, split memberships, or training source hashes changed. Submission
records are saved as `environment-repair-submission.json` in each campaign.
Avoid comma-list `--export=ALL,XFI_CAMPAIGN=...`; use the environment prefix and
plain `--export=ALL` instead.
