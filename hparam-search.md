# X-Fi backbone hyperparameter search

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
