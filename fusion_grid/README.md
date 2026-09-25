# Full X-Fi staged grid

This records the historical fixed-epoch search. The launcher retains its original
epoch-100 selection policy. Current EMA selections and corrected fusion-final
status are recorded in the shared
[hyperparameter report](https://github.com/Project-Phantom-Echo/wireless-sensing/blob/main/docs/Filyas-notes/models-for-xrf55/hparam-tuning.md)
and [final results](https://github.com/Project-Phantom-Echo/wireless-sensing/blob/main/docs/Filyas-notes/models-for-xrf55/final-results.md).

Initial grid: batch 16, LR 3e-5/1e-4/3e-4, WD 0.01, seed 1, 100 epochs,
four shared splits: 12 jobs. Validation every epoch; select epoch 100, no test.
Reuses the split-matched train-only X-Fi backbone checkpoints prepared for COMPASS.

```bash
../compass/.venv/bin/python fusion_grid/launch.py --submit
```

The initial campaign is prepared. Its source/checkpoint hashes and exact tasks
are archived in `campaign/initial-manifest.json`. Duplicate submissions are rejected.
From a fresh checkout with the referenced artifacts, `--submit` also prepares the
campaign. Data, weights, histories and runtime snapshots are ignored by Git.

Optional extension, only after deciding to expand the grid: `--stage extend-wd --submit`.
This adds WD 0.1 in a separate campaign: 12 new jobs and no repeated configurations.
The approximately 213-hour initial estimate is based on historical throughput;
measure actual timing before deciding on expansion. Decide before test evaluation.

The fusion extension adds 15 configurations per split: LR {3e-5, 1e-4,
3e-4} × WD {0, .01, .1} × fusion dropout {0, .3}, excluding the three
completed WD=.01/dropout=0 cells. Submit each split independently with
`launch.py --stage extend-fusion --protocol <protocol> --submit`.
Each split has a separate frozen campaign and submission lock. Reuse seed-1
train-only backbones, train 100 epochs, and select across all 18 configurations
using epoch-100 validation accuracy. Estimated additional search: 175 GPU-hours;
210 total, based on measured initial runtimes. No test evaluation during search.
The final launcher now selects from the combined 72 runs after
this extension finishes.
