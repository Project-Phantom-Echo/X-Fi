# Historical fixed-epoch X-Fi finals

This launcher selects one configuration per split from the 12 initial and 60
extension fusion-search runs, using epoch-100 validation accuracy. It retrains
the exact train+validation union for 100 epochs with seeds 1/2/3, retains the
split-matched train-only backbones, and evaluates test data after training.

The newer EMA-selected finals use separate frozen campaigns in the sibling
`wireless-sensing` workspace. Consult its
[hyperparameter report](https://github.com/Project-Phantom-Echo/wireless-sensing/blob/main/docs/Filyas-notes/models-for-xrf55/hparam-tuning.md)
and [final results](https://github.com/Project-Phantom-Echo/wireless-sensing/blob/main/docs/Filyas-notes/models-for-xrf55/final-results.md)
for current results. This historical launcher does not implement that newer
selection policy.

Preparation requires the completed local campaigns under
`XRF55_HAR/fusion_grid/initial-seed1-four-splits` and
`XRF55_HAR/fusion_grid/extend-fusion-seed1-<protocol>` for all four splits.
Their frozen sources, inventories, histories and referenced backbone artifacts
must be present and match the recorded hashes. Keep `x-fi`, `compass` and
`wireless-sensing` as sibling repositories and use the recorded COMPASS Python
environment. A Git clone alone does not include these experiment artifacts.

`launch.py --prepare --campaign <new-directory>` prepares a separate campaign;
`--task 0 --inspect --campaign <directory>` prints its command without training.
`--submit` submits jobs and rejects an existing submission lock. The default
campaign is `XRF55_HAR/fusion_grid/final-expanded-four-splits-seeds123`.
Preparation applies `train-validation.patch` only inside the new source copy.

`test_final.py` runs a tiny CPU test with mocked data and models against that
default frozen campaign. It therefore requires its source snapshot. The grid's
`test_grid.py` likewise requires the initial campaign and its referenced
backbones for its integrity check. Run these only in a review copy with those
artifacts available; no experiment training or scheduler submission is required.
