# Seed-1 backbone grid replay

`seed1-four-splits/` contains the exact submitted campaign manifests, initial and
replacement Slurm submissions, and versions from the interpreter used by the jobs
(`compass/.venv/bin/python` in the original workspace). Package versions are an
inventory, not a minimal installer. Use that Python/PyTorch/CUDA stack on H100 for
the closest numerical reproduction; identical seeds do not guarantee bitwise
identity across hardware/library versions.

The frozen `protocol_snapshot_v2` catalog is the data selection authority. The
companion wireless-sensing commit archives the XRF55 file hashes and exact split
membership in `experiments-on-xrf55/reproduction/seed1-four-splits/dataset.json.gz`.
Raw data and weights are not committed. Use the same XRF55 release.

From `XRF55_HAR`, with the recorded environment activated, create new campaigns:

```bash
python protocol_backbone_grid.py --campaign /path/to/new-subjects \
  --raw-root /path/to/XRF55 --seeds 1 --protocol unseen_subjects_001
python protocol_backbone_grid.py --campaign /path/to/new-rotations \
  --raw-root /path/to/XRF55 --seeds 1 \
  --protocol unseen_room_subject_rotation_1 \
  --protocol unseen_room_subject_rotation_2 \
  --protocol unseen_room_subject_rotation_3
python protocol_backbone_grid.py --campaign /path/to/new-subjects --task 0
```

Execute tasks 0–143 for subjects and 0–431 for rotations. `--seeds 1` is essential:
the original CLI default is seeds 1/2. New campaign creation records/checks source
hashes. Original campaign files preserve their cluster paths as historical evidence.

Slurm: use `submit_protocol_backbone_grid.sh` with `XFI_CAMPAIGN`, `XFI_RUN_DIR`
and `XFI_PYTHON` in the submitting environment, plain `sbatch --export=ALL`, and
`--array=0-143` or `--array=0-431`. No concurrency throttle. Avoid comma-separated
export overrides: these caused user-environment retrieval failures on this cluster.
100 epochs, seed 1, validation each epoch, final-epoch validation selection; stop
when best training accuracy remains ≤2.2% through epoch 30. No test evaluation.
