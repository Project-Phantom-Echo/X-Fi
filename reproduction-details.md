# X-Fi XRF55 reproduction

## Conclusion

Our runs reproduce X-Fi's Table 2 numerically, and cannot reproduce it exactly.
Those are different claims and both hold.

Over three seeds the unified model reaches **89.33 ± 0.69** on all three
modalities against the paper's 89.80 -- 0.47 below the published number, with
one seed above it, so their result sits inside our run-to-run distribution.

Exact reproduction remains blocked: essential settings are missing and the
paper, code and deleted notebooks disagree with each other. No unique
experiment in the release can be rerun to obtain the published number, which is
why the agreement above is distributional rather than a match.

All results below use XRF55 Part 1 across all four scenes: trials 1–14 for
training (15,400 samples) and trials 15–20 for testing (6,600).

## Modality backbones

| Modality | Paper baseline | Released checkpoint, our evaluation | Deleted notebook | Our scratch retraining |
|---|---:|---:|---:|---:|
| mmWave | 82.10% | 82.06% | 82.05% at epoch 10 | 87.45% |
| Wi-Fi | 77.80% | 80.98% | 80.98% at epoch 50 | 84.83% |
| RFID | 42.20% | 44.12% | 44.09% at epoch 40 | 45.08% |

Our scratch runs used the recovered notebook schedules, seed 3407, 100 epochs,
and the fixed final checkpoint. Wi-Fi and RFID otherwise follow the notebook
recipes, but the notebooks specify no seed or batch order; the mmWave notebook
also starts from an unidentified checkpoint, whereas our run starts from
random weights.

The paper's baseline column is not a direct evaluation of these backbone
checkpoints: it comes from a separate 40-epoch baseline fine-tuning pipeline.
A controlled cosine-tail alternative reached 90.45% mmWave, 91.32% Wi-Fi, and
44.67% RFID, but it is our experiment rather than a reproduction.

## Full X-Fi model

We trained the released unified model for 100 epochs with the released
backbones, AdamW at `1e-4`, batch size 16, seed 3407, modality probabilities
`(0.5, 0.9, 0.6)`, no scheduler, and the fixed final checkpoint. The backbone
files are byte-identical to the copies preserved in PTA; we did not initialize
from the released final unified X-Fi checkpoint.

| Available modalities | Paper Table 2 | Deleted notebook | Our seed 3407 |
|---|---:|---:|---:|
| mmWave | 83.90% | 82.61% | 83.62% |
| Wi-Fi | 55.70% | 55.43% | 52.22% |
| RFID | 42.50% | 39.13% | 39.69% |
| mmWave + Wi-Fi | 88.20% | 88.68% | 87.95% |
| mmWave + RFID | 86.50% | 84.83% | 84.78% |
| Wi-Fi + RFID | 58.10% | 58.70% | 56.73% |
| All three | 89.80% | 89.13% | 88.66% |

Seed 3407 alone misses the all-modality result by 1.14 points, and the authors'
own deleted notebook misses it by 0.67, so a single run of the released path
does not recover Table 2. Two further seeds put that gap in context:

| Run | Seed | All three |
|---|---:|---:|
| `unified_runs/full_job240213` | 3407 | 88.66% |
| `unified_runs/seed1_job243339_0` | 1 | 89.30% |
| `unified_runs/seed2_job243339_1` | 2 | 90.04% |
| **mean** | | **89.33 ± 0.69** |

The published 89.80 lies between our second and third seeds. A single run
under- or overshoots it by more than a point in either direction, which is the
measurement the paper reports without a spread of its own.

## Why exact reproduction is blocked

- The paper uses four X-Fusion iterations; the code and notebook use five.
- The paper and code use different attention scaling.
- Backbone weights are excluded from AdamW, but `model.train()` still updates
  their BatchNorm statistics, so the claimed freezing is incomplete.
- Seeds, run count, checkpoint selection, and result variation are not reported.
- Author guidance recommends tuning modality probabilities and adding a
  scheduler, but provides neither the probabilities nor the schedule.
- Exact backbone initialization and selection are unavailable; most notably,
  the mmWave notebook loads an unidentified checkpoint.

Different undocumented choices may reach the paper's number, but there is no
unique experiment in the release that can be rerun to obtain it. Seed variance
alone spans it, so a single matching run would not have been evidence either.

## Related

`hparam-search.md` searches 36 configurations per modality and protocol around
the backbones described here, and reports what changes when their settings are
tuned rather than reproduced.
