# X-Fi XRF55 reproduction failure

## Setup

We trained the released unified X-Fi model on XRF55 Part 1: all four scenes,
trials 1–14 for training (15,400 samples), and trials 15–20 for testing (6,600).
The model started from the released mmWave, Wi-Fi, and RFID backbones; their
files are byte-identical to the copies preserved in PTA.

The run used 100 epochs, AdamW at `1e-4`, batch size 16, seed 3407, modality
probabilities `(0.5, 0.9, 0.6)`, no scheduler, and the fixed final checkpoint.
It did not use the released final unified X-Fi checkpoint.

## Results

| Modalities | Paper Table 2 | Ours | Difference |
|---|---:|---:|---:|
| mmWave | 83.90% | 83.62% | −0.28 pp |
| Wi-Fi | 55.70% | 52.22% | −3.48 pp |
| RFID | 42.50% | 39.69% | −2.81 pp |
| mmWave + Wi-Fi | 88.20% | 87.95% | −0.25 pp |
| mmWave + RFID | 86.50% | 84.78% | −1.72 pp |
| Wi-Fi + RFID | 58.10% | 56.73% | −1.37 pp |
| All three | 89.80% | 88.66% | −1.14 pp |

The job completed without numerical or runtime errors, but did not exactly
reproduce Table 2. The deleted released notebook reports 89.13% for all three
modalities, which also differs from the paper.

## Reproduction limits

- The paper uses four X-Fusion iterations; the code uses five.
- The paper and code use different attention scaling.
- Backbone parameters are excluded from AdamW, but BatchNorm statistics update.
- The paper omits run count, seeds, checkpoint selection, and variation.
- Author guidance mentions tuning modality probabilities and adding a scheduler,
  but gives neither setting.

This is a failed exact reproduction, not evidence that the result is impossible
to reproduce.
