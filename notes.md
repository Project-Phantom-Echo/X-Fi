# X-Fi

## Paper

- **Title:** X-Fi: A Modality-Invariant Foundation Model for Multimodal Human Sensing
- **Authors:** Xinyan Chen, Jianfei Yang
- **Venue:** ICLR 2025
- **Links:** [Paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/f25602918e8a0d0c86e3c752ecfbbaa1-Paper-Conference.pdf) · [Project](https://xyanchen.github.io/X-Fi/) · [Code](https://github.com/NTUMARS/X-Fi)

## Method

- X-Fi is a transformer-based fusion model, not an end-to-end transformer:
  modality-specific backbones first extract features, and transformers then
  combine those features.
- Pretrain one task model per modality, remove its output head, and freeze the
  extractor during X-Fi training.
- Convert each available modality into 32 feature vectors (“tokens”), each
  containing 512 numbers.
- X-Fusion concatenates the active modalities, giving `32 × number of
  modalities` tokens. Self-attention mixes information across them, and pooling
  reduces them to 32 shared tokens.
- A separate small network also converts each modality's original 32 tokens
  into keys and values. The 32 shared tokens act as queries in cross-attention,
  retrieving relevant information separately from every active modality.
- This produces 32 updated tokens per modality. X-Fusion concatenates them,
  mixes and reduces them to 32 shared tokens again, and repeats the process.
  Finally, it averages the 32 tokens and predicts the action class.
- One model is trained on random non-empty modality subsets. For XRF55,
  `(P_R, P_W, P_RF) = (0.5, 0.9, 0.6)`.

## XRF55 protocol

For 55-class HAR, X-Fi uses ResNet-18 backbones for mmWave (`R`), Wi-Fi (`W`),
and RFID (`RF`) on all four scenes in XRF55 Part 1, with repetitions 1–14 for
training (`15,400` samples) and 15–20 for testing (`6,600` samples). This Part 1
split is supported by the current code, deleted notebooks, and released weights,
although the paper only says “original split” and does not report Table 2's run
count, random seeds, variation, validation split, or checkpoint-selection rule.

## Paper Table 2

| Modalities | Baseline | X-Fi |
|---|---:|---:|
| R | 82.1 | 83.9 |
| W | 77.8 | 55.7 |
| RF | 42.2 | 42.5 |
| R + W | 86.8 | 88.2 |
| R + RF | 71.4 | 86.5 |
| W + RF | 55.6 | 58.1 |
| R + W + RF | 70.6 | 89.8 |

The paper defines decision- and feature-level baselines but collapses them into
one “Baseline” column in Table 2. For a single modality, the released baseline
code removes the pretrained ResNet classifier, adds a new `LayerNorm + Linear`
head, and fine-tunes the resulting model for 40 epochs. These values are
therefore not direct evaluations of the released backbone checkpoints.
The released baseline prediction arrays recover Wi-Fi `77.82` and RFID `42.17`
(Table 2 after rounding); their mmWave result is `84.91`, not Table 2's `82.1`.

## Released-code differences from the paper

- Paper: four X-Fusion iterations. Code and deleted `main.ipynb`: five.
- Paper: attention scale `0.125`. Cross-modal code instead divides
  post-softmax attention by `sqrt(512)`.
- Backbones are excluded from AdamW, but `model.train()` re-enables their
  BatchNorm state updates; “frozen” is therefore incomplete.
- Current X-Fi training uses 100 epochs, AdamW `1e-4`, batch 16, seed 3407;
  it evaluates every five epochs and saves the final state.
- In official issue #2, a user reported 70.55% when training XRF55 from
  scratch although the released checkpoint matched Table 2. The author advised
  tuning the probabilities used to include each modality during training and
  adding a learning-rate scheduler absent from the release, but did not specify
  a different probability setting or scheduler.

## Deleted notebook evidence

Recovered from commit `68b5fcf`. All use Part 1 and test every 10 epochs.
The displayed outputs stop before the requested 100 epochs.

| Modality | Initialization | Train/test batch | LR / scheduler | Last visible | Best visible test |
|---|---|---:|---|---:|---:|
| Wi-Fi | scratch | 32/32 | `1e-3`; none | 50 | 80.9848% (epoch 50) |
| RFID | scratch | 16/32 | `1e-3`; milestones 20/40, gamma `0.1` | 50 | 44.0909% (epoch 40) |
| mmWave | unidentified checkpoint | 16/32 | `1e-4`; none | 10 | 82.0455% (epoch 10) |

Learning-rate schedule:

- **Wi-Fi:** constant `1e-3` for all 100 epochs; no scheduler.
- **RFID:** `MultiStepLR`, stepped once after every epoch. The learning rate is
  `1e-3` for epochs 1–20, `1e-4` for epochs 21–40, and `1e-5` for epochs
  41–100. Each milestone multiplies it by `0.1`; there is no warmup.
- **mmWave:** constant `1e-4` for all 100 epochs; no scheduler.

The notebooks use the test set to select the saved “best,” set no seed, and
do not record the run count or hardware. The mmWave notebook is not
self-contained and its saved source contains a malformed save statement.

Deleted unified-model output:

| Modalities | Deleted `main.ipynb` | Paper Table 2 |
|---|---:|---:|
| R | 82.61 | 83.9 |
| W | 55.43 | 55.7 |
| RF | 39.13 | 42.5 |
| R + W | 88.68 | 88.2 |
| R + RF | 84.83 | 86.5 |
| W + RF | 58.70 | 58.1 |
| R + W + RF | 89.13 | 89.8 |

This is a full X-Fi run, not a backbone result. It is close but not identical
to Paper Table 2. The available materials do not say whether Table 2 reports
one run or an average of multiple runs.

## Backbone reproduction

- Protocol: Part 1, all scenes, seed 3407, 100 epochs, H100.
- Wi-Fi/RFID use the notebook's model code, sample counts, batch sizes,
  optimizer, learning rates, RFID scheduler, loss, and 10-epoch test interval.
  This is not the identical run: the notebooks set no seed, and their saved
  execution order iterates the shuffled data loader before creating the model.
  Their initial weights and batch order therefore cannot be reconstructed. Our
  run uses seed 3407 and a separate seeded shuffle. The notebooks also compute
  loss on CPU, while ours computes it on GPU; their hardware is not recorded.
- The mmWave comparison differs further: its notebook starts from an unknown
  checkpoint, while our run starts from random weights.
- We report the fixed epoch-100 checkpoints and do not select an earlier epoch
  using test accuracy.
- Official Google Drive weights were downloaded, and their file checksums
  exactly match the PTA/COMPASS copies used by the evaluator.

### Final scratch results

The official and reproduced backbones were evaluated on the same 6,600-sample
test set. Both scratch runs use fixed epoch-100 checkpoints. The released
checkpoints' training epochs are unknown, so comparisons with them are not
controlled training comparisons.

Direct evaluation of the intact released checkpoints gives `80.98` Wi-Fi,
`44.12` RFID, and `82.06` mmWave. Those numbers match the deleted backbone
notebook outputs (`80.9848`, `44.0909`, and `82.0455`) within `0.03` percentage
points. They exceed Table 2's Wi-Fi/RFID baseline because Table 2 uses the
separately retrained baseline pipeline described above, not because our
evaluator improved the released models.

| Modality | Official release | Recovered-schedule scratch | Cosine-tail scratch | Cosine change (pp) |
|---|---:|---:|---:|---:|
| Wi-Fi | 80.98 | 84.83 | **91.32** | **+6.49** |
| RFID | 44.12 | **45.08** | 44.67 | **−0.41** |
| mmWave | 82.06 | 87.45 | **90.45** | **+3.00** |

### Same-epoch notebook comparison

Wi-Fi and RFID use the same recovered training recipe, but not the same random
initialization or batch order. The mmWave row is not comparable: its notebook
loads an unidentified checkpoint before training, while ours starts from
random weights.

| Modality | Epoch | Deleted notebook | Ours | Difference (pp) | Interpretation |
|---|---:|---:|---:|---:|---|
| Wi-Fi | 50 | 80.98 | 82.39 | +1.41 | Same recipe; different random run |
| RFID | 40 | 44.09 | 44.05 | −0.04 | Same recipe; effectively equal |
| mmWave | 10 | 82.05 | 79.70 | — | **Not comparable: checkpoint start vs scratch** |

Outputs: `code/XRF55_HAR/reproduced_backbones/`.

### Controlled cosine-tail run

This is a new controlled experiment, not a reconstruction of an undocumented
official schedule:

- One scratch run per backbone on the same Part 1 split, using AdamW and the
  original batch sizes and starting learning rates.
- Seed `3407` fixes Python, NumPy, PyTorch, CUDA, and DataLoader shuffling. This
  seed comes from the released unified X-Fi code; the backbone notebooks set
  no seed.
- No warmup. The learning rate stays constant through epoch 70, then follows a
  cosine decay over epochs 71–100 to 1% of its starting value: `1e-5` for
  Wi-Fi/RFID and `1e-6` for mmWave.
- The cosine schedule replaces RFID's historical `MultiStepLR`. Results are
  reported at the fixed epoch-100 checkpoint, without test-set selection.
- Job `205181` trained the three backbones sequentially on one H100 in
  `2:51:15`.
- This is not a pure scheduler ablation: this run also enforces deterministic
  cuDNN, which the earlier seeded run did not enforce.

Outputs: `code/XRF55_HAR/cosine_backbones/seed3407_job205181/`.

## Remaining reproduction problems

1. Exact backbone recipes and released-checkpoint epochs are undocumented.
2. The mmWave notebook's starting checkpoint is unknown; a scratch run is a
   controlled alternative, not an exact reconstruction.
3. Paper/code conflict on X-Fusion iterations and attention scaling.
4. Frozen-backbone BatchNorm behavior is unspecified.
5. Available materials do not state whether Table 2 uses one or multiple runs,
   or which baseline its single baseline column represents.
6. The deleted unified checkpoint/run does not exactly reproduce Table 2.
