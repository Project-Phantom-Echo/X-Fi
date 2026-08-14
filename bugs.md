# Known Bugs

## XRF55 backbone training uses the test set for checkpoint selection by default

**Location:** `code/XRF55_HAR/pretrain_backbones.py`

The script defaults to `--checkpoint-selection best-test`. It evaluates the
test split during training and retains the epoch with the highest test
accuracy, without using an independent validation set. Repeatedly consulting
test accuracy to choose an epoch turns the test set into a validation set and
makes the final reported test result optimistically biased.

Create a validation subset using only the training split and select checkpoints
with validation performance. The held-out test split should be evaluated only
after checkpoint selection is complete. Using `--checkpoint-selection final`
avoids direct best-test selection, but an independent validation set is still
needed for principled tuning and early stopping.
