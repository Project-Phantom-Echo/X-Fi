#!/usr/bin/env python3
"""PyTorch datasets over the XRF55 split protocols.

`split` decides *which* files belong to a split; this module reads them.
One class serves both training styles: a modality-specific backbone asks for one
modality and receives a bare tensor, while a fusion model asks for several and
receives a dict keyed by modality. The per-modality sample lists of a split are
guaranteed by `build_split` to cover identical sample identities in identical
order, so one index addresses the same recording in every modality.

Preprocessing deliberately does not live here. `log1p`, standardization and the
rest are `nn.Module`s inside the model, which is what lets preprocessing be a
sweep axis rather than a dataset rebuild. This module reads, validates and
converts to float32. A malformed array therefore fails with its path before it
can be silently reshaped or poison an optimizer step.

Released arrays are float64 for Wi-Fi and RFID and float32 for mmWave. Tensors
are float32 in every case, matching the original loader. Note that this does not
reduce disk traffic: the float64 bytes are still read before the cast. Halving
Wi-Fi and RFID traffic would require re-encoding the `.npy` files on disk.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

import split as xs


# Model-facing names for the release's modality directories.
MODALITY_KEYS: dict[str, str] = {"RFID": "rfid", "WiFi": "wifi", "mmWave": "mmwave"}
DIRECTORY_FOR_KEY = {key: directory for directory, key in MODALITY_KEYS.items()}

# Released shapes. mmWave stores a redundant leading axis which is removed only
# after the exact on-disk shape has been checked.
MMWAVE_SHAPE = (17, 256, 128)
MODALITY_SHAPES = {
    "RFID": (23, 148),
    "WiFi": (270, 1000),
}


def expected_shape(modality: str) -> tuple[int, ...]:
    if modality == "mmWave":
        return (1, *MMWAVE_SHAPE)
    if modality not in MODALITY_SHAPES:
        raise ValueError(f"unknown modality: {modality}")
    return MODALITY_SHAPES[modality]


def load_array(path, modality: str) -> torch.Tensor:
    """Read and validate one released array as a contiguous float32 tensor."""
    data = np.load(path)
    shape = expected_shape(modality)
    if data.shape != shape:
        raise ValueError(f"{path}: expected {modality} shape {shape}, got {data.shape}")
    if not np.issubdtype(data.dtype, np.floating):
        raise TypeError(
            f"{path}: expected real floating-point {modality} data, got {data.dtype}"
        )
    minimum = data.min()
    maximum = data.max()
    if not np.isfinite(minimum) or not np.isfinite(maximum):
        raise ValueError(f"{path}: {modality} data contains non-finite values")
    if minimum < 0:
        raise ValueError(f"{path}: {modality} data contains negative values")
    if modality == "RFID" and maximum >= 2 * np.pi:
        raise ValueError(f"{path}: RFID phase must be in [0, 2*pi)")
    if modality == "mmWave":
        data = data[0]
    return torch.from_numpy(np.ascontiguousarray(data, dtype=np.float32))


class XRF55Dataset(Dataset):
    """One split of one protocol, over one or more modalities."""

    def __init__(
        self,
        per_modality: dict[str, list[xs.Sample]],
        modalities: tuple[str, ...] | None = None,
    ) -> None:
        super().__init__()
        modalities = tuple(modalities or per_modality)
        if not modalities:
            raise ValueError("at least one modality is required")
        missing = [name for name in modalities if name not in per_modality]
        if missing:
            raise ValueError(f"split does not contain {missing}")

        reference = per_modality[modalities[0]]
        for modality in modalities[1:]:
            samples = per_modality[modality]
            if [s.identity for s in samples] != [s.identity for s in reference]:
                raise ValueError(
                    f"{modality} is not aligned with {modalities[0]}; a shared index "
                    "would address different recordings in each modality"
                )

        self.modalities = modalities
        self.samples = {modality: per_modality[modality] for modality in modalities}
        self.reference = reference

    def __len__(self) -> int:
        return len(self.reference)

    def __getitem__(self, index: int):
        label = self.reference[index].label
        if len(self.modalities) == 1:
            modality = self.modalities[0]
            return load_array(self.samples[modality][index].path, modality), label
        tensors = {
            MODALITY_KEYS[modality]: load_array(
                self.samples[modality][index].path, modality
            )
            for modality in self.modalities
        }
        return tensors, label

    def identity(self, index: int) -> tuple[str, int, int, int]:
        """The `(scene, subject, action, repetition)` addressed by an index."""
        return self.reference[index].identity


def build_datasets(
    raw_root,
    protocol: str,
    modalities: tuple[str, ...] = xs.MODALITIES,
) -> dict[str, XRF55Dataset]:
    """Resolve a protocol into `{split name: dataset}`.

    Split names are `split`'s logical names, so the training split is
    `"train"` and the rest are the protocol's evaluation splits.
    """
    specification = xs.build_protocol(protocol)
    split_samples = xs.build_split(raw_root, specification, modalities)
    return {
        split: XRF55Dataset(per_modality, modalities)
        for split, per_modality in split_samples.items()
    }


def resolve_modalities(names) -> tuple[str, ...]:
    """Accept either model-facing names or release directory names."""
    resolved = []
    for name in names:
        if name in xs.MODALITIES:
            resolved.append(name)
        elif name in DIRECTORY_FOR_KEY:
            resolved.append(DIRECTORY_FOR_KEY[name])
        else:
            raise ValueError(f"unknown modality: {name}")
    return tuple(modality for modality in xs.MODALITIES if modality in set(resolved))
