#!/usr/bin/env python3
"""Compare two compatible PyTorch model/state-dict checkpoints."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch


def load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location="cpu")

    if isinstance(obj, torch.nn.Module):
        return obj.state_dict()
    if isinstance(obj, dict):
        for key in ("state_dict", "model_state_dict"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
        if all(isinstance(value, torch.Tensor) for value in obj.values()):
            return obj
    raise TypeError(f"Unsupported checkpoint format: {path}")


def compare(reference: dict[str, torch.Tensor], candidate: dict[str, torch.Tensor]) -> dict:
    reference_keys = set(reference)
    candidate_keys = set(candidate)
    common = sorted(reference_keys & candidate_keys)
    mismatched_shapes = [
        key for key in common if reference[key].shape != candidate[key].shape
    ]
    comparable = [
        key
        for key in common
        if reference[key].shape == candidate[key].shape
        and (reference[key].is_floating_point() or reference[key].is_complex())
    ]

    dot = ref_sq = cand_sq = diff_sq = absolute_sum = 0.0
    numel = exact_numel = 0
    for key in comparable:
        ref = reference[key].double().reshape(-1)
        cand = candidate[key].double().reshape(-1)
        diff = cand - ref
        dot += torch.dot(ref, cand).item()
        ref_sq += torch.dot(ref, ref).item()
        cand_sq += torch.dot(cand, cand).item()
        diff_sq += torch.dot(diff, diff).item()
        absolute_sum += diff.abs().sum().item()
        numel += ref.numel()
        exact_numel += torch.eq(ref, cand).sum().item()

    return {
        "common_keys": len(common),
        "reference_only": len(reference_keys - candidate_keys),
        "candidate_only": len(candidate_keys - reference_keys),
        "shape_mismatches": len(mismatched_shapes),
        "float_values": numel,
        "exact_fraction": exact_numel / numel if numel else float("nan"),
        "cosine": dot / math.sqrt(ref_sq * cand_sq) if ref_sq and cand_sq else float("nan"),
        "relative_l2": math.sqrt(diff_sq / ref_sq) if ref_sq else float("nan"),
        "mean_absolute_difference": absolute_sum / numel if numel else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidates", type=Path, nargs="+")
    args = parser.parse_args()

    reference = load_state_dict(args.reference)
    print("candidate\tkeys\tshape_mismatch\texact\tcosine\trel_l2\tmean_abs")
    for path in args.candidates:
        result = compare(reference, load_state_dict(path))
        keys = (
            f"{result['common_keys']}"
            f"(-{result['reference_only']}+{result['candidate_only']})"
        )
        print(
            f"{path}\t{keys}\t{result['shape_mismatches']}\t"
            f"{result['exact_fraction']:.6f}\t{result['cosine']:.6f}\t"
            f"{result['relative_l2']:.6f}\t"
            f"{result['mean_absolute_difference']:.6e}"
        )


if __name__ == "__main__":
    main()
