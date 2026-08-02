"""Evaluate X-Fi-format XRF55 backbones overall and by scene."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from XRF55_Dataset import XRF55_Datase
from pretrain_backbones import MODEL_SPECS, select_input


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--weights-root", type=Path, required=True)
    parser.add_argument(
        "--modalities",
        nargs="+",
        choices=tuple(MODEL_SPECS),
        default=list(MODEL_SPECS),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_model(path: Path, device: torch.device) -> torch.nn.Module:
    try:
        model = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        model = torch.load(path, map_location=device)
    return model.to(device).eval()


@torch.inference_mode()
def accuracy(
    model: torch.nn.Module,
    loader: DataLoader,
    modality: str,
    device: torch.device,
) -> float:
    correct = 0
    count = 0
    for batch in loader:
        inputs, labels = select_input(batch, modality)
        inputs = inputs.to(device)
        labels = labels.to(device)
        correct += (model(inputs).argmax(dim=1) == labels).sum().item()
        count += labels.numel()
    return correct / count


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    results: dict[str, dict[str, float | int]] = {}

    for modality in args.modalities:
        _, relative_path = MODEL_SPECS[modality]
        model_path = args.weights_root / relative_path
        if modality == "rfid" and not model_path.exists():
            released_name = model_path.with_name("RFID_ResNet18.pt")
            if released_name.exists():
                model_path = released_name
        model = load_model(model_path, device)
        modality_results: dict[str, float | int] = {}
        total_correct_equivalent = 0.0
        total_count = 0

        for scene in ("Scene1", "Scene2", "Scene3", "Scene4"):
            dataset = XRF55_Datase(str(args.dataset), scene=[scene], is_train=False)
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.workers,
            )
            scene_accuracy = accuracy(model, loader, modality, device)
            modality_results[scene] = scene_accuracy
            modality_results[f"{scene}_samples"] = len(dataset)
            total_correct_equivalent += scene_accuracy * len(dataset)
            total_count += len(dataset)

        modality_results["all"] = total_correct_equivalent / total_count
        modality_results["all_samples"] = total_count
        results[modality] = modality_results
        print(
            f"{modality}: all={modality_results['all']:.4f}, "
            + ", ".join(
                f"{scene}={modality_results[scene]:.4f}"
                for scene in ("Scene1", "Scene2", "Scene3", "Scene4")
            )
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
