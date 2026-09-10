#!/usr/bin/env python3
"""Tune one released X-Fi backbone on a shared XRF55 hard protocol.

This runner deliberately never opens the held-out test trials. A grid cell is
scored once, at its fixed final epoch, and the companion summary script ranks
complete cells by their mean over seeds 1 and 2.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from pretrain_backbones import HISTORICAL_SETTINGS, MODEL_SPECS


RELEASE_MODALITY = {"rfid": "RFID", "wifi": "WiFi", "mmwave": "mmWave"}
VALIDATION_PROTOCOLS = (
    "unseen_subjects_validation",
    "unseen_scenes",
    "unseen_scene_unseen_subject",
)
# These digests describe the run source snapshot that --protocol-code-dir points
# at, not the live experiments-on-xrf55 tree. The snapshot cannot change, which
# is the point: the grid reads the protocol code exactly as it stood when the
# campaign was pinned, and edits to the working tree neither break the guard nor
# leak into the runs. Do not re-pin these to the live files.
EXPECTED_PROTOCOL_SHA256 = {
    "split.py": "9bc6b50c0fa6a1cb35359c29c5ca0a537cb05b8b6235aded3527c478be4195b8",
    "dataset.py": "3013551c6f583f1200d51ff1a26f4bb89779e8d91ca0e31a2818ae7166e6e0b4",
}
EXPECTED_LOCAL_SHA256 = {
    "pretrain_backbones.py": "30c721009eb796f3dfd0ac7a94b63e872a3fd022366d20cf17582b5194ef4bad",
    "backbone_models/RFID/ResNet.py": "c5cfa4b0cd1ad28811ee618f6899fce8ae1c0af373128f1b2ecddf505dd6ebb1",
    "backbone_models/WIFI/ResNet.py": "f0dc9da6d0499dddd45142bee6a44dcfb0011a9c2a73f07fa25e8fa88b9a6dd1",
    "backbone_models/mmWave/ResNet.py": "43c98e1b53729fdbb985cf08dfb632a43b8647e9b3a203af4f2291edb7a68593",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--protocol-code-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", choices=VALIDATION_PROTOCOLS, required=True)
    parser.add_argument("--modality", choices=tuple(MODEL_SPECS), required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--weight-decay", type=float, required=True)
    parser.add_argument("--dropout", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--validation-batch-size", type=int, default=64)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-validation-batches", type=int)
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def check_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0 or args.validation_batch_size <= 0:
        raise SystemExit("batch sizes must be positive")
    if args.epochs <= 0 or args.workers < 0:
        raise SystemExit("epochs must be positive and workers nonnegative")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise SystemExit("learning rate must be positive and weight decay nonnegative")
    if not 0 <= args.dropout < 1:
        raise SystemExit("dropout must be in [0, 1)")
    if not args.raw_root.is_dir():
        raise SystemExit(f"raw XRF55 root does not exist: {args.raw_root}")
    if not (args.protocol_code_dir / "split.py").is_file():
        raise SystemExit(f"split.py is missing from {args.protocol_code_dir}")
    if not (args.protocol_code_dir / "dataset.py").is_file():
        raise SystemExit(f"dataset.py is missing from {args.protocol_code_dir}")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_code_snapshot(args: argparse.Namespace) -> None:
    local_root = Path(__file__).resolve().parent
    checks = [
        (args.protocol_code_dir / name, digest)
        for name, digest in EXPECTED_PROTOCOL_SHA256.items()
    ] + [
        (local_root / name, digest) for name, digest in EXPECTED_LOCAL_SHA256.items()
    ]
    drift = [
        f"{path}: expected {expected}, got {sha256(path)}"
        for path, expected in checks
        if sha256(path) != expected
    ]
    if drift:
        raise RuntimeError(
            "reproducibility-critical source changed after grid submission:\n"
            + "\n".join(drift)
        )


def selector_record(selector) -> dict:
    return {
        "scenes": list(selector.scenes),
        "subjects": sorted(selector.subjects) if selector.subjects is not None else None,
        "repetitions": (
            sorted(selector.repetitions) if selector.repetitions is not None else None
        ),
        "part1_only": selector.part1_only,
    }


def load_shared_data(args: argparse.Namespace):
    # dataset.py intentionally imports its sibling as `split`, so put this exact
    # directory first and reject a previously imported module from elsewhere.
    protocol_dir = str(args.protocol_code_dir.resolve())
    sys.path.insert(0, protocol_dir)
    for name in ("dataset", "split"):
        old = sys.modules.get(name)
        if old is not None:
            origin = Path(getattr(old, "__file__", "")).resolve()
            if origin.parent != args.protocol_code_dir.resolve():
                del sys.modules[name]
    split_module = importlib.import_module("split")
    dataset_module = importlib.import_module("dataset")

    release_name = RELEASE_MODALITY[args.modality]
    protocol = split_module.build_protocol(args.protocol)
    validation_splits = protocol.validation_splits()
    if len(validation_splits) != 1:
        raise RuntimeError(
            f"{args.protocol} must have exactly one validation split, got "
            f"{validation_splits}"
        )
    validation_split = validation_splits[0]
    train_samples = {
        release_name: split_module.collect(
            args.raw_root, release_name, protocol.train
        )
    }
    validation_selector = protocol.evaluations[validation_split]
    validation_samples = {
        release_name: split_module.collect(
            args.raw_root, release_name, validation_selector
        )
    }
    train_data = dataset_module.XRF55Dataset(train_samples, (release_name,))
    validation_data = dataset_module.XRF55Dataset(
        validation_samples, (release_name,)
    )
    if protocol.expected is None:
        raise RuntimeError(f"{args.protocol} has no pinned expected sample counts")
    expected_train, expected_evaluations = protocol.expected
    expected_validation = expected_evaluations[validation_split]
    if (len(train_data), len(validation_data)) != (
        expected_train,
        expected_validation,
    ):
        raise RuntimeError(
            f"{args.protocol} membership changed: expected train={expected_train} "
            f"and validation={expected_validation}, got {len(train_data)} and "
            f"{len(validation_data)}"
        )
    return train_data, validation_data, protocol, validation_split


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    max_batches: int | None,
) -> tuple[float, float, int]:
    training = optimizer is not None
    model.train(training)
    criterion = nn.CrossEntropyLoss()
    loss_sum = 0.0
    correct = 0
    count = 0

    for batch_index, (inputs, labels) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        if inputs.ndim < 3:
            raise ValueError(f"input has invalid shape {tuple(inputs.shape)}")
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.long().to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(inputs)
            loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss after {count} samples")
        if training:
            loss.backward()
            optimizer.step()

        size = labels.numel()
        loss_sum += loss.item() * size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        count += size

    if count == 0:
        raise RuntimeError("loader produced no samples")
    return loss_sum / count, correct / count, count


def main() -> None:
    args = parse_args()
    check_args(args)
    check_code_snapshot(args)
    seed_everything(args.seed)
    start = time.monotonic()

    train_data, validation_data, protocol, validation_split = load_shared_data(args)
    generator = torch.Generator().manual_seed(args.seed)
    loader_options = {
        "num_workers": args.workers,
        "pin_memory": args.device.startswith("cuda"),
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        **loader_options,
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=args.validation_batch_size,
        shuffle=False,
        **loader_options,
    )

    constructor, _ = MODEL_SPECS[args.modality]
    model = constructor()
    if not isinstance(model.fc, nn.Linear):
        raise TypeError(f"unexpected released classifier: {type(model.fc).__name__}")
    if args.dropout > 0:
        model.fc = nn.Sequential(nn.Dropout(args.dropout), model.fc)
    device = torch.device(args.device)
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    settings = HISTORICAL_SETTINGS[args.modality]
    milestones = settings["milestones"]
    scheduler = (
        torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
        if milestones
        else None
    )

    print(
        f"protocol={args.protocol} validation_split={validation_split} "
        f"train={len(train_data)} validation={len(validation_data)} "
        f"modality={args.modality} batch={args.batch_size} lr={args.learning_rate:g} "
        f"weight_decay={args.weight_decay:g} dropout={args.dropout:g} seed={args.seed} "
        f"epochs={args.epochs} milestones={milestones}",
        flush=True,
    )
    final_train_loss = final_train_accuracy = None
    trained_samples = 0
    for epoch in range(1, args.epochs + 1):
        epoch_lr = optimizer.param_groups[0]["lr"]
        final_train_loss, final_train_accuracy, trained_samples = run_epoch(
            model,
            train_loader,
            device,
            optimizer,
            args.max_train_batches,
        )
        if scheduler is not None:
            scheduler.step()
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"epoch={epoch:03d} lr={epoch_lr:.8g} train_loss={final_train_loss:.6f} "
                f"train_accuracy={final_train_accuracy:.6f}",
                flush=True,
            )

    validation_loss, validation_accuracy, validated_samples = run_epoch(
        model,
        validation_loader,
        device,
        optimizer=None,
        max_batches=args.max_validation_batches,
    )
    elapsed = time.monotonic() - start
    print(
        f"fixed_epoch={args.epochs} validation_loss={validation_loss:.6f} "
        f"validation_accuracy={validation_accuracy:.6f} elapsed_seconds={elapsed:.1f}",
        flush=True,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = None
    if args.save_model:
        model_path = args.output_dir / "model.state_dict.pt"
        torch.save(model.cpu().state_dict(), model_path)

    author_point = {
        "batch_size": settings["batch_size"],
        "learning_rate": settings["learning_rate"],
        "weight_decay": 0.01,
        "dropout": 0.0,
    }
    current_point = {
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
    }
    report = {
        "status": "completed",
        "model": "released X-Fi modality-specific ResNet-18",
        "modality": args.modality,
        "protocol": args.protocol,
        "protocol_description": protocol.description,
        "train_membership": selector_record(protocol.train),
        "validation_split": validation_split,
        "validation_membership": selector_record(
            protocol.evaluations[validation_split]
        ),
        "test_data_opened": False,
        "test_paths_enumerated": False,
        "train_samples": len(train_data),
        "validation_samples": len(validation_data),
        "epochs": args.epochs,
        "fixed_score_epoch": args.epochs,
        "checkpoint_selection": "fixed final epoch",
        "batch_size": args.batch_size,
        "validation_batch_size": args.validation_batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "weight_decay_scope": "all parameters",
        "dropout": args.dropout,
        "dropout_location": "classifier input only",
        "optimizer": "AdamW",
        "scheduler": "historical released notebook recipe",
        "lr_milestones": milestones,
        "lr_gamma": 0.1 if milestones else None,
        "seed": args.seed,
        "cudnn_deterministic": True,
        "mmwave_initialization": "random scratch" if args.modality == "mmwave" else None,
        "author_hyperparameter_point": author_point,
        "matches_author_hyperparameter_point": current_point == author_point,
        "final_train_loss": final_train_loss,
        "final_train_accuracy": final_train_accuracy,
        "validation_loss": validation_loss,
        "validation_accuracy": validation_accuracy,
        "trained_samples_in_final_epoch": trained_samples,
        "validated_samples": validated_samples,
        "elapsed_seconds": elapsed,
        "workers": args.workers,
        "max_train_batches": args.max_train_batches,
        "max_validation_batches": args.max_validation_batches,
        "raw_root": str(args.raw_root.resolve()),
        "protocol_code_dir": str(args.protocol_code_dir.resolve()),
        "split_py_sha256": sha256(args.protocol_code_dir / "split.py"),
        "dataset_py_sha256": sha256(args.protocol_code_dir / "dataset.py"),
        "critical_local_source_sha256": EXPECTED_LOCAL_SHA256,
        "runner_sha256": sha256(Path(__file__)),
        "model_path": str(model_path) if model_path else None,
    }
    (args.output_dir / "result.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
