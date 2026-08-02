"""Recreate the XRF55 backbone recipes recovered from X-Fi's git history.

The final repository deleted the original preparation notebooks. Commit
68b5fcf contains them under XRF55/prepare. Their modality-specific settings
are encoded below. The mmWave notebook is not self-contained: it loads an
unidentified existing checkpoint before training. Use --mmwave-init only when
that initialization is known, or explicitly choose --mmwave-from-scratch for
a controlled non-historical baseline.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from XRF55_Dataset import XRF55_Datase
from backbone_models.RFID.ResNet import resnet18 as rfid_resnet18
from backbone_models.WIFI.ResNet import resnet18 as wifi_resnet18
from backbone_models.mmWave.ResNet import resnet18 as mmwave_resnet18


MODEL_SPECS = {
    "mmwave": (mmwave_resnet18, Path("mmWave/mmwave_ResNet18.pt")),
    "wifi": (wifi_resnet18, Path("WIFI/wifi_ResNet18.pt")),
    "rfid": (rfid_resnet18, Path("RFID/rfid_ResNet18.pt")),
}

HISTORICAL_SETTINGS = {
    "wifi": {
        "epochs": 100,
        "batch_size": 32,
        "learning_rate": 1e-3,
        "milestones": [],
    },
    "rfid": {
        "epochs": 100,
        "batch_size": 16,
        "learning_rate": 1e-3,
        "milestones": [20, 40],
    },
    "mmwave": {
        "epochs": 100,
        "batch_size": 16,
        "learning_rate": 1e-4,
        "milestones": [],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pretrain X-Fi's modality-specific XRF55 backbones."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./reproduced_backbones"),
        help="Root for modality subfolders; use ./backbone_models to run X-Fi directly.",
    )
    parser.add_argument(
        "--modalities",
        nargs="+",
        choices=tuple(MODEL_SPECS),
        default=list(MODEL_SPECS),
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        choices=("all", "Scene1", "Scene2", "Scene3", "Scene4"),
        default=["all"],
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Override the historical 100 epochs; useful only for smoke tests.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Override the historical modality-specific batch size.",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--max-train-batches",
        type=int,
        help="Stop each training epoch after this many batches (smoke tests only).",
    )
    parser.add_argument(
        "--max-test-batches",
        type=int,
        help="Stop each evaluation after this many batches (smoke tests only).",
    )
    parser.add_argument(
        "--save-eval-checkpoints",
        action="store_true",
        help="Save a state_dict at every evaluation epoch.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Override the historical modality-specific learning rate.",
    )
    parser.add_argument(
        "--scheduler",
        choices=("historical", "cosine-tail"),
        default="historical",
        help=(
            "historical keeps each notebook's schedule; cosine-tail holds the "
            "initial LR constant and cosine-decays it over the final fraction."
        ),
    )
    parser.add_argument(
        "--cosine-tail-fraction",
        type=float,
        default=0.3,
        help="Fraction of training used by cosine-tail (default: final 30%%).",
    )
    parser.add_argument(
        "--cosine-min-lr-ratio",
        type=float,
        default=0.01,
        help="Final LR as a fraction of the initial LR (default: 0.01).",
    )
    parser.add_argument(
        "--checkpoint-selection",
        choices=("best-test", "final"),
        default="best-test",
        help="Save either the highest test checkpoint (historical) or fixed final epoch.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="The preparation notebooks do not report or set a seed.",
    )
    parser.add_argument(
        "--mmwave-init",
        type=Path,
        help="Existing full-model checkpoint loaded by the historical mmWave notebook.",
    )
    parser.add_argument(
        "--mmwave-from-scratch",
        action="store_true",
        help=(
            "Initialize mmWave randomly. This is not the deleted notebook's exact "
            "protocol because its starting checkpoint is unidentified."
        ),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def select_input(batch: list[torch.Tensor], modality: str) -> tuple[torch.Tensor, torch.Tensor]:
    wifi, rfid, mmwave, labels = batch
    inputs = {"wifi": wifi, "rfid": rfid, "mmwave": mmwave}[modality]
    expected_channels = {"wifi": 270, "rfid": 23, "mmwave": 17}[modality]
    if inputs.ndim < 3 or inputs.shape[1] != expected_channels:
        raise ValueError(
            f"{modality} input has shape {tuple(inputs.shape)}; expected channel-first "
            f"samples with {expected_channels} channels"
        )
    return inputs.float(), labels.long()


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    modality: str,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    max_batches: int | None = None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    criterion = nn.CrossEntropyLoss()
    loss_sum = 0.0
    correct = 0
    count = 0

    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        inputs, labels = select_input(batch, modality)
        inputs = inputs.to(device)
        labels = labels.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(inputs)
            loss = criterion(logits, labels)
        if training:
            loss.backward()
            optimizer.step()

        batch_size = labels.numel()
        loss_sum += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        count += batch_size

    return loss_sum / count, correct / count


def main() -> None:
    args = parse_args()
    if not 0.0 < args.cosine_tail_fraction <= 1.0:
        raise SystemExit("--cosine-tail-fraction must be in (0, 1]")
    if not 0.0 <= args.cosine_min_lr_ratio < 1.0:
        raise SystemExit("--cosine-min-lr-ratio must be in [0, 1)")
    if args.mmwave_init is not None and args.mmwave_from_scratch:
        raise SystemExit(
            "--mmwave-init and --mmwave-from-scratch are mutually exclusive"
        )
    if (
        "mmwave" in args.modalities
        and args.mmwave_init is None
        and not args.mmwave_from_scratch
    ):
        raise SystemExit(
            "The historical mmWave notebook starts from an existing checkpoint. "
            "Pass --mmwave-init, explicitly pass --mmwave-from-scratch, or omit mmwave."
        )
    if args.seed is not None:
        set_seed(args.seed)
    device = torch.device(args.device)
    scene = "all" if args.scenes == ["all"] else args.scenes

    train_data = XRF55_Datase(str(args.dataset), scene=scene, is_train=True)
    test_data = XRF55_Datase(str(args.dataset), scene=scene, is_train=False)
    print(f"protocol: scenes={scene}, train={len(train_data)}, test={len(test_data)}")
    if scene == "all" and (len(train_data), len(test_data)) != (15400, 6600):
        print(
            "warning: the committed preparation notebooks record train=15400 and "
            "test=6600 (XRF55 part 1); your dataset does not match that checkpoint recipe"
        )

    for modality in args.modalities:
        if args.seed is not None:
            set_seed(args.seed)
        settings = HISTORICAL_SETTINGS[modality]
        epochs = args.epochs or settings["epochs"]
        batch_size = args.batch_size or settings["batch_size"]
        learning_rate = args.learning_rate or settings["learning_rate"]
        constructor, relative_path = MODEL_SPECS[modality]
        if modality == "mmwave" and args.mmwave_init is not None:
            try:
                model = torch.load(
                    args.mmwave_init, map_location=device, weights_only=False
                )
            except TypeError:
                model = torch.load(args.mmwave_init, map_location=device)
        else:
            model = constructor()
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        scheduler = None
        cosine_epochs = None
        constant_epochs = None
        minimum_learning_rate = None
        if args.scheduler == "cosine-tail":
            cosine_epochs = max(1, round(epochs * args.cosine_tail_fraction))
            constant_epochs = epochs - cosine_epochs
            minimum_learning_rate = learning_rate * args.cosine_min_lr_ratio

            def cosine_tail_multiplier(epoch_index: int) -> float:
                if epoch_index < constant_epochs:
                    return 1.0
                if cosine_epochs == 1:
                    return args.cosine_min_lr_ratio
                progress = min(
                    1.0,
                    (epoch_index - constant_epochs) / (cosine_epochs - 1),
                )
                cosine_weight = 0.5 * (1.0 + math.cos(math.pi * progress))
                return args.cosine_min_lr_ratio + (
                    1.0 - args.cosine_min_lr_ratio
                ) * cosine_weight

            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=cosine_tail_multiplier
            )
            print(
                f"{modality} scheduler=cosine-tail base_lr={learning_rate:g} "
                f"constant_epochs=1-{constant_epochs} "
                f"cosine_epochs={constant_epochs + 1}-{epochs} "
                f"minimum_lr={minimum_learning_rate:g} warmup=none"
            )
        elif settings["milestones"]:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=settings["milestones"], gamma=0.1
            )
        generator = None
        if args.seed is not None:
            generator = torch.Generator().manual_seed(args.seed)
        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=args.workers,
            generator=generator,
        )
        test_loader = DataLoader(
            test_data,
            batch_size=32,
            shuffle=False,
            num_workers=args.workers,
        )

        best_accuracy = -1.0
        best_state = None
        final_test_accuracy = None
        for epoch in range(1, epochs + 1):
            epoch_learning_rate = optimizer.param_groups[0]["lr"]
            train_loss, train_accuracy = run_epoch(
                model,
                train_loader,
                modality,
                device,
                optimizer,
                max_batches=args.max_train_batches,
            )
            if scheduler is not None:
                scheduler.step()
            if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
                print(
                    f"{modality} epoch={epoch:03d} "
                    f"lr={epoch_learning_rate:.8g} "
                    f"loss={train_loss:.5f} train_accuracy={train_accuracy:.4f}"
                )
            if epoch % 10 == 0 or epoch == epochs:
                test_loss, test_accuracy = run_epoch(
                    model,
                    test_loader,
                    modality,
                    device,
                    optimizer=None,
                    max_batches=args.max_test_batches,
                )
                print(
                    f"{modality} epoch={epoch:03d} "
                    f"test_loss={test_loss:.5f} test_accuracy={test_accuracy:.4f}"
                )
                final_test_accuracy = test_accuracy
                if args.save_eval_checkpoints:
                    checkpoint_path = (
                        args.output_dir
                        / relative_path.parent
                        / f"{relative_path.stem}.epoch{epoch:03d}.state_dict.pt"
                    )
                    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(), checkpoint_path)
                if (
                    args.checkpoint_selection == "best-test"
                    and test_accuracy >= best_accuracy
                ):
                    best_accuracy = test_accuracy
                    best_state = copy.deepcopy(model.state_dict())

        destination = args.output_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        model = model.cpu()
        final_state = copy.deepcopy(model.state_dict())
        torch.save(model, destination.with_name(destination.stem + ".final.pt"))
        if final_test_accuracy is None:
            raise RuntimeError("No evaluation checkpoint was produced")
        if args.checkpoint_selection == "best-test":
            if best_state is None:
                raise RuntimeError("No best-test checkpoint was produced")
            selected_state = best_state
            selected_accuracy = best_accuracy
        else:
            selected_state = final_state
            selected_accuracy = final_test_accuracy
        model.load_state_dict(selected_state)
        torch.save(model, destination)
        torch.save(selected_state, destination.with_suffix(".state_dict.pt"))

        metadata = {
            "status": (
                "controlled cosine-tail experiment based on the recovered recipe"
                if args.scheduler == "cosine-tail"
                else "recovered from deleted preparation notebooks at commit 68b5fcf"
            ),
            "modality": modality,
            "scenes": scene,
            "train_samples": len(train_data),
            "test_samples": len(test_data),
            "epochs": epochs,
            "batch_size": batch_size,
            "test_batch_size": 32,
            "optimizer": "AdamW",
            "learning_rate": learning_rate,
            "scheduler": args.scheduler,
            "lr_milestones": (
                settings["milestones"] if args.scheduler == "historical" else []
            ),
            "lr_gamma": (
                0.1
                if args.scheduler == "historical" and settings["milestones"]
                else None
            ),
            "cosine_tail_fraction": (
                args.cosine_tail_fraction if args.scheduler == "cosine-tail" else None
            ),
            "cosine_tail_epochs": cosine_epochs,
            "constant_lr_epochs": constant_epochs,
            "minimum_learning_rate": minimum_learning_rate,
            "warmup_epochs": 0,
            "seed": args.seed,
            "cudnn_deterministic": args.seed is not None,
            "checkpoint_selection": args.checkpoint_selection,
            "max_train_batches": args.max_train_batches,
            "max_test_batches": args.max_test_batches,
            "saved_every_evaluation": args.save_eval_checkpoints,
            "selected_test_accuracy": selected_accuracy,
            "final_test_accuracy": final_test_accuracy,
            "best_test_accuracy": (
                best_accuracy if args.checkpoint_selection == "best-test" else None
            ),
            "mmwave_initial_checkpoint": (
                str(args.mmwave_init) if modality == "mmwave" else None
            ),
            "mmwave_from_scratch": (
                args.mmwave_from_scratch if modality == "mmwave" else None
            ),
            "model_path": str(destination),
        }
        destination.with_suffix(".json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )
        print(
            f"saved {destination}; selection={args.checkpoint_selection} "
            f"selected_test_accuracy={selected_accuracy:.4f}"
        )


if __name__ == "__main__":
    main()
