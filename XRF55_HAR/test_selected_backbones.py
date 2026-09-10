#!/usr/bin/env python3
"""Train frozen validation winners, then evaluate held-out splits once."""
from __future__ import annotations

import argparse
import importlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

import search_backbone as search

ROOT = Path(__file__).resolve().parent


def selected_task(index):
    choices = json.loads((ROOT / "selected_backbones.json").read_text())
    if not 0 <= index < len(choices) * 3:
        raise ValueError("task must be in 0..26")
    choice = choices[index // 3]
    return choice, choice["test_seeds"][index % 3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=ROOT / "backbone_tests")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    choice, seed = selected_task(args.task)
    protocol_name = ("unseen_subjects" if choice["protocol"] == "unseen_subjects_validation"
                     else choice["protocol"])
    protocol_dir = ROOT / "protocol_snapshot"
    search.check_code_snapshot(argparse.Namespace(protocol_code_dir=protocol_dir))
    sys.path.insert(0, str(protocol_dir))
    split = importlib.import_module("split")
    dataset = importlib.import_module("dataset")
    protocol = split.build_protocol(protocol_name)
    output = args.output_root / protocol_name / choice["modality"] / f"seed{seed}"
    output.mkdir(parents=True, exist_ok=False)
    release = search.RELEASE_MODALITY[choice["modality"]]
    search.seed_everything(seed)
    started = time.monotonic()
    train_data = dataset.XRF55Dataset(
        {release: split.collect(args.raw_root, release, protocol.train)}, (release,))
    assert len(train_data) == protocol.expected[0]
    options = dict(num_workers=args.workers, pin_memory=True,
                   persistent_workers=args.workers > 0)
    loader = DataLoader(train_data, batch_size=choice["batch_size"], shuffle=True,
                        generator=torch.Generator().manual_seed(seed), **options)
    constructor, _ = search.MODEL_SPECS[choice["modality"]]
    model = constructor()
    if choice["dropout"] > 0:
        model.fc = nn.Sequential(nn.Dropout(choice["dropout"]), model.fc)
    device = torch.device("cuda")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=choice["learning_rate"],
                                 weight_decay=choice["weight_decay"])
    milestones = search.HISTORICAL_SETTINGS[choice["modality"]]["milestones"]
    scheduler = (torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1)
                 if milestones else None)
    sources = ["test_selected_backbones.py", "search_backbone.py", "selected_backbones.json",
               *search.EXPECTED_LOCAL_SHA256,
               "protocol_snapshot/split.py", "protocol_snapshot/dataset.py"]
    report = dict(status="training", selection=choice, seed=seed, protocol=protocol_name,
                  initialization="random scratch", epochs=100, checkpoint_selection="fixed final epoch",
                  train_samples=len(train_data), train_membership=search.selector_record(protocol.train),
                  raw_root=str(args.raw_root.resolve()), workers=args.workers,
                  source_sha256={name: search.sha256(ROOT / name) for name in sources},
                  environment=dict(python=platform.python_version(), torch=torch.__version__,
                                   numpy=np.__version__, cuda=torch.version.cuda,
                                   gpu=torch.cuda.get_device_name()),
                  git_commit=subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip())
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    for epoch in range(1, 101):
        loss, accuracy, _ = search.run_epoch(model, loader, device, optimizer, None)
        if scheduler:
            scheduler.step()
        if epoch == 1 or epoch % 10 == 0:
            print(f"epoch={epoch} train_loss={loss:.6f} train_accuracy={accuracy:.6f}", flush=True)
    checkpoint = output / "model.state_dict.pt"
    torch.save(model.state_dict(), checkpoint)
    report.update(final_train_loss=loss, final_train_accuracy=accuracy,
                  checkpoint_sha256=search.sha256(checkpoint), evaluations={})
    # Test membership and data are accessed only after the fixed training budget.
    for name in protocol.test_splits():
        data = dataset.XRF55Dataset(
            {release: split.collect(args.raw_root, release, protocol.evaluations[name])}, (release,))
        assert len(data) == protocol.expected[1][name]
        evaluation_loader = DataLoader(data, batch_size=64, shuffle=False, **options)
        loss, accuracy, count = search.run_epoch(model, evaluation_loader, device, None, None)
        report["evaluations"][name] = dict(loss=loss, accuracy=accuracy, samples=count,
            membership=search.selector_record(protocol.evaluations[name]))
        print(f"test={name} accuracy={accuracy:.8f} loss={loss:.8f}", flush=True)
    assert report["evaluations"]
    report.update(status="completed", elapsed_seconds=time.monotonic() - started)
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
