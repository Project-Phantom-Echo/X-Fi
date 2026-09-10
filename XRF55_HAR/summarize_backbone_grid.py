#!/usr/bin/env python3
"""Rank complete X-Fi backbone grid cells by the seeds 1-and-2 mean."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


FIELDS = ("batch_size", "learning_rate", "weight_decay", "dropout")
REQUIRED_SEEDS = {1, 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grouped: dict[tuple[str, str], dict[tuple, dict[int, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    reports = []
    if args.root.is_file():
        evidence = json.loads(args.root.read_text())
        inputs = [(row["source_record"], {**evidence["provenance"], **row})
                  for row in evidence["runs"]]
    else:
        inputs = [(str(path), json.loads(path.read_text()))
                  for path in sorted(args.root.rglob("result.json"))]
    for path, report in inputs:
        if report.get("status") != "completed" or report.get("test_data_opened") is not False:
            continue
        if (report.get("epochs") != 100 or report.get("fixed_score_epoch") != 100
                or report.get("max_train_batches") is not None
                or report.get("max_validation_batches") is not None):
            continue
        reports.append(report)
        key = tuple(report[field] for field in FIELDS)
        group = (report["protocol"], report["modality"])
        if int(report["seed"]) in grouped[group][key]:
            raise SystemExit(f"duplicate seed for {group} {key}: {path}")
        grouped[group][key][int(report["seed"])] = float(
            report["validation_accuracy"]
        )

    if not reports:
        raise SystemExit(f"no completed result.json files under {args.root}")
    runner_hashes = {report.get("runner_sha256") for report in reports}
    protocol_hashes = {report.get("split_py_sha256") for report in reports}
    dataset_hashes = {report.get("dataset_py_sha256") for report in reports}
    if any(len(hashes) != 1 for hashes in (runner_hashes, protocol_hashes, dataset_hashes)):
        raise SystemExit("refusing to mix results produced by different code snapshots")
    for protocol, modality in sorted(grouped):
        complete = []
        for key, scores in grouped[(protocol, modality)].items():
            if set(scores) == REQUIRED_SEEDS:
                complete.append((mean(scores.values()), key, scores))
        print(
            f"{protocol}/{modality}: {len(complete)} complete two-seed cells"
        )
        for rank, (score, key, scores) in enumerate(sorted(complete, reverse=True)[:10], 1):
            config = " ".join(f"{name}={value:g}" for name, value in zip(FIELDS, key))
            print(
                f"  {rank:2d}. mean={score:.6f} seed1={scores[1]:.6f} "
                f"seed2={scores[2]:.6f} {config}"
            )


if __name__ == "__main__":
    main()
