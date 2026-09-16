#!/usr/bin/env python3
"""Train X-Fi backbones on the frozen HTML split catalog, validation only."""
from pathlib import Path
import hashlib
import json
import time

from near_chance import NearChanceStop, TrainingAtChance

import protocol_backbone_runner as runner

SNAPSHOT = Path(__file__).resolve().parent / 'protocol_snapshot_v2'
PROTOCOLS = tuple(f'unseen_subjects_{i:03d}' for i in range(1, 217)) + tuple(
    f'unseen_room_subject_rotation_{i}' for i in range(1, 4))


def configure():
    runner.VALIDATION_PROTOCOLS = PROTOCOLS
    runner.EXPECTED_PROTOCOL_SHA256 = json.loads((SNAPSHOT / 'sha256.json').read_text())


def main():
    configure()
    args = runner.parse_args()
    if args.protocol_code_dir.resolve() != SNAPSHOT:
        raise ValueError(f'This campaign requires the frozen snapshot: {SNAPSHOT}')
    if (args.output_dir / 'result.json').exists():
        raise FileExistsError(args.output_dir / 'result.json')
    stop = NearChanceStop()
    original_run_epoch = runner.run_epoch
    started = time.monotonic()

    def guarded_epoch(model, loader, device, optimizer, max_batches):
        metrics = original_run_epoch(model, loader, device, optimizer, max_batches)
        if optimizer is not None:
            stop.observe(*metrics)
        return metrics

    runner.run_epoch = guarded_epoch
    path = args.output_dir / 'result.json'
    try:
        runner.main()
        report = json.loads(path.read_text())
    except TrainingAtChance as error:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        report = {
            'status': 'aborted_at_chance', 'reason': str(error),
            'protocol': args.protocol, 'modality': args.modality, 'seed': args.seed,
            'epochs': args.epochs, 'batch_size': args.batch_size,
            'learning_rate': args.learning_rate, 'weight_decay': args.weight_decay,
            'dropout': args.dropout, 'validation_accuracy': None,
            'validation_loss': None, 'validated_samples': 0,
            'checkpoint_selection': 'none: training failed near-chance criterion',
            'test_data_opened': False, 'test_paths_enumerated': False,
            'elapsed_seconds': time.monotonic() - started,
            'raw_root': str(args.raw_root.resolve()),
            'protocol_code_dir': str(args.protocol_code_dir.resolve()),
        }
        print(f'aborted_at_chance: {error}', flush=True)
    finally:
        runner.run_epoch = original_run_epoch
    report['near_chance_stop'] = stop.record()
    report['protocol_source_sha256'] = runner.EXPECTED_PROTOCOL_SHA256
    report['protocol_runner_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report['subject_identity_assumption'] = 'matching subject IDs across scenes'
    # A bounded smoke test must never enter the ranking of complete grid cells.
    if args.max_train_batches or args.max_validation_batches or args.epochs != 100:
        report['status'] = 'smoke_test'
    path.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
