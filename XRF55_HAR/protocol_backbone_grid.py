#!/usr/bin/env python3
"""Create and execute the proposed 48-cell X-Fi backbone grid."""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SUBJECTS = tuple(f'unseen_subjects_{i:03d}' for i in range(1, 217))
ROOMS = tuple(f'unseen_room_subject_rotation_{i}' for i in range(1, 4))


def cells(protocols, seeds=(1, 2)):
    if not protocols or len(protocols) != len(set(protocols)):
        raise ValueError('Specify distinct protocols')
    if any(p not in SUBJECTS + ROOMS for p in protocols):
        raise ValueError('Unknown protocol')
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError('Specify distinct seeds')
    keys = ('protocol', 'modality', 'batch_size', 'learning_rate', 'weight_decay', 'dropout', 'seed')
    return [dict(zip(keys, values)) for values in itertools.product(
        protocols, ('wifi', 'rfid', 'mmwave'), (16, 32, 64),
        (3e-5, 1e-4, 3e-4, 1e-3), (0.01, 0.3), (0.0, 0.3), seeds)]


def source_hashes():
    paths = [ROOT / name for name in ('search_protocol_backbone.py', 'search_backbone.py', 'protocol_backbone_runner.py',
             'protocol_backbone_grid.py', 'near_chance.py', 'pretrain_backbones.py', 'XRF55_Dataset.py')]
    paths += sorted((ROOT / 'protocol_snapshot_v2').glob('*.py'))
    paths += sorted((ROOT / 'protocol_snapshot_v2').glob('*.json'))
    paths += sorted((ROOT / 'backbone_models').glob('*/ResNet.py'))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def command(cell, campaign, raw_root):
    name = '-'.join(f'{key}={value}' for key, value in cell.items())
    args = [sys.executable, '-u', str(ROOT / 'search_protocol_backbone.py'),
            '--raw-root', str(raw_root), '--protocol-code-dir', str(ROOT / 'protocol_snapshot_v2'),
            '--output-dir', str(campaign / cell['protocol'] / cell['modality'] / name),
            '--epochs', '100', '--workers', '12']
    for key, value in cell.items():
        args.extend(['--' + key.replace('_', '-'), str(value)])
    return args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--protocol', action='append')
    parser.add_argument('--raw-root', type=Path)
    parser.add_argument('--task', type=int)
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 2])
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    manifest_path = campaign / 'campaign.json'
    if args.task is None:
        if not args.raw_root or not args.raw_root.is_dir():
            parser.error('--raw-root must name an existing dataset')
        tasks = cells(args.protocol, args.seeds)
        campaign.mkdir(parents=True, exist_ok=False)
        manifest = {'tasks': tasks, 'raw_root': str(args.raw_root.resolve()),
                    'source_sha256': source_hashes(), 'epochs': 100,
                    'selection': 'mean final-epoch validation accuracy over configured seeds',
                    'seeds': args.seeds, 'validation_interval_epochs': 1,
                    'near_chance_stop': {'patience_epochs':30, 'threshold_accuracy':0.022},
                    'test_data_opened': False}
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
        print(f'{len(tasks)} jobs; submit --array=0-{len(tasks)-1}')
        return
    manifest = json.loads(manifest_path.read_text())
    if source_hashes() != manifest['source_sha256']:
        raise RuntimeError('Source changed since campaign creation; refusing mixed runs')
    if not 0 <= args.task < len(manifest['tasks']):
        parser.error('task index outside campaign')
    cmd = command(manifest['tasks'][args.task], campaign, manifest['raw_root'])
    os.execv(sys.executable, cmd)


if __name__ == '__main__':
    main()
