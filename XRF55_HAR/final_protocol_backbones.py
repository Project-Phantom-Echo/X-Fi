#!/usr/bin/env python3
"""Freeze validation-selected X-Fi configurations; retrain on train+validation."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def sources():
    from protocol_backbone_grid import source_hashes
    return {**source_hashes(), Path(__file__).name: digest(__file__)}


def choose(candidates):
    return max(candidates, key=lambda row: row['validation_accuracy'])


def memberships(inventory, protocol, modality):
    release = {'wifi': 'WiFi', 'rfid': 'RFID', 'mmwave': 'mmWave'}[modality]
    splits = inventory['protocols'][protocol]['splits']
    train = splits['train']['modalities'][release]['members']
    validation = splits['validation']['modalities'][release]['members']
    test = splits['test']['modalities'][release]['members']
    identities = lambda paths: {tuple(inventory['files'][p]['identity']) for p in paths}
    a, b, c = map(identities, (train, validation, test))
    if a & b or a & c or b & c:
        raise ValueError('train/validation/test identity overlap')
    return train + validation, test


def prepare(campaign, inventory_path):
    evidence, selected = [], []
    for group in ('seed1-subjects', 'seed1-rotations'):
        folder = ROOT / 'backbone_grid' / group
        manifest = json.loads((folder / 'campaign.json').read_text())
        from protocol_backbone_grid import source_hashes
        if source_hashes() != manifest['source_sha256']:
            raise RuntimeError('Grid sources changed')
        by_pair = {}
        for task, cell in enumerate(manifest['tasks']):
            name = '-'.join(f'{k}={v}' for k, v in cell.items())
            path = folder / cell['protocol'] / cell['modality'] / name / 'result.json'
            result = json.loads(path.read_text())
            if result['status'] != 'completed' or result['fixed_score_epoch'] != 100:
                raise ValueError(f'Incomplete grid: {path}')
            assert all(result[k] == v for k, v in cell.items())
            record = dict(cell=cell, campaign=group, task=task,
                          validation_accuracy=result['validation_accuracy'],
                          result_path=str(path), result_sha256=digest(path))
            evidence.append(record)
            by_pair.setdefault((cell['protocol'], cell['modality']), []).append(record)
        selected.extend(choose(values) for values in by_pair.values())
    assert len(evidence) == 576 and len(selected) == 12
    inventory = json.loads(inventory_path.read_text())
    sys.path.insert(0, str(ROOT / 'protocol_snapshot_v2'))
    import split
    for record in selected:
        cell = record['cell']
        release = {'wifi':'WiFi','rfid':'RFID','mmwave':'mmWave'}[cell['modality']]
        protocol = split.build_protocol(cell['protocol'])
        for role, selector in [('train', protocol.train), *protocol.evaluations.items()]:
            expected = inventory['protocols'][cell['protocol']]['splits'][role]['modalities'][release]['members']
            actual = [str(s.path.relative_to(inventory['raw_root_hint'])) for s in split.collect(Path(inventory['raw_root_hint']), release, selector)]
            assert actual == expected, (cell['protocol'], release, role)
        train, test = memberships(inventory, cell['protocol'], cell['modality'])
        record.update(train_samples=len(train), test_samples=len(test))
    tasks = [dict(**{**r['cell'], 'seed': seed}, selection=r) for r in selected for seed in (1,2,3)]
    campaign.mkdir(parents=True, exist_ok=False)
    (campaign / 'dataset.json').write_bytes(inventory_path.read_bytes())
    manifest = dict(tasks=tasks, epochs=100, training_membership='exact union of grid train and validation',
                    selection='seed-1 epoch-100 validation accuracy; ties by original task order',
                    selection_evidence=evidence, source_sha256=sources(),
                    dataset_sha256=digest(campaign/'dataset.json'),
                    near_chance_stop=dict(patience_epochs=30, threshold_accuracy=0.022),
                    test_policy='one evaluation after epoch 100; no test evaluation for aborted runs')
    write(campaign/'campaign.json', manifest)
    print(json.dumps({'tasks':len(tasks), 'pairs':[(r['cell']['protocol'],r['cell']['modality'],r['train_samples'],r['test_samples']) for r in selected]},indent=2))


def execute(campaign, task_id, inspect=False):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    import protocol_backbone_runner as runner
    from near_chance import NearChanceStop, TrainingAtChance
    manifest = json.loads((campaign/'campaign.json').read_text())
    if sources() != manifest['source_sha256'] or digest(campaign/'dataset.json') != manifest['dataset_sha256']:
        raise RuntimeError('Final campaign sources or membership changed')
    task = manifest['tasks'][task_id]
    inventory = json.loads((campaign/'dataset.json').read_text())
    train_paths, test_paths = memberships(inventory, task['protocol'], task['modality'])
    if inspect:
        print(json.dumps(dict(task=task, train_samples=len(train_paths), test_samples=len(test_paths), overlap=0)))
        return
    output = campaign / 'runs' / f'{task_id:02d}'
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    runner.seed_everything(task['seed'])
    sys.path.insert(0, str(ROOT/'protocol_snapshot_v2'))
    import dataset
    import split
    release = runner.RELEASE_MODALITY[task['modality']]
    raw_root = Path(inventory['raw_root_hint'])
    def make_data(paths):
        samples = []
        for path in paths:
            record = inventory['files'][path]
            absolute = raw_root/path
            if absolute.stat().st_size != record['bytes']:
                raise ValueError(f'Data size changed: {absolute}')
            scene, subject, action, repetition = record['identity']
            samples.append(split.Sample(absolute, release, scene, subject, action, repetition))
        return dataset.XRF55Dataset({release:samples}, (release,))
    # Test tensors are not read until training and checkpoint saving finish.
    train_data = make_data(train_paths)
    options = dict(num_workers=12, pin_memory=True, persistent_workers=True)
    train_loader = DataLoader(train_data, batch_size=task['batch_size'], shuffle=True,
                             generator=torch.Generator().manual_seed(task['seed']), **options)
    model = runner.MODEL_SPECS[task['modality']][0]()
    if task['dropout']:
        model.fc = nn.Sequential(nn.Dropout(task['dropout']), model.fc)
    device = torch.device('cuda')
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=task['learning_rate'], weight_decay=task['weight_decay'])
    milestones = runner.HISTORICAL_SETTINGS[task['modality']]['milestones']
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1) if milestones else None
    stop = NearChanceStop()
    history = []
    report = dict(task=task, train_samples=len(train_data), test_samples=len(test_paths),
                  test_data_opened=False, source_sha256=manifest['source_sha256'],
                  dataset_sha256=manifest['dataset_sha256'], validation_during_retraining=False)
    try:
        for epoch in range(1, manifest['epochs']+1):
            lr = optimizer.param_groups[0]['lr']
            loss, accuracy, count = runner.run_epoch(model, train_loader, device, optimizer, None)
            if scheduler:
                scheduler.step()
            history.append(dict(epoch=epoch, learning_rate=lr, train_loss=loss, train_accuracy=accuracy, samples=count))
            write(output/'history.json', history)
            print(f'epoch={epoch} train_loss={loss:.6f} train_accuracy={accuracy:.6f}', flush=True)
            stop.observe(loss, accuracy, count)
    except TrainingAtChance as error:
        report.update(status='aborted_at_chance', reason=str(error), near_chance_stop=stop.record(),
                      elapsed_seconds=time.monotonic()-started)
        write(output/'result.json', report)
        return
    torch.save(model.state_dict(), output/'model.state_dict.pt')
    test_data = make_data(test_paths)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False,
                            generator=torch.Generator().manual_seed(task['seed']+1000000), **options)
    test_loss, test_accuracy, tested = runner.run_epoch(model, test_loader, device, None, None)
    report.update(status='completed', epochs=100, history=history, test_data_opened=True,
                  test_loss=test_loss, test_accuracy=test_accuracy, tested_samples=tested,
                  near_chance_stop=stop.record(), elapsed_seconds=time.monotonic()-started,
                  checkpoint_selection='fixed epoch 100', model_path=str(output/'model.state_dict.pt'))
    write(output/'result.json', report)
    print(f'test_loss={test_loss:.6f} test_accuracy={test_accuracy:.6f}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--inventory', type=Path)
    parser.add_argument('--task', type=int)
    parser.add_argument('--inspect', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare(args.campaign.resolve(), args.inventory.resolve())
    else:
        execute(args.campaign.resolve(), args.task, args.inspect)


if __name__ == '__main__':
    main()
