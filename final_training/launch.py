#!/usr/bin/env python3
"""Launch three-seed final full X-Fi training and testing."""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

REPO=Path(__file__).resolve().parents[1]
WORKSPACE=REPO.parent
PROTOCOLS=['unseen_subjects_001']+[f'unseen_room_subject_rotation_{i}' for i in (1,2,3)]


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def prepare(campaign,stage):
    base=REPO/'XRF55_HAR/fusion_grid'
    grids=[base/'initial-seed1-four-splits']+[base/f'extend-fusion-seed1-{p}' for p in PROTOCOLS]
    rows=[];weights={}
    import math
    for grid in grids:
        manifest=verify(grid)
        weights.update(manifest['backbone_sha256'])
        for i,original in enumerate(manifest['tasks']):
            cell=dict(original);cell.setdefault('dropout',0.0)
            path=grid/'runs'/f'task-{i:03d}'/'result.json'
            result=json.loads(path.read_text());h=result['history']
            metadata=json.loads((path.parent/'metadata.json').read_text())
            if result['status']!='completed' or result['test_data_opened'] or result['protocol']!=cell['protocol']:
                raise ValueError(f'Invalid search run {grid.name}/{i}')
            if any(metadata['configuration'].get(k,0.0 if k=='dropout' else None)!=v for k,v in cell.items()):
                raise ValueError(f'Configuration mismatch {grid.name}/{i}')
            if [e['epoch'] for e in h]!=list(range(1,101)) or any(not math.isfinite(e[k]) for e in h for k in ('train_loss','train_accuracy','validation_loss','validation_accuracy')):
                raise ValueError(f'Incomplete/nonfinite search run {grid.name}/{i}')
            rows.append(dict(task=i,campaign=str(grid),cell=cell,validation_accuracy=h[-1]['validation_accuracy'],result_sha256=digest(path)))
    for protocol in PROTOCOLS:
        cells=[r['cell'] for r in rows if r['cell']['protocol']==protocol]
        if len(cells)!=18 or len({(c['learning_rate'],c['weight_decay'],c['dropout']) for c in cells})!=18:
            raise ValueError(f'Expected 18 unique configurations for {protocol}')
    # Extension snapshots include configurable fusion dropout; require identical sources and inventory.
    grid=grids[1];manifest=verify(grid)
    for other in grids[2:]:
        for name,h in manifest['source_sha256'].items():
            if name.startswith('source/') and digest(other/name)!=h:
                raise ValueError(f'Extension source mismatch: {other}/{name}')
        if digest(other/'dataset.json')!=digest(grid/'dataset.json'):
            raise ValueError('Extension inventory mismatch')
    selected=[max((r for r in rows if r['cell']['protocol']==p),key=lambda r:r['validation_accuracy']) for p in PROTOCOLS]
    tasks=[dict(**{k:v for k,v in row['cell'].items() if k!='seed'},seed=seed) for row in selected for seed in (1,2,3)]
    campaign.mkdir(parents=True,exist_ok=False);campaign.chmod(0o777)
    for name in ('logs','runs'):(campaign/name).mkdir();(campaign/name).chmod(0o777)
    shutil.copytree(grid/'source',campaign/'source',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copyfile(grid/'dataset.json',campaign/'dataset.json')
    for source,target in [('launch.py','launcher.py'),('submit.sh','submit.sh'),('train-validation.patch','train-validation.patch')]:
        shutil.copyfile(REPO/'final_training'/source,campaign/target)
    subprocess.run(['patch','--batch','--forward','-p0','-i',str(campaign/'train-validation.patch')],cwd=campaign/'source',check=True)
    hashes={str(p.relative_to(campaign)):digest(p) for p in (campaign/'source').rglob('*') if p.is_file()}
    hashes.update({name:digest(campaign/name) for name in ('launcher.py','submit.sh','dataset.json','train-validation.patch')})
    write(campaign/'manifest.json',dict(tasks=tasks,selected=selected,evidence=rows,source_sha256=hashes,
        backbone_sha256=weights,raw_root=manifest['raw_root'],python=manifest['python'],
        selection='epoch-100 validation accuracy across 18 configurations; ties by initial then extension task order',training='exact train+validation union; seeds 1/2/3; 100 epochs',
        backbone_policy=manifest['backbone_policy'],test_policy='all-modality test once after fixed final epoch'))
    print(f'Prepared {len(tasks)} final jobs at {campaign}')


def verify(campaign):
    m=json.loads((campaign/'manifest.json').read_text())
    for name,h in m['source_sha256'].items():
        if digest(campaign/name)!=h:raise ValueError(f'Frozen source changed: {name}')
    for name,h in m['backbone_sha256'].items():
        if digest(name)!=h:raise ValueError(f'Backbone changed: {name}')
    return m


def command(campaign,m,index):
    cmd=[m['python'],'-u',str(campaign/'source/x-fi/XRF55_HAR/run_protocol.py'),
         '--raw-root',m['raw_root'],'--output-dir',str(campaign/'runs'/f'task-{index:03d}'),'--workers','12','--train-with-validation','--evaluate-test']
    for key,value in m['tasks'][index].items():cmd+=['--'+key.replace('_','-'),str(value)]
    return cmd


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.set_defaults(stage='extend-fusion');p.add_argument('--campaign',type=Path)
    a=p.add_mutually_exclusive_group(required=True)
    a.add_argument('--prepare',action='store_true');a.add_argument('--submit',action='store_true');a.add_argument('--task',type=int)
    p.add_argument('--inspect',action='store_true');args=p.parse_args()
    campaign=(args.campaign or REPO/'XRF55_HAR/fusion_grid/final-expanded-four-splits-seeds123').resolve()
    if args.prepare:prepare(campaign,args.stage);return
    if args.submit and not campaign.exists():prepare(campaign,args.stage)
    m=verify(campaign)
    if args.submit:
        if len(m.get('evidence',[]))!=72:raise ValueError('Final submission requires all 72 search results')
        lock=campaign/'submission.lock';lock.mkdir(exist_ok=False)
        cmd=['sbatch','--parsable','--export=ALL',f'--array=0-{len(m["tasks"])-1}',
             f'--output={campaign}/logs/%x_%A_%a.out',f'--error={campaign}/logs/%x_%A_%a.err',
             str(campaign/'submit.sh'),str(campaign),m['python']]
        result=subprocess.run(cmd,text=True,capture_output=True)
        if result.returncode:lock.rmdir();raise RuntimeError(result.stderr)
        write(campaign/'submission.json',dict(job_id=result.stdout.strip(),user=os.environ.get('USER'),command=cmd))
        print(result.stdout.strip());return
    if args.task<0 or args.task>=len(m['tasks']):raise ValueError('Invalid task')
    cmd=command(campaign,m,args.task)
    if args.inspect:print(json.dumps(cmd));return
    (campaign/f'task-{args.task:03d}.lock').mkdir(exist_ok=False)
    raise SystemExit(subprocess.call(cmd,cwd=campaign/'source/x-fi/XRF55_HAR'))


if __name__=='__main__':main()
