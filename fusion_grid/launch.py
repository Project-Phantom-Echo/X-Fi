#!/usr/bin/env python3
"""Launch an extendable full X-Fi validation grid using split-matched backbones."""
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


def cells(stage):
    if stage=='extend-fusion':
        return [dict(protocol=p,seed=1,epochs=100,batch_size=16,learning_rate=lr,weight_decay=wd,dropout=drop)
                for p,lr,wd,drop in itertools.product(PROTOCOLS,(3e-5,1e-4,3e-4),(0.,.01,.1),(0.,.3))
                if not (wd==.01 and drop==0.)]
    wd={'initial':.01,'extend-wd':.1}[stage]
    return [dict(protocol=p,seed=1,epochs=100,batch_size=16,learning_rate=lr,weight_decay=wd)
            for p,lr in itertools.product(PROTOCOLS,(3e-5,1e-4,3e-4))]


def prepare(campaign,stage,protocol=None):
    shared=WORKSPACE/'wireless-sensing/experiments-on-xrf55'
    sys.path.insert(0,str(shared))
    from baseline_protocols import collect, selector_record, load_split
    inventory=shared/'jobs/proposed-seed1-four-splits/dataset.json';inv=json.loads(inventory.read_text())
    weights={};tasks=[];memberships={}
    for protocol in ([protocol] if protocol else PROTOCOLS):
        group='subjects' if protocol==PROTOCOLS[0] else 'rotations'
        backbone=WORKSPACE/f'compass/outputs/proposed-seed1-four-splits/{group}/backbones'/protocol
        data,meta=collect(inv['raw_root_hint'],protocol)
        for role,modalities in data.items():
            for modality,samples in modalities.items():
                assert [str(s.path.relative_to(inv['raw_root_hint'])) for s in samples]==inv['protocols'][protocol]['splits'][role]['modalities'][modality]['members']
        for modality in ('wifi','rfid','mmwave'):
            report=json.loads((backbone/modality/'result.json').read_text())
            if report['status']!='completed' or report['test_data_opened'] or report['protocol']!=protocol or report['train_membership']!=meta['train_membership']:
                raise ValueError(f'Invalid train-only backbone: {backbone/modality}')
            for name in ('result.json','model.state_dict.pt'):weights[str(backbone/modality/name)]=digest(backbone/modality/name)
        memberships[protocol]=meta
        tasks.extend(dict(**cell,backbone_runs=str(backbone)) for cell in cells(stage) if cell['protocol']==protocol)
    sources=list((REPO/'XRF55_HAR').glob('*.py'))+list((REPO/'XRF55_HAR/backbone_models').rglob('*.py'))
    sources += [shared/name for name in ('baseline_protocols.py','split.py','subject_splits.json')]
    campaign.mkdir(parents=True,exist_ok=False);campaign.chmod(0o777)
    for name in ('logs','runs'):(campaign/name).mkdir();(campaign/name).chmod(0o777)
    for source in sources:
        target=campaign/'source'/source.relative_to(WORKSPACE)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    for source,target in [('launch.py','launcher.py'),('submit.sh','submit.sh')]:shutil.copyfile(REPO/'fusion_grid'/source,campaign/target)
    shutil.copyfile(inventory,campaign/'dataset.json')
    hashes={str(p.relative_to(campaign)):digest(p) for p in (campaign/'source').rglob('*') if p.is_file()}
    hashes.update({name:digest(campaign/name) for name in ('launcher.py','submit.sh','dataset.json')})
    write(campaign/'manifest.json',dict(stage=stage,tasks=tasks,source_sha256=hashes,backbone_sha256=weights,
          memberships=memberships,raw_root=inv['raw_root_hint'],python=str(WORKSPACE/'compass/.venv/bin/python'),
          selection='epoch-100 validation accuracy; no test evaluation',
          backbone_policy='reuse split-matched train-only seed-1 backbones; frozen parameters',
          extension_policy='extensions use separate tasks and outputs; combine with initial results for selection; never rerun initial cells'))
    print(f'Prepared {len(tasks)} {stage} jobs at {campaign}')


def verify(campaign):
    m=json.loads((campaign/'manifest.json').read_text())
    for name,h in m['source_sha256'].items():
        if digest(campaign/name)!=h:raise ValueError(f'Frozen source changed: {name}')
    for name,h in m['backbone_sha256'].items():
        if digest(name)!=h:raise ValueError(f'Backbone changed: {name}')
    return m


def command(campaign,m,index):
    cmd=[m['python'],'-u',str(campaign/'source/x-fi/XRF55_HAR/run_protocol.py'),
         '--raw-root',m['raw_root'],'--output-dir',str(campaign/'runs'/f'task-{index:03d}'),'--workers','12']
    for key,value in m['tasks'][index].items():cmd+=['--'+key.replace('_','-'),str(value)]
    return cmd


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('initial','extend-wd','extend-fusion'),default='initial');p.add_argument('--campaign',type=Path)
    p.add_argument('--protocol',choices=PROTOCOLS)
    a=p.add_mutually_exclusive_group(required=True)
    a.add_argument('--prepare',action='store_true');a.add_argument('--submit',action='store_true');a.add_argument('--task',type=int)
    p.add_argument('--inspect',action='store_true');args=p.parse_args()
    if args.stage=='extend-fusion' and not args.protocol and args.task is None:
        p.error('--extend-fusion requires --protocol')
    suffix=args.protocol or 'four-splits'
    campaign=(args.campaign or REPO/f'XRF55_HAR/fusion_grid/{args.stage}-seed1-{suffix}').resolve()
    if args.prepare:prepare(campaign,args.stage,args.protocol);return
    if args.submit and not campaign.exists():prepare(campaign,args.stage,args.protocol)
    m=verify(campaign)
    if args.submit:
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
