#!/usr/bin/env python3
"""Full X-Fi on shared train/validation/test splits, with per-epoch monitoring."""
import argparse
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from X_Fi import X_Fi
from pretrain_backbones import MODEL_SPECS
from utils import generate_none_empth_modality_list

sys.path.append(str(Path(__file__).resolve().parents[2] / 'wireless-sensing/experiments-on-xrf55'))
from baseline_protocols import (NEW_PROTOCOLS, collect, FusionDataset, load_backbone_runs,
                                evaluation_rng, save_history)


def epoch(model, loader, device, optimizer=None, max_batches=None, mask=None):
    model.train(optimizer is not None)
    loss_sum=correct=count=0
    for index, (wifi,rfid,mmwave,labels) in enumerate(loader):
        if max_batches is not None and index>=max_batches:break
        wifi,rfid,mmwave,labels=(v.to(device) for v in (wifi,rfid,mmwave,labels))
        chosen=generate_none_empth_modality_list() if optimizer is not None else (mask or [True]*3)
        if optimizer is not None:optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(optimizer is not None):
            logits=model(mmwave,wifi,rfid,chosen)
            loss=torch.nn.functional.cross_entropy(logits,labels.long())
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
        if optimizer is not None:loss.backward();optimizer.step()
        count+=len(labels);loss_sum+=loss.item()*len(labels)
        correct+=(logits.argmax(1)==labels).sum().item()
    if count==0:raise ValueError('Empty epoch')
    return {'loss':loss_sum/count,'accuracy':correct/count,'samples':count}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-root',type=Path,required=True)
    p.add_argument('--protocol',choices=NEW_PROTOCOLS,required=True)
    p.add_argument('--backbone-runs',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--learning-rate',type=float,default=1e-4)
    p.add_argument('--weight-decay',type=float,default=.01)
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--device',default='cuda')
    p.add_argument('--evaluate-test',action='store_true')
    p.add_argument('--max-train-batches',type=int)
    p.add_argument('--max-eval-batches',type=int)
    args=p.parse_args()
    if args.epochs<1 or args.batch_size<1:raise ValueError('Positive epochs/batch size required')
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    data,metadata=collect(args.raw_root,args.protocol)
    models,provenance=load_backbone_runs(args.backbone_runs,
        {m:spec[0] for m,spec in MODEL_SPECS.items()},args.protocol,metadata['train_membership'])
    device=torch.device(args.device)
    model=X_Fi(5,55,backbone_source='protocol',backbone_models=models).to(device)
    model.feature_extractor.requires_grad_(False)
    optimizer=torch.optim.AdamW(list(model.linear_projector.parameters())+list(model.X_Fusion_block.parameters()),
                                lr=args.learning_rate,weight_decay=args.weight_decay)
    def loader(samples,shuffle=False):
        return DataLoader(FusionDataset(samples),batch_size=args.batch_size,shuffle=shuffle,
            num_workers=args.workers,pin_memory=device.type=='cuda',
            generator=torch.Generator().manual_seed(args.seed+(0 if shuffle else 1000000)))
    train,validation=loader(data['train'],True),loader(data['validation'])
    args.output_dir.mkdir(parents=True,exist_ok=False)
    metadata.update(configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                    backbones=provenance,checkpoint_selection='fixed final epoch',validation_interval_epochs=1)
    save_history(args.output_dir/'metadata.json',metadata)
    history=[];started=time.monotonic()
    for number in range(1,args.epochs+1):
        random.seed(number-1)
        tr=epoch(model,train,device,optimizer,args.max_train_batches)
        with evaluation_rng():val=epoch(model,validation,device,max_batches=args.max_eval_batches)
        history.append({'epoch':number,'learning_rate':optimizer.param_groups[0]['lr'],
                        'train_loss':tr['loss'],'train_accuracy':tr['accuracy'],
                        'validation_loss':val['loss'],'validation_accuracy':val['accuracy']})
        save_history(args.output_dir/'history.json',history)
        print(json.dumps(history[-1]),flush=True)
    torch.save(model.state_dict(),args.output_dir/'model.state_dict.pt')
    results={'status':'smoke_test' if args.max_train_batches or args.max_eval_batches else 'completed',
             'protocol':args.protocol,'seed':args.seed,'history':history,'elapsed_seconds':time.monotonic()-started,
             'test_data_opened':False}
    if args.evaluate_test:
        test,_=collect(args.raw_root,args.protocol,roles=('test',))
        results['test']=epoch(model,loader(test['test']),device,max_batches=args.max_eval_batches)
        results['test_data_opened']=True
    save_history(args.output_dir/'result.json',results)


if __name__=='__main__':main()
