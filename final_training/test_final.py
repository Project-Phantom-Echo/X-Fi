import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]


def test_final_union_and_test_only_after_training(tmp_path):
    campaign=ROOT/'XRF55_HAR/fusion_grid/final-expanded-four-splits-seeds123'
    source=campaign/'source/x-fi/XRF55_HAR'
    sys.path.insert(0,str(source))
    spec=importlib.util.spec_from_file_location('final_xfi_run',source/'run_protocol.py')
    run=importlib.util.module_from_spec(spec);spec.loader.exec_module(run)
    class Tiny(nn.Module):
        def __init__(self,*args,**kwargs):
            super().__init__();self.feature_extractor=nn.Identity();self.linear_projector=nn.Linear(1,4);self.X_Fusion_block=nn.Linear(4,55)
        def forward(self,mmwave,wifi,rfid,mask):return self.X_Fusion_block(self.linear_projector(wifi))
    seen=[]
    def dataset(samples):
        ids=[s.identity for s in samples['WiFi']];seen.append(ids)
        return [(np.ones(1,dtype=np.float32),)*3+(0,) for _ in ids]
    def collect(root,protocol,roles=None):
        if roles==('test',):
            assert len(json.loads((tmp_path/'out/history.json').read_text()))==2
            return {'test':{'WiFi':[SimpleNamespace(identity=4)]}},{}
        return {r:{'WiFi':[SimpleNamespace(identity=i) for i in ids]} for r,ids in [('train',[1,2]),('validation',[3])]}, {'train_membership':{}}
    argv=['run','--raw-root',str(tmp_path),'--output-dir',str(tmp_path/'out'),'--backbone-runs',str(tmp_path),'--protocol','unseen_subjects_001','--seed','1','--epochs','2','--batch-size','2','--workers','0','--device','cpu','--train-with-validation','--evaluate-test']
    with patch('sys.argv',argv),patch.object(run,'collect',side_effect=collect),patch.object(run,'FusionDataset',side_effect=dataset),patch.object(run,'load_backbone_runs',return_value=({},{})),patch.object(run,'X_Fi',Tiny):run.main()
    assert seen==[[1,2,3],[4]]
    result=json.loads((tmp_path/'out/result.json').read_text())
    assert result['test_data_opened'] and result['test']['samples']==1
    assert all('validation_loss' not in r for r in result['history'])
