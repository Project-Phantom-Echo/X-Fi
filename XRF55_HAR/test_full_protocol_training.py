import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
import run_protocol as run


class TinyXFi(nn.Module):
    def __init__(self,*args,**kwargs):
        super().__init__();self.feature_extractor=nn.Identity()
        self.linear_projector=nn.Linear(1,4);self.X_Fusion_block=nn.Linear(4,55)
    def forward(self,mmwave,wifi,rfid,mask):return self.X_Fusion_block(self.linear_projector(wifi))


def test_full_xfi_validation_logged_every_epoch():
    with TemporaryDirectory() as tmp:
        out=Path(tmp)/'run';ds=[(np.ones(1,dtype=np.float32),)*3+(0,) for _ in range(4)]
        argv=['run_protocol.py','--raw-root',tmp,'--output-dir',str(out),'--backbone-runs',tmp,
              '--protocol','unseen_subjects_001','--seed','1','--epochs','2','--device','cpu','--workers','0','--batch-size','2']
        with patch('sys.argv',argv),patch.object(run,'collect',return_value=({'train':ds,'validation':ds},{'train_membership':{}})) as collect, \
             patch.object(run,'FusionDataset',side_effect=lambda x:x), \
             patch.object(run,'load_backbone_runs',return_value=({},{})),patch.object(run,'X_Fi',TinyXFi):
            run.main()
        assert collect.call_count==1
        h=json.loads((out/'history.json').read_text());assert [r['epoch'] for r in h]==[1,2]
        assert all(np.isfinite(r['validation_loss']) for r in h)
        assert not json.loads((out/'result.json').read_text())['test_data_opened']


def test_real_xfi_accepts_protocol_backbone_models():
    from X_Fi import X_Fi
    from pretrain_backbones import MODEL_SPECS
    models={m:spec[0]() for m,spec in MODEL_SPECS.items()}
    model=X_Fi(5,55,backbone_source='protocol',backbone_models=models).eval()
    with torch.inference_mode():
        logits=model(torch.zeros(1,17,256,128),torch.zeros(1,270,1000),
                     torch.zeros(1,23,148),[True,True,True])
    assert logits.shape==(1,55) and torch.isfinite(logits).all()


def test_fusion_dropout_is_configurable():
    from X_Fi import X_Fi
    models={m:spec[0]() for m,spec in run.MODEL_SPECS.items()}
    model=X_Fi(1,55,backbone_source='protocol',backbone_models=models,dropout=.3)
    drops=[m.p for m in model.X_Fusion_block.modules() if isinstance(m,nn.Dropout)]
    assert drops and all(p==.3 for p in drops)
