import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import TensorDataset
import protocol_backbone_runner as runner


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 55)

    def forward(self, x):
        return self.fc(x.flatten(1))


class EpochValidationTest(unittest.TestCase):
    def test_history_saved_each_epoch_and_final_score_matches(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ('split.py', 'dataset.py'):
                (root / name).write_text('test fixture')
            args = SimpleNamespace(raw_root=root, protocol_code_dir=root, output_dir=root,
                protocol='test', modality='wifi', seed=1, epochs=3, batch_size=2,
                learning_rate=0.001, weight_decay=0.01, dropout=0.0, workers=0,
                validation_batch_size=2, max_train_batches=None, max_validation_batches=None,
                save_model=False, device='cpu')
            selector = SimpleNamespace(scenes=('Scene1',), subjects=None,
                                       repetitions=None, part1_only=False)
            protocol = SimpleNamespace(train=selector, evaluations={'validation':selector}, description='test')
            dataset = TensorDataset(torch.ones(4,1,4), torch.zeros(4,dtype=torch.long))
            with patch.object(runner, 'parse_args', return_value=args), \
                 patch.object(runner, 'check_code_snapshot'), \
                 patch.object(runner, 'load_shared_data', return_value=(dataset,dataset,protocol,'validation')), \
                 patch.dict(runner.MODEL_SPECS, {'wifi':(TinyModel,None)}):
                runner.main()
            history=json.loads((root/'history.json').read_text())
            result=json.loads((root/'result.json').read_text())
            self.assertEqual([r['epoch'] for r in history],[1,2,3])
            self.assertTrue(all(r['validation_samples']==4 for r in history))
            self.assertEqual(result['history'],history)
            self.assertEqual(result['validation_accuracy'],history[-1]['validation_accuracy'])
            self.assertFalse(result['test_data_opened'])


if __name__ == '__main__':
    unittest.main()
