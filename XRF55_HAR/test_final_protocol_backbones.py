import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
import torch._dynamo
from torch import nn
import final_protocol_backbones as final
import protocol_backbone_runner as runner

class FinalProtocolTests(unittest.TestCase):
    def inventory(self, root):
        files, splits = {}, {}
        for i, role in enumerate(('train', 'validation', 'test')):
            name = role + '.npy'
            (root/name).write_bytes(b'fixture')
            files[name] = dict(identity=['Scene1', i+1, 1, 1], bytes=7)
            splits[role] = {'modalities': {'WiFi': {'members': [name]}}}
        return dict(raw_root_hint=str(root), files=files, protocols={'fixture': {'splits':splits}})

    def test_union_and_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            inventory = self.inventory(Path(directory))
            self.assertEqual(final.memberships(inventory, 'fixture', 'wifi'),
                             (['train.npy','validation.npy'], ['test.npy']))
            inventory['files']['test.npy']['identity'] = inventory['files']['train.npy']['identity']
            with self.assertRaises(ValueError):
                final.memberships(inventory, 'fixture', 'wifi')

    def test_selection(self):
        rows = [dict(validation_accuracy=.2, task=0), dict(validation_accuracy=.3, task=1), dict(validation_accuracy=.3, task=2)]
        self.assertEqual(final.choose(rows)['task'], 1)

    def exercise(self, collapse):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            final.write(root/'dataset.json', self.inventory(root))
            task = dict(protocol='fixture', modality='wifi', seed=1, batch_size=16,
                        learning_rate=.001, weight_decay=.01, dropout=.3)
            final.write(root/'campaign.json', dict(tasks=[task], epochs=100, source_sha256={}, dataset_sha256=final.digest(root/'dataset.json')))
            seen, calls = [], []
            class Dataset:
                def __init__(self, samples, modalities):
                    self.samples = samples['WiFi']
                    seen.append([s.path.name for s in self.samples])
                def __len__(self):
                    return len(self.samples)
            class Model(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc = nn.Linear(4,55)
            def epoch(model, loader, device, optimizer, maximum):
                training = optimizer is not None
                calls.append(training)
                if not training:
                    self.assertEqual(len(calls),101)
                    self.assertTrue((root/'runs/00/model.state_dict.pt').exists())
                return 4., .018 if collapse else .5, len(loader)
            def sample(path, modality, scene, subject, action, repetition):
                return SimpleNamespace(path=path)
            with patch.object(final, 'sources', return_value={}), \
                 patch.dict(sys.modules, {'dataset':SimpleNamespace(XRF55Dataset=Dataset), 'split':SimpleNamespace(Sample=sample)}), \
                 patch.dict(runner.MODEL_SPECS, {'wifi':(Model,None)}), \
                 patch('torch.utils.data.DataLoader', side_effect=lambda dataset, **kwargs:dataset), \
                 patch('torch.device', return_value='cpu'), \
                 patch.object(runner, 'run_epoch', side_effect=epoch):
                final.execute(root,0)
            report=json.loads((root/'runs/00/result.json').read_text())
            self.assertEqual(seen[0], ['train.npy','validation.npy'])
            if collapse:
                self.assertEqual(len(calls),30)
                self.assertEqual(len(seen),1)
                self.assertFalse(report['test_data_opened'])
                self.assertEqual(report['status'],'aborted_at_chance')
            else:
                self.assertEqual(seen[1],['test.npy'])
                self.assertEqual(calls,[True]*100+[False])
                self.assertEqual(report['status'],'completed')

    def test_final_checkpoint_before_test(self):
        self.exercise(False)

    def test_collapse_skips_test(self):
        self.exercise(True)

if __name__ == '__main__':
    unittest.main()
