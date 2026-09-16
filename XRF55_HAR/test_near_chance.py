import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from near_chance import NearChanceStop, TrainingAtChance


class NearChanceTest(unittest.TestCase):
    def test_stops_at_epoch_30_inclusive(self):
        stop = NearChanceStop()
        for _ in range(29):
            stop.observe(4.0, 0.022, 100)
        with self.assertRaises(TrainingAtChance):
            stop.observe(4.0, 0.022, 100)
        self.assertEqual(stop.record()['completed_epochs'], 30)

    def test_recovery_at_deadline(self):
        stop = NearChanceStop()
        for _ in range(29):
            stop.observe(4.0, 0.018, 100)
        stop.observe(3.9, 0.023, 100)
        for _ in range(70):
            stop.observe(4.0, 0.018, 100)
        self.assertEqual(stop.best_accuracy, 0.023)

    def test_stop_saved_without_validation(self):
        import search_protocol_backbone as entry
        with TemporaryDirectory() as directory:
            args = SimpleNamespace(output_dir=Path(directory), protocol_code_dir=entry.SNAPSHOT,
                raw_root=Path(directory), protocol='unseen_subjects_001', modality='wifi',
                seed=1, epochs=100, batch_size=16, learning_rate=0.0001,
                weight_decay=0.01, dropout=0.3, max_train_batches=None,
                max_validation_batches=None)
            evaluated = []
            def epoch(model, loader, device, optimizer, max_batches):
                if optimizer is None:
                    evaluated.append(True)
                return 4.0, 0.018, 100
            def training():
                for _ in range(100):
                    entry.runner.run_epoch(None, None, None, object(), None)
                entry.runner.run_epoch(None, None, None, None, None)
            with patch.object(entry.runner, 'parse_args', return_value=args), \
                 patch.object(entry.runner, 'run_epoch', side_effect=epoch), \
                 patch.object(entry.runner, 'main', side_effect=training):
                entry.main()
            report = json.loads((Path(directory) / 'result.json').read_text())
            self.assertEqual(report['status'], 'aborted_at_chance')
            self.assertEqual(report['near_chance_stop']['completed_epochs'], 30)
            self.assertIsNone(report['validation_accuracy'])
            self.assertEqual(evaluated, [])
            self.assertFalse(report['test_data_opened'])


if __name__ == '__main__':
    unittest.main()
