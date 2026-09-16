"""Check grid coverage, source provenance and split membership isolation."""
import importlib.util
from pathlib import Path
import sys
import unittest

import protocol_backbone_grid as grid


class ProtocolGridTest(unittest.TestCase):
    def test_complete_grid(self):
        tasks = grid.cells(['unseen_subjects_001', *grid.ROOMS])
        self.assertEqual(len(tasks), 1152)
        self.assertEqual(len({tuple(c.items()) for c in tasks}), len(tasks))
        for protocol in ['unseen_subjects_001', *grid.ROOMS]:
            for modality in ('wifi', 'rfid', 'mmwave'):
                rows = [c for c in tasks if c['protocol'] == protocol and c['modality'] == modality]
                self.assertEqual(len(rows), 96)
                self.assertEqual({c['seed'] for c in rows}, {1, 2})
                self.assertEqual({c['learning_rate'] for c in rows}, {3e-5, 1e-4, 3e-4, 1e-3})

    def test_invalid_campaign(self):
        for protocols in ([], ['unseen_subjects_217'], ['unseen_subjects_001'] * 2):
            with self.assertRaises(ValueError):
                grid.cells(protocols)

    def test_snapshot_isolation(self):
        path = Path(__file__).parent / 'protocol_snapshot_v2/split.py'
        spec = importlib.util.spec_from_file_location('xfi_frozen_split', path)
        xs = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = xs
        spec.loader.exec_module(xs)
        for name in grid.SUBJECTS + grid.ROOMS:
            protocol = xs.build_protocol(name)
            people = {}
            for role, selector in protocol.splits():
                people[role] = set().union(*(xs.SCENE_SUBJECTS[scene] for scene in selector.scenes))
                if selector.subjects is not None:
                    people[role] &= selector.subjects
            for a, b in [('train','validation'), ('train','test'), ('validation','test')]:
                self.assertFalse(people[a] & people[b], name)
        self.assertEqual([xs.build_protocol(n).expected[0] for n in grid.ROOMS], [29700,30800,30800])

    def test_source_manifest_covers_catalog(self):
        hashes = grid.source_hashes()
        self.assertIn('protocol_snapshot_v2/subject_splits.json', hashes)
        self.assertIn('search_backbone.py', hashes)
        self.assertIn('search_protocol_backbone.py', hashes)


if __name__ == '__main__':
    unittest.main()
