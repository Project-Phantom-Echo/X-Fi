import importlib.util
from pathlib import Path

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('full_xfi_launch',ROOT/'launch.py')
launch=importlib.util.module_from_spec(spec);spec.loader.exec_module(launch)


def test_extension_adds_only_new_cells():
    initial=launch.cells('initial');extra=launch.cells('extend-wd')
    assert len(initial)==len(extra)==12
    key=lambda x:tuple(sorted(x.items()))
    assert not set(map(key,initial))&set(map(key,extra))
    assert len(set(map(key,initial+extra)))==24
    assert {r['weight_decay'] for r in initial}=={.01}
    assert {r['weight_decay'] for r in extra}=={.1}
    assert {r['learning_rate'] for r in initial}=={3e-5,1e-4,3e-4}


def test_prepared_grid_uses_train_only_checkpoints_and_never_requests_test():
    campaign=launch.REPO/'XRF55_HAR/fusion_grid/initial-seed1-four-splits'
    manifest=launch.verify(campaign)
    for i,task in enumerate(manifest['tasks']):
        command=launch.command(campaign,manifest,i)
        assert '--evaluate-test' not in command
        assert command[command.index('--epochs')+1]=='100'
        assert command[command.index('--seed')+1]=='1'
        assert '/backbone_tests/' not in task['backbone_runs']


def test_fusion_extension_reuses_initial_cells():
    extra=launch.cells('extend-fusion')
    assert len(extra)==60
    key=lambda r:(r['protocol'],r['learning_rate'],r['weight_decay'],r.get('dropout',0.))
    initial=launch.cells('initial')
    assert not set(map(key,initial)) & set(map(key,extra))
    assert len(set(map(key,initial+extra)))==72
    for protocol in launch.PROTOCOLS:
        assert sum(r['protocol']==protocol for r in extra)==15
