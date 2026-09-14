"""Operator review output must remain separate from scientific training."""
import importlib.util
import json
import sys
from pathlib import Path

import polars as pl
import pytest

from hgcl.config import Config, defaults
from hgcl.provenance import atomic_json, source_identity
import hgcl.lab_audit as audit_module


def test_audit_compacts_identifiers_and_counts_recurrence(tmp_path, monkeypatch):
    values = defaults('lab')
    values['paths']['project_root'] = str(tmp_path)
    values['split'] = {'train': [1], 'validation': [2], 'test': [3]}
    values['features']['regimes'] = ['principal']
    config = Config(json.dumps(values))
    prepared = tmp_path / 'prepared'
    atomic_json(prepared / 'manifest.json', {'code_hash': source_identity(tmp_path)['sha256'],
                                           'preparation_hash': 'p', 'payload_hash': 'd'})
    atomic_json(prepared / 'diagnostics.json', {
        'snapshots': {f'principal/{s}': dict(addresses=2, transactions=1, edges=2,
                                           address_dim=123, transaction_dim=32) for s in (1,2,3)},
        'source_counts': {}, 'selected_counts': {}, 'duplicates': {}, 'transaction_missingness': {}})
    atomic_json(prepared / 'principal/missingness.json', {})
    atomic_json(prepared / 'targets/budgets.json', [dict(seed=11, fraction=1., counts={},
                selected_overlap=1, hash='mask', selected={'train': ['secret-address']})])
    pl.DataFrame({'address': ['secret-address', 'secret-address', 'new-address'],
                  'step': [1,2,3], 'seen_in_train': [True,True,False],
                  'seen_in_development': [True,True,False]}).write_parquet(prepared / 'targets/recurrence.parquet')
    calls = []
    def validate(p, c):
        calls.append((p,c))
        return {'status': 'PASS'}
    monkeypatch.setattr(audit_module, 'validate', validate)
    result = audit_module.audit(config, prepared)
    assert calls == [(prepared,config)]
    assert 'secret-address' not in json.dumps(result) and 'new-address' not in json.dumps(result)
    assert result['recurrence']['test']['seen_in_development_observations'] == 0
    assert result['researcher_acceptance'] == 'pending' and not result['training_performed']
    # A failed validator must not be converted to a PASS receipt.
    def invalid(*args):
        raise ValueError('corrupt payload')
    monkeypatch.setattr(audit_module, 'validate', invalid)
    with pytest.raises(ValueError, match='corrupt payload'):
        audit_module.audit(config, prepared)


def test_lab_dry_run_never_executes_matrix_and_rejects_stale_audit(tmp_path, monkeypatch):
    import hgcl.config
    import hgcl.matrix
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location('lab_run', root / 'scripts/lab/run.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    values = defaults('lab')
    values['paths'].update(project_root=str(root), artifacts_root=str(tmp_path / 'artifacts'))
    config = Config(json.dumps(values))
    monkeypatch.setattr(hgcl.config, 'load_config', lambda p: config)
    monkeypatch.setenv('HGCL_NIX_SYSTEM', 'x86_64-linux')
    monkeypatch.setattr(sys, 'prefix', str(root / '.venv-lab'))
    monkeypatch.setattr(sys, 'argv', ['run.py', 'dry-run'])
    folder = config.artifacts_root / 'environment'
    receipt = dict(status='PASS', source_sha256=source_identity(root)['sha256'],
                   prepared=str(tmp_path / 'prepared'), preparation_hash='p', payload_hash='d')
    for stage in ('doctor','smoke','prepare','audit'):
        atomic_json(folder / f'lab-{stage}.json', receipt)
    def forbidden(*args, **kwargs):
        pytest.fail('Dry-run started scientific training')
    monkeypatch.setattr(hgcl.matrix, 'execute', forbidden)
    calls = []
    def dry_run(c,p):
        calls.append(p)
        return {**hgcl.matrix.design(c.values), 'status': 'planned', 'training_performed': False,
                'data_integrity_checked': True, 'preparation': receipt}
    monkeypatch.setattr(hgcl.matrix, 'dry_run', dry_run)
    assert module.main() == 0
    result = json.loads((folder / 'lab-dry-run.json').read_text())
    assert result['expected_evaluations'] == 100 and len(calls) == 1
    assert result['researcher_acceptance'] == 'pending'
    atomic_json(folder / 'lab-audit.json', {**receipt, 'payload_hash': 'old'})
    assert module.main() == 3 and len(calls) == 1
    assert json.loads((folder / 'lab-dry-run.json').read_text())['status'] == 'FAIL'
