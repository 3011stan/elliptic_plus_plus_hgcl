"""Orchestration without running the scientific experiment or requiring CUDA."""
import json
from pathlib import Path
import pytest
from hgcl.config import Config, defaults
from hgcl.provenance import atomic_json, file_hash, source_identity
import hgcl.matrix as matrix
from hgcl.cli import main


def test_serial_restart_reuses_groups_and_ten_caches(tmp_path,monkeypatch):
    value=defaults('lab');value['paths'].update(project_root=str(tmp_path),artifacts_root=str(tmp_path/'artifacts'),data_root=str(tmp_path/'raw'))
    config=Config(json.dumps(value,sort_keys=True))
    (tmp_path/'requirements').mkdir();(tmp_path/'requirements/lab-cuda.lock').write_text('fixture only')
    prepared=tmp_path/'prepared';(prepared/'targets').mkdir(parents=True)
    atomic_json(prepared/'manifest.json',{'preparation_hash':'p','payload_hash':'d'})
    records=[{'seed':s,'fraction':f,'hash':f'{s}-{f}'} for s in value['labels']['seeds'] for f in value['labels']['fractions']]
    atomic_json(prepared/'targets/budgets.json',records)
    monkeypatch.setattr(matrix,'dry_run',lambda c,p:{**matrix.design(c.values),'status':'planned'})
    calls=[];caches=set();fail=True
    def fake_fit(c,p,run_id,*,record,regime,ssl_directory):
        run=c.artifacts_root/'runs'/run_id;run.mkdir(parents=True)
        deps={'config':c.hash,'source':source_identity(c.root)['sha256'],'preparation':'p','payload':'d',
              'environment_lock':file_hash(c.root/'requirements/lab-cuda.lock'), 'mask':record['hash'],'regime':regime,'seed':record['seed']}
        atomic_json(run/'provenance.json',{'dependencies':deps})
        atomic_json(run/'status.json',{'state':'frozen'})
        calls.append(run_id);caches.add(str(ssl_directory))
        return {'run':str(run)}
    def fake_evaluate(run):
        nonlocal fail
        run=Path(run)
        if len(calls)==3 and fail:
            fail=False
            raise InterruptedError('simulated interruption')
        atomic_json(run/'metrics.json',{})
        atomic_json(run/'status.json',{'state':'complete'})
        return {'status':'complete','run':str(run)}
    monkeypatch.setattr(matrix,'fit',fake_fit);monkeypatch.setattr(matrix,'evaluate',fake_evaluate)
    monkeypatch.setattr(matrix,'resume',fake_evaluate)
    with pytest.raises(InterruptedError):matrix.execute(config,prepared,'fixture')
    folder=config.artifacts_root/'matrices/fixture'
    assert sum(g['state']=='incomplete' for g in json.loads((folder/'groups.json').read_text()).values())==1
    result=matrix.execute(config,prepared,'fixture')
    assert result['evaluations']==100 and len(calls)==25 and len(caches)==10
    matrix.execute(config,prepared,'fixture')
    assert len(calls)==25
    # Changing source/lock identity prevents any further group execution.
    (tmp_path/'requirements/lab-cuda.lock').write_text('changed')
    with pytest.raises(ValueError,match='Incompatible matrix restart'):matrix.execute(config,prepared,'fixture')


def test_cli_report_marks_absent_matrix_results_incomplete(tmp_path,capsys):
    plan=matrix.design(defaults('lab'))
    atomic_json(tmp_path/'manifest.json',plan)
    assert main(['report','--matrix',str(tmp_path)])==4
    report=json.loads(capsys.readouterr().out)
    assert report['status']=='incomplete' and len(report['missing_evaluations'])==100
    assert (tmp_path/'report.md').exists()
