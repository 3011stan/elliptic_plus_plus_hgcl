"""Train/evaluation access boundaries on a miniature valid source fixture."""
from pathlib import Path
import json
import polars as pl
import pytest
import yaml
from hgcl.config import defaults,load_config
from hgcl.data.schema import TX_COLUMNS,NATIVE_COLUMNS
from hgcl.provenance import file_hash
from hgcl.pipeline import prepare
from hgcl.training.runner import fit,aligned
from hgcl.training.supervised import permitted_targets
from hgcl.evaluation.metrics import metrics


@pytest.fixture
def prepared_fixture(tmp_path):
    root=tmp_path/'project';data=root/'raw';data.mkdir(parents=True)
    transactions=[];wallets=[];senders=[];receivers=[];labels={'shared':1}
    for step in (1,2,29,35):
        recipient=f'licit-{step}';labels[recipient]=2
        wallets.extend([{'address':a,'Time step':step,**{c:1. for c in NATIVE_COLUMNS}} for a in ('shared',recipient)])
        for i in range(2):
            tx_id=f'{step}-{i}'
            transactions.append({'txId':tx_id,'Time step':step,**{c:float(i+1) for c in TX_COLUMNS}})
            senders.append({'input_address':'shared','txId':tx_id})
            receivers.append({'txId':tx_id,'output_address':recipient})
    tables={'txs_features.csv':pl.DataFrame(transactions),'wallets_features.csv':pl.DataFrame(wallets),
            'AddrTx_edgelist.csv':pl.DataFrame(senders),'TxAddr_edgelist.csv':pl.DataFrame(receivers),
            'wallets_classes.csv':pl.DataFrame({'address':list(labels),'class':list(labels.values())})}
    for name,table in tables.items():table.write_csv(data/name)
    for name in ('txs_classes.csv','wallets_features_classes_combined.csv','AddrAddr_edgelist.csv','txs_edgelist.csv'):
        (data/name).write_text('unused\n')
    files={p.name:{'bytes':p.stat().st_size,'sha256':file_hash(p)} for p in data.glob('*.csv')}
    evidence=root/'docs/data';(evidence/'inv-001').mkdir(parents=True)
    (evidence/'source-manifest.json').write_text(json.dumps({'files':files}))
    (evidence/'inv-001/report.json').write_text('{}')
    (evidence/'transaction-input-check.json').write_text('{}')
    (root/'requirements').mkdir()
    (root/'requirements/mac-cpu.lock').write_text('# fixture-only lock identity\n')
    value=defaults('smoke');value['paths'].update(project_root='.',data_root='raw')
    path=root/'config.yaml';path.write_text(yaml.safe_dump(value));config=load_config(path)
    result=prepare(config)
    return config,Path(result['prepared'])


def test_fit_does_not_materialize_global_or_test_targets(prepared_fixture,monkeypatch):
    config,prepared=prepared_fixture
    original_read=pl.read_parquet
    def guarded_read(path,*args,**kwargs):
        if Path(path).name=='global.parquet':
            raise AssertionError('fit materialized global targets, including test addresses, before training')
        return original_read(path,*args,**kwargs)
    monkeypatch.setattr(pl,'read_parquet',guarded_read)
    # Stop normally after preflight, before expensive fitting. The access guard must
    # remain silent; valid entry into the model stage reaches this sentinel instead.
    class ReachedTraining(Exception):pass
    def reached(*args,**kwargs):raise ReachedTraining()
    monkeypatch.setattr('hgcl.training.runner.pretrain',reached)
    with pytest.raises(ReachedTraining):fit(config,prepared,'boundary-fixture')


def test_supervision_api_refuses_test_partition():
    with pytest.raises(ValueError,match='cannot request test'):
        permitted_targets(Path('not-read'),{},defaults('smoke'),'test')


def test_prediction_keys_must_match():
    first=pl.DataFrame({'address':['a'],'step':[35],'score':[.1]})
    wrong=pl.DataFrame({'address':['b'],'step':[35],'score':[.2]})
    with pytest.raises(ValueError,match='populations'):aligned({'rf':first,'hgcl':wrong})
    with pytest.raises(ValueError,match='Duplicate'):aligned({'rf':first.vstack(first)})


def test_empty_or_single_class_stratum_is_explicit():
    empty=metrics([],[],.5);single=metrics([1,1],[.2,.8],.5)
    assert empty['support']==0 and empty['ap'] is None and 'ap' in empty['undefined']
    assert single['ap'] is None and single['mcc'] is None


def test_dry_run_never_materializes_targets(prepared_fixture,monkeypatch):
    from hgcl.config import Config
    from hgcl.matrix import dry_run
    config,prepared=prepared_fixture
    lab=defaults('lab');lab['paths']=config.values['paths']
    # Enumeration has no preparation dependency; full integrity is separately optional.
    def forbidden(*args,**kwargs):raise AssertionError('Dry-run read parquet targets')
    monkeypatch.setattr(pl,'read_parquet',forbidden)
    result=dry_run(Config(json.dumps(lab,sort_keys=True)))
    assert len(result['evaluations'])==100 and not result['test_labels_materialized']
    assert not result['data_integrity_checked']


def test_partition_mask_cannot_backfill_labels(prepared_fixture):
    config,prepared=prepared_fixture
    record=json.loads((prepared/'targets/budgets.json').read_text())[0]
    frame=permitted_targets(prepared,record,config.values,'train')
    assert set(frame['step'])=={1,2}
    assert 'licit-29' not in frame['address'].to_list()
    assert 'licit-35' not in frame['address'].to_list()


def test_complete_fit_access_and_public_resume(prepared_fixture,monkeypatch):
    import torch
    import hgcl.training.runner as runner
    config,prepared=prepared_fixture
    original=runner.pretrain
    original_read=pl.read_parquet
    fitting=True
    def guarded(path,*args,**kwargs):
        if fitting and Path(path).parent.name=='targets':
            raise AssertionError('Eager target materialization during fitting')
        return original_read(path,*args,**kwargs)
    monkeypatch.setattr(pl,'read_parquet',guarded)
    # Interrupt after the first SSL epoch, then exercise the public run lifecycle.
    def interrupt(*args,**kwargs):return original(*args,**kwargs,stop_after_epoch=1)
    monkeypatch.setattr(runner,'pretrain',interrupt)
    with pytest.raises(InterruptedError):runner.fit(config,prepared,'resume-fixture')
    run=config.artifacts_root/'runs/resume-fixture'
    assert json.loads((run/'status.json').read_text())['state']=='interrupted'
    monkeypatch.setattr(runner,'pretrain',original)
    # Reject configuration changes without mutating the interrupted run.
    saved=(run/'config.json').read_text();state_before=(run/'status.json').read_bytes()
    altered=json.loads(saved);altered['ssl']['epochs']+=1
    (run/'config.json').write_text(json.dumps(altered))
    with pytest.raises(ValueError,match='Configuration changed'):runner.resume(run)
    assert (run/'status.json').read_bytes()==state_before
    (run/'config.json').write_text(saved)
    original_evaluate=runner.evaluate
    def evaluate_after_fit(*args,**kwargs):
        nonlocal fitting
        fitting=False
        return original_evaluate(*args,**kwargs)
    monkeypatch.setattr(runner,'evaluate',evaluate_after_fit)
    result=runner.resume(run)
    assert result['status']=='complete'
    assert len(list((run/'attempts').iterdir()))==1
    for method in result['strata'].values():
        assert method['overall']['support']==2
        assert method['seen']['support']==1 and method['unseen']['support']==1
    before=(run/'status.json').read_bytes()
    with pytest.raises(ValueError,match='Cannot refit'):runner.resume(run)
    assert (run/'status.json').read_bytes()==before
