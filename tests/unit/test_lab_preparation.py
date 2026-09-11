import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import yaml
from hgcl.config import defaults,load_config
from hgcl.data.download import acquire
from hgcl.provenance import file_hash,source_identity
from hgcl.environment import operation_probe


def make_download_fixture(tmp_path):
    files={}
    for i in range(9):
        path=tmp_path/f'source-{i}.csv';path.write_text(f'fixture {i}\n')
        files[path.name]={'bytes':path.stat().st_size,'sha256':file_hash(path)}
    manifest=tmp_path/'docs/data/source-manifest.json';manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({'files':files}))
    values=defaults('lab');values['paths'].update(project_root='.',data_root='raw')
    path=tmp_path/'config.yaml';path.write_text(yaml.safe_dump(values))
    class Client:
        calls=0
        def download_folder(self,**kwargs):
            assert kwargs['skip_download'] and not kwargs['use_cookies']
            return [SimpleNamespace(id=n,path='folder/'+n) for n in files]
        def download(self,id,output,**kwargs):
            self.calls+=1;Path(output).write_bytes((tmp_path/id).read_bytes());return output
    return load_config(path),Client()


def test_download_verifies_and_reuses_without_network(tmp_path):
    config,client=make_download_fixture(tmp_path)
    assert acquire(config,client)['status']=='PASS' and client.calls==9
    assert acquire(config,client)['downloaded']==[] and client.calls==9
    first=config.data_root/'source-0.csv';first.write_text('changed')
    with pytest.raises(ValueError,match='Existing original differs'):acquire(config,client)
    assert first.read_text()=='changed' and client.calls==9


def test_download_rejects_bad_hash_before_publication(tmp_path):
    config,client=make_download_fixture(tmp_path)
    def corrupt(id,output,**kwargs):Path(output).write_text('bad');return output
    client.download=corrupt
    with pytest.raises(ValueError,match='checksum'):acquire(config,client)
    assert not list(config.data_root.glob('*.csv'))


def test_download_rejects_missing_listing(tmp_path):
    config,client=make_download_fixture(tmp_path)
    client.download_folder=lambda **kwargs:[]
    with pytest.raises(ValueError,match='exactly one'):acquire(config,client)
    assert client.calls==0


def test_gpu_smoke_preserves_small_temporal_recut(tmp_path):
    v=defaults('smoke-gpu');v['paths']['project_root']='.'
    p=tmp_path/'config.yaml';p.write_text(yaml.safe_dump(v));config=load_config(p)
    assert config.values['split']==defaults('smoke')['split']
    assert v['resources']['recut_transactions']==256
    assert v['resources']['device']=='cuda' and v['resources']['loading']=='neighbor'
    assert v['resources']['timeout_seconds']==600
    assert defaults('smoke')['resources']['device']=='cpu'


def test_operation_probe_cpu_exercises_both_anchor_types():
    report=operation_probe(defaults('smoke')['resources'],hidden=8)
    assert set(report)=={'address','transaction'}
    assert all(r['anchors']==[0,1] and r['directed_edges']>0 for r in report.values())


def test_nix_and_launchers_participate_in_source_identity(tmp_path):
    first=source_identity(tmp_path)['sha256']
    (tmp_path/'flake.lock').write_text('{}')
    second=source_identity(tmp_path)['sha256'];assert second!=first
    (tmp_path/'scripts').mkdir();(tmp_path/'scripts/run.py').write_text('x=1')
    assert source_identity(tmp_path)['sha256']!=second
