import json
from pathlib import Path
import pytest
import yaml
from hgcl.config import defaults, load_config
from hgcl.provenance import atomic_json, status, source_identity


def test_strict_config_and_precedence(tmp_path, monkeypatch):
    value=defaults('smoke'); value['paths']['project_root']='.'
    path=tmp_path/'smoke.yaml';path.write_text(yaml.safe_dump(value))
    monkeypatch.setenv('ELLIPTIC_DATA_ROOT','environment-data')
    first=load_config(path);assert first.data_root==tmp_path/'environment-data'
    assert load_config(path,data_root='explicit-data').data_root==tmp_path/'explicit-data'
    monkeypatch.chdir(tmp_path.parent);assert load_config(path).hash==first.hash
    value['split']['train']=[1,35];path.write_text(yaml.safe_dump(value))
    with pytest.raises(ValueError):load_config(path)
    value=defaults('smoke');value['encoder']['typo']=1;path.write_text(yaml.safe_dump(value))
    with pytest.raises(ValueError):load_config(path)


def test_roots_cannot_overlap(tmp_path):
    value=defaults('smoke');value['paths'].update(project_root='.',data_root='data',artifacts_root='data/output')
    path=tmp_path/'smoke.yaml';path.write_text(yaml.safe_dump(value))
    with pytest.raises(ValueError):load_config(path)


def test_atomic_state_and_unborn_identity(tmp_path):
    path=tmp_path/'status.json';status(path,'planned');status(path,'validated');status(path,'complete')
    with pytest.raises(ValueError):status(path,'fitting')
    assert json.loads(path.read_text())['state']=='complete'
    with pytest.raises(ValueError):atomic_json(path, {'bad':float('nan')})
    assert json.loads(path.read_text())['state']=='complete'
    assert source_identity(tmp_path)['git_commit'] is None


def test_source_content_hash_tracks_code_not_outputs(tmp_path):
    folder=tmp_path/'src';folder.mkdir();code=folder/'example.py';code.write_text('x = 1\n')
    first=source_identity(tmp_path)['sha256']
    (tmp_path/'artifacts').mkdir();(tmp_path/'artifacts/output.json').write_text('{}')
    assert source_identity(tmp_path)['sha256']==first
    code.write_text('x = 2\n');assert source_identity(tmp_path)['sha256']!=first


def test_wrong_types_and_mutation_are_rejected(tmp_path):
    value=defaults('smoke');path=tmp_path/'smoke.yaml';path.write_text(yaml.safe_dump(value))
    cfg=load_config(path);cfg.values['split']['train'].append(49)
    assert cfg.values['split']['train']==[1,2]
    value['encoder']['layers']=True;path.write_text(yaml.safe_dump(value))
    with pytest.raises(ValueError):load_config(path)


def test_invalid_yaml_reports_configuration_error(tmp_path):
    path=tmp_path/'bad.yaml';path.write_text('paths: [')
    with pytest.raises(ValueError,match='Invalid YAML'):load_config(path)
