import json
import polars as pl
import pytest
from hgcl.data.ingest import load_inputs,load_targets,verify_sources
from hgcl.data.schema import TX_COLUMNS
from hgcl.provenance import file_hash


def source_fixture(root):
    pl.DataFrame({'txId':['x','y'],'Time step':['1','2'],**{c:['0',''] for c in TX_COLUMNS}}).write_csv(root/'txs_features.csv')
    pl.DataFrame({'address':['a','a','b'],'Time step':['1','1','2']}).write_csv(root/'wallets_features.csv')
    pl.DataFrame({'input_address':['a','a'],'txId':['x','x']}).write_csv(root/'AddrTx_edgelist.csv')
    pl.DataFrame({'txId':['x','y'],'output_address':['a','b']}).write_csv(root/'TxAddr_edgelist.csv')
    pl.DataFrame({'address':['a','b'],'class':['1','3']}).write_csv(root/'wallets_classes.csv')


def test_projected_ingestion_and_separate_targets(tmp_path):
    source_fixture(tmp_path);data=load_inputs(tmp_path)
    assert data.wallets.height==2 and data.diagnostics['duplicate_wallet_rows']==1
    assert data.diagnostics['duplicate_edges']['sender']==1
    assert 'class' not in data.wallets.columns and 'class' not in data.transactions.columns
    assert data.transactions['fees'].to_list()==[0.,None]
    targets=load_targets(tmp_path,data.wallets);assert targets['class'].to_list()==[1,3]
    pl.DataFrame({'input_address':['outside'],'txId':['x']}).write_csv(tmp_path/'AddrTx_edgelist.csv')
    with pytest.raises(ValueError,match='unsupported'):load_inputs(tmp_path)


def test_unsupported_wallet_occurrence(tmp_path):
    source_fixture(tmp_path)
    pl.DataFrame({'address':['a','b','unconnected'],'Time step':['1','2','1']}).write_csv(tmp_path/'wallets_features.csv')
    with pytest.raises(ValueError,match='wallet coverage'):load_inputs(tmp_path)


def test_hash_mismatch_is_not_silently_accepted(tmp_path):
    files={}
    for i in range(9):
        path=tmp_path/f'{i}.csv';path.write_text('id\n1\n');files[path.name]={'bytes':path.stat().st_size,'sha256':file_hash(path)}
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps({'files':files}))
    assert len(verify_sources(tmp_path,manifest))==9
    (tmp_path/'2.csv').write_text('id\n2\n')
    with pytest.raises(ValueError,match='2.csv'):verify_sources(tmp_path,manifest)


def test_missing_header_reports_source(tmp_path):
    from hgcl.data.ingest import read_columns
    path=tmp_path/'wrong.csv';path.write_text('wrong\n1\n')
    with pytest.raises(ValueError,match='wrong.csv'):read_columns(path,['txId'])
