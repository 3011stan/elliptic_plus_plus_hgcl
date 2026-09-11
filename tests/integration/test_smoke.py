"""Original-data end-to-end engineering acceptance, explicitly opt-in."""
import json
import os
from pathlib import Path
import numpy as np
import polars as pl
import pytest
from hgcl.config import load_config
from hgcl.pipeline import prepare
from hgcl.training.runner import smoke
from hgcl.provenance import file_hash


@pytest.mark.original_data
@pytest.mark.skipif(os.environ.get('HGCL_RUN_SMOKE')!='1',reason='Set HGCL_RUN_SMOKE=1 for real training smoke')
def test_original_data_smoke(project_root):
    config=load_config(project_root/'configs/smoke.yaml')
    prepared=prepare(config)
    result=smoke(config,Path(prepared['prepared']),os.environ.get('HGCL_SMOKE_RUN_ID','smoke-001'))
    assert result['status']=='complete' and result['reload_verified']
    assert result['fitting_selection_seconds']<=600
    run=Path(result['run']);report=json.loads((run/'training-report.json').read_text())
    assert report['ssl']['encoder_changed'] and report['ssl']['updates']>0
    for method in ('graph_supervised','hgcl'):
        assert all(candidate['encoder_changed'] and candidate['head_changed'] for candidate in report[method])
        assert all(np.isfinite(epoch['mean_loss']) for candidate in report[method] for epoch in candidate['history'])
    frozen=json.loads((run/'frozen.json').read_text())
    assert file_hash(run/'ssl/ssl-final.pt')==frozen['files']['ssl/ssl-final.pt']
    predictions=pl.read_parquet(run/'predictions.parquet')
    assert predictions.select('address','step','method').unique().height==predictions.height
    assert set(predictions['method'])=={'rf','graph_supervised','hgcl','fusion'}
    assert predictions['score'].is_finite().all()
    assert len(set(predictions.group_by('method').len()['len']))==1
