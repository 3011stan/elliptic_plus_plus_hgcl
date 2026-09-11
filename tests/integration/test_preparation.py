"""Real-source repeat preparation; opt-in to avoid expensive default test runs."""
import os
from pathlib import Path
import pytest
from hgcl.config import load_config
from hgcl.pipeline import prepare,validate


@pytest.mark.original_data
@pytest.mark.skipif(os.environ.get('HGCL_RUN_ORIGINAL_TESTS')!='1',reason='Set HGCL_RUN_ORIGINAL_TESTS=1 for original-data preparation')
def test_original_preparation_twice(project_root):
    config=load_config(project_root/'configs/smoke.yaml')
    first=prepare(config);second=prepare(config)
    assert first['attempt']!=second['attempt']
    assert first['preparation_hash']==second['preparation_hash']
    assert first['payload_hash']==second['payload_hash']
    assert validate(Path(first['prepared']),config)['status']=='PASS'
