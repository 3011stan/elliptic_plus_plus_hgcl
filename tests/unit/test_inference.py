import torch
import polars as pl
import pytest
from hgcl.models.encoder import Encoder
from hgcl.training.inference import infer
from hgcl.training.runner import compare_reloaded_predictions


@pytest.mark.parametrize('node_chunk,edge_chunk',[(1,1),(3,2),(4096,100000)])
def test_exact_inference_matches_full(node_chunk,edge_chunk):
    torch.manual_seed(11)
    x={'address':torch.randn(7,6),'transaction':torch.randn(4,4)}
    s=torch.tensor([[0,1,2,3,4,5,6],[0,0,1,1,2,3,3]])
    r=torch.tensor([[0,1,2,3],[1,3,5,6]])
    e={'sender':s,'sender_reverse':s.flip(0),'receiver':r,'receiver_reverse':r.flip(0)}
    model=Encoder(6,4,8);model.train()
    with torch.no_grad():expected=model(x,e)
    actual=infer(model,x,e,node_chunk,edge_chunk)
    assert model.training
    for t in x:torch.testing.assert_close(expected[t],actual[t],atol=2e-6,rtol=1e-5)


def prediction_frame(values):
    return pl.DataFrame({'rf':values,'graph_supervised':values,'hgcl':values,'fusion':values})


def test_reload_comparison_accepts_float32_noise_without_decision_changes():
    first=prediction_frame([.1,.5001,.9]);second=prediction_frame([.100001,.500101,.900001])
    choices={name:{'threshold':.5} for name in first.columns}
    report=compare_reloaded_predictions(first,second,choices,first.columns)
    assert all(row['within_tolerance'] and row['decision_mismatches']==0 for row in report.values())


def test_reload_comparison_rejects_tolerance_or_decision_instability():
    choices={name:{'threshold':.5} for name in ('rf','graph_supervised','hgcl','fusion')}
    with pytest.raises(RuntimeError,match='max_abs'):
        compare_reloaded_predictions(prediction_frame([.1,.9]),prediction_frame([.1,.8]),choices,choices)
    with pytest.raises(RuntimeError,match='decision_mismatches=1'):
        compare_reloaded_predictions(prediction_frame([.4999999,.9]),prediction_frame([.5000001,.9]),choices,choices)
