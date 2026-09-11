import torch
import pytest
from hgcl.models.encoder import Encoder
from hgcl.training.inference import infer


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
