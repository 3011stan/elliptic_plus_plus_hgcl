import torch
import pytest
from hgcl.training.checkpoint import save,load
from hgcl.data.sampling import anchor_batches


def test_epoch_replay_and_dependency_rejection(tmp_path):
    torch.manual_seed(4);m=torch.nn.Linear(2,1);o=torch.optim.Adam(m.parameters());g=torch.Generator().manual_seed(9)
    def update():
        o.zero_grad();m(torch.rand(3,2,generator=g)).square().mean().backward();o.step()
    update();path=tmp_path/'epoch.pt';save(path,m,o,1,{'source':'a'},generators={'sampler':g})
    update();expected={k:v.clone() for k,v in m.state_dict().items()}
    load(path,m,o,{'source':'a'},generators={'sampler':g});update()
    for key in expected:torch.testing.assert_close(m.state_dict()[key],expected[key],rtol=0,atol=0)
    with pytest.raises(ValueError):load(path,m,o,{'source':'b'},generators={'sampler':g})
    with pytest.raises(ValueError):save(path,m,o,2,{},immutable=True)


def test_tail_has_no_fabricated_or_lost_anchors():
    chunks=anchor_batches(torch.arange(65),32,shuffle=False,min_size=2)
    assert [len(x) for x in chunks]==[32,33]
    assert torch.equal(torch.cat(chunks),torch.arange(65))
    assert anchor_batches([0],32,min_size=2)==[]


@pytest.mark.parametrize('count',[0,1,2,31,32,33,63,64,65,97])
def test_anchor_coverage_across_boundaries(count):
    chunks=anchor_batches(torch.arange(count),32,shuffle=False,min_size=2)
    if count<2:
        assert chunks==[]
    else:
        assert all(len(c)>=2 for c in chunks)
        assert torch.equal(torch.cat(chunks),torch.arange(count))
