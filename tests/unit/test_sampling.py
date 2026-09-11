import numpy as np
import torch
import pytest
from torch_geometric.data import HeteroData
from hgcl.data.sampling import Graph,full_batch,paired_sample
from hgcl.models.encoder import RELATIONS


def test_sample_reconstructs_reverse_pair_and_canonical_anchors():
    x={'address':torch.randn(3,4),'transaction':torch.randn(2,4)}
    edges={'sender':torch.tensor([[0,1,2],[0,0,1]]),'receiver':torch.tensor([[0,1],[1,2]])}
    edges.update({r+'_reverse':edges[r].flip(0) for r in ('sender','receiver')})
    graph=Graph(1,x,edges,{}, {},np.empty((3,0)))
    sampled=HeteroData()
    for t in x:sampled[t].x=x[t];sampled[t].n_id=torch.arange(len(x[t]))
    sampled['address'].batch_size=2
    for r,(src,dst) in RELATIONS.items():
        sampled[src,r,dst].e_id=torch.tensor([0]) if r in ('sender','receiver_reverse') else torch.empty(0,dtype=torch.long)
    resources={'max_nodes':10,'max_edges':10}
    xx,ee,local,global_ids=paired_sample(graph,sampled,'address',resources)
    assert torch.equal(ee['sender_reverse'],ee['sender'].flip(0))
    assert torch.equal(ee['receiver_reverse'],ee['receiver'].flip(0))
    assert torch.equal(local,global_ids)
    assert ee['sender'].shape[1]==1
    with pytest.raises(RuntimeError):full_batch(graph,'address',[0],{'max_nodes':1,'max_edges':10})
