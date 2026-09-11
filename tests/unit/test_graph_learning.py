import torch
import pytest
from hgcl.models.encoder import Encoder, RELATIONS
from hgcl.models.heads import Classifier
from hgcl.models.augmentations import augment
from hgcl.models.contrastive import contrastive_loss


def graph():
    x={'address':torch.randn(5,6),'transaction':torch.randn(3,4)}
    s=torch.tensor([[0,1,2,3,4],[0,1,1,2,2]])
    r=torch.tensor([[0,1,2],[1,2,4]])
    return x,{'sender':s,'sender_reverse':s.flip(0),'receiver':r,'receiver_reverse':r.flip(0)}


def test_encoder_and_head_receive_gradients():
    torch.manual_seed(11);x,e=graph();model=Classifier(Encoder(6,4,8),8,seed=11)
    before={n:p.detach().clone() for n,p in model.named_parameters()}
    opt=torch.optim.Adam(model.parameters(),lr=.01)
    loss=model(x,e).square().mean();loss.backward();opt.step()
    assert any(not torch.equal(p,before[n]) for n,p in model.named_parameters() if n.startswith('encoder'))
    assert any(not torch.equal(p,before[n]) for n,p in model.named_parameters() if n.startswith('head'))
    paired=Classifier(Encoder(6,4,8),8,seed=11)
    fresh=Classifier(Encoder(6,4,8),8,seed=11)
    assert torch.equal(paired.head.weight,fresh.head.weight)


def test_pair_dropout_and_feature_masks():
    x,e=graph();x['transaction'][:,-1]=1
    original={k:v.clone() for k,v in x.items()}
    a,b=augment(x,e,{'address':[0,1],'transaction':[0,1,2]},1.,.5,torch.Generator().manual_seed(11))
    assert (a['address'][:,:2]==0).all()
    assert torch.equal(a['transaction'][:,-1],x['transaction'][:,-1])
    assert torch.equal(b['sender'].flip(0),b['sender_reverse'])
    assert torch.equal(b['receiver'].flip(0),b['receiver_reverse'])
    assert all(torch.equal(x[k],original[k]) for k in x)


def test_contrastive_alignment_and_negatives():
    z=torch.eye(4,requires_grad=True);ids=torch.arange(4)
    correct=contrastive_loss(z,z,ids,ids,.2)
    wrong=contrastive_loss(z,z.flip(0),ids,ids,.2)
    assert correct<wrong
    correct.backward();assert z.grad is not None and torch.isfinite(z.grad).all()
    with pytest.raises(ValueError):contrastive_loss(z,z,ids,ids.flip(0),.2)
    with pytest.raises(ValueError):contrastive_loss(z[:1],z[:1],ids[:1],ids[:1],.2)


def test_permutation_equivariance():
    torch.manual_seed(3);x,e=graph();m=Encoder(6,4,8).eval();out=m(x,e)
    p=torch.tensor([4,0,2,1,3]);inverse=torch.argsort(p)
    changed={k:v.clone() for k,v in e.items()}
    for name,(src,dst) in RELATIONS.items():
        if src=='address':changed[name][0]=inverse[changed[name][0]]
        if dst=='address':changed[name][1]=inverse[changed[name][1]]
    perm=m({'address':x['address'][p],'transaction':x['transaction']},changed)
    torch.testing.assert_close(out['address'][p],perm['address'],atol=1e-6,rtol=1e-5)
