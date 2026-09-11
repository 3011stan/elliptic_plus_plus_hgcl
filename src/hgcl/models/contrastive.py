"""NT-Xent over aligned same-type, same-snapshot anchors only."""
import torch
from torch import nn
from torch.nn import functional as F


def contrastive_loss(z1,z2,ids1,ids2,temperature):
    n=len(z1)
    if n<2 or z1.shape!=z2.shape or len(ids1)!=n or not torch.equal(ids1,ids2) or len(torch.unique(ids1))!=n:
        raise ValueError('Contrastive anchors must be aligned, unique and number at least two')
    if temperature<=0:raise ValueError('Invalid contrastive temperature')
    z=F.normalize(torch.cat([z1,z2]),dim=1)
    scores=z@z.T/temperature
    scores=scores.masked_fill(torch.eye(2*n,dtype=torch.bool,device=z.device),float('-inf'))
    target=(torch.arange(2*n,device=z.device)+n)%(2*n)
    return F.cross_entropy(scores,target)


class Projection(nn.Module):
    def __init__(self,hidden,dim):
        super().__init__();self.heads=nn.ModuleDict({t:nn.Sequential(nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,dim)) for t in ('address','transaction')})
    def forward(self,kind,h):return self.heads[kind](h)
