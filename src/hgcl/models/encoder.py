"""Synchronous heterogeneous relation-sum encoder with one root per node type."""
import torch
from torch import nn

TYPES=('address','transaction')
RELATIONS={'sender':('address','transaction'),'sender_reverse':('transaction','address'),
           'receiver':('transaction','address'),'receiver_reverse':('address','transaction')}


class Layer(nn.Module):
    def __init__(self,hidden):
        super().__init__()
        self.messages=nn.ModuleDict({r:nn.Linear(hidden,hidden,bias=False) for r in RELATIONS})
        self.epsilon=nn.ParameterDict({t:nn.Parameter(torch.zeros(())) for t in TYPES})
        self.update=nn.ModuleDict({t:nn.Sequential(nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,hidden),nn.LayerNorm(hidden)) for t in TYPES})

    def forward(self,h,edges):
        summed={t:(1+self.epsilon[t])*h[t] for t in TYPES}
        for name,(src,dst) in RELATIONS.items():
            edge=edges[name]
            messages=self.messages[name](h[src][edge[0]])
            summed[dst]=summed[dst].index_add(0,edge[1],messages)
        return {t:self.update[t](summed[t]) for t in TYPES}


class Encoder(nn.Module):
    def __init__(self,address_dim,transaction_dim,hidden=128,layers=2):
        super().__init__();self.hidden=hidden
        self.projections=nn.ModuleDict({'address':nn.Linear(address_dim,hidden),'transaction':nn.Linear(transaction_dim,hidden)})
        self.layers=nn.ModuleList([Layer(hidden) for _ in range(layers)])

    def forward(self,x,edges):
        h={t:self.projections[t](x[t]) for t in TYPES}
        for layer in self.layers:h=layer(h,edges)
        return h
