import torch
from torch import nn


class Classifier(nn.Module):
    def __init__(self,encoder,hidden,seed):
        super().__init__();self.encoder=encoder
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+10000)
            self.head=nn.Linear(hidden,1)

    def forward(self,x,edges):
        return self.head(self.encoder(x,edges)['address']).squeeze(-1)
