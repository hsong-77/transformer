import torch
from torch import nn

class LayerNorm(nn.Module):
  
    def __init__(self, d_model, eps=1e-11):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        out = (x - mean) / (std + self.eps)
        out = self.gamma * out + self.beta
        return out

class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, d_hidden, drop_rate):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_hidden)
        self.w_2 = nn.Linear(d_hidden, d_model)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        out = self.w_2(self.dropout(self.w_1(x).relu()))
        return out
