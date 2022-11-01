import torch
from torch import nn
import math

def position_encoding(d_model, seq_len):
    pos = torch.arange(0, seq_len).reshape(-1, 1)
    dim = torch.arange(0, d_model, 2).reshape(1, -1).float()
    div = pos / (10000 ** (dim / d_model))

    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(div)
    pe[:, 1::2] = torch.cos(div)
    pe.unsqueeze(0)

    return pe


class TokenEmbeddings(nn.Module):

    def __init__(self, d_model, vocab_size):
        super(TokenEmbeddings, self).__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        out = self.emb(x) * math.sqrt(self.d_model)
        return out


class TransformerEmbeddings(nn.Module):

    def __init__(self, d_model, vocab_size, drop_rate):
        super(TransformerEmbeddings, self).__init__()
        self.d_model = d_model
        self.tok_emb = TokenEmbeddings(d_model, vocab_size)
        self.dropout = nn.Dropout(drop_rate)
  
    def forward(self, x):
        tok_emb = self.tok_emb(x)
        pos_enc = position_encoding(self.d_model, x.size(1))
        out = self.dropout(tok_emb + pos_enc)
        return out
