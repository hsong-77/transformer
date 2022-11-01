import torch
from torch import nn

from embedding import TransformerEmbeddings
from multi_head_attention import MultiHeadedAttention
from layers import LayerNorm, PositionwiseFeedForward

class EncoderLayer(nn.Module):

    def __init__(self, d_model, n_head, d_hidden, drop_rate):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadedAttention(d_model, n_head)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(drop_rate)

        self.ffn = PositionwiseFeedForward(d_model, d_hidden, drop_rate)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(drop_rate)

    def forward(self, x, mask):
        _x = x
        x = self.attention(x, x, x, mask)
        x = self.norm1(_x + x)
        x = self.dropout1(x)

        _x = x
        x = self.ffn(x)
        x = self.norm2(_x + x)
        x = self.dropout2(x)

        return x


class Encoder(nn.Module):

    def __init__(self, n_layer, d_model, n_head, d_hidden, vocab_size, drop_rate):
        super(Encoder, self).__init__()
        self.emb = TransformerEmbeddings(d_model, vocab_size, drop_rate)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, n_head, d_hidden, drop_rate) for _ in range(n_layer)])

    def forward(self, x, mask):
        x = self.emb(x)
        for layer in self.encoder_layers:
            x = layer(x, mask)  
    
        return x
