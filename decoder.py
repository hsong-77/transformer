import torch
from torch import nn

from embedding import TransformerEmbeddings
from multi_head_attention import MultiHeadedAttention
from layers import LayerNorm, PositionwiseFeedForward

class DecoderLayer(nn.Module):

    def __init__(self, d_model, n_head, d_hidden, drop_rate):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadedAttention(d_model, n_head)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(drop_rate)

        self.enc_dec_attention = MultiHeadedAttention(d_model, n_head)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(drop_rate)

        self.ffn = PositionwiseFeedForward(d_model, d_hidden, drop_rate)
        self.norm3 = LayerNorm(d_model)
        self.dropout3 = nn.Dropout(drop_rate)

    def forward(self, tgt, memory, src_mask, tgt_mask):
        _x = tgt
        x = self.self_attention(tgt, tgt, tgt, tgt_mask);
        x = self.norm1(_x + x)
        x = self.dropout1(x)

        _x = x
        x = self.enc_dec_attention(x, memory, memory, src_mask)
        x = self.norm2(_x + x)
        x = self.dropout2(x)

        _x = x
        x = self.ffn(x)
        x = self.norm3(_x + x)
        x = self.dropout3(x)

        return x


class Decoder(nn.Module):

    def __init__(self, n_layer, d_model, n_head, d_hidden, vocab_size, drop_rate):
        super(Decoder, self).__init__()
        self.emb = TransformerEmbeddings(d_model, vocab_size, drop_rate)
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, n_head, d_hidden, drop_rate) for _ in range(n_layer)])
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, memory, src_mask, tgt_mask):
        tgt = self.emb(tgt)
        for layer in self.decoder_layers:
            tgt = layer(tgt, memory, src_mask, tgt_mask)

        out = self.linear(tgt)
        return out
