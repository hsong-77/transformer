import torch
from torch import nn

from encoder import Encoder
from decoder import Decoder

class Transformer(nn.Module):

    def __init__(
        self, 
        src_vocab_size,
        tgt_vocab_size,
        src_pad_idx,
        tgt_pad_idx,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_model=512,
        n_head=8,
        n_hidden=2048,
        drop_rate=0.1
    ):
        super(Transformer, self).__init__()
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.encoder = Encoder(
            num_encoder_layers,
            d_model,
            n_head,
            n_hidden,
            src_vocab_size,
            drop_rate,
        )
        self.decoder = Decoder(
            num_decoder_layers,
            d_model,
            n_head,
            n_hidden,
            tgt_vocab_size,
            drop_rate,
        )
    
    def forward(self, src, tgt):
        src_mask = self.src_mask(src, self.src_pad_idx)
        tgt_mask = self.tgt_mask(tgt, self.tgt_pad_idx)

        memory = self.encoder(src, src_mask)
        output = self.decoder(tgt, memory, src_mask, tgt_mask)
        return output

    def src_mask(self, src, pad):
        return self.padding_mask(src, pad)

    def tgt_mask(self, tgt, pad):
        return self.padding_mask(tgt, pad) & self.subsequent_mask(tgt.size(1))

    def padding_mask(self, x, pad):
        return (x != pad).unsqueeze(-2)

    def subsequent_mask(self, size):
        attn_shape = (1, size, size)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal = 1).type(torch.uint8)
        return subsequent_mask == 0

#sanity check
# src = torch.rand(64, 32).type(torch.int)
# tgt = torch.rand(64, 32).type(torch.int)
# out = Transformer(128, 128, 1, 1)(src, tgt)
# print(out.shape)
