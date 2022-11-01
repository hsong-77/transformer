import torch
from torch import nn
import math

def scaled_dot_product_attention(query, key, value, mask=None, eps=1e-11):
    d_key = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_key) #[batch_size, n_head, seq_len, seq_len]
    if mask is not None:
        scores = scores.masked_fill_(mask==0, eps)
    scores = scores.softmax(dim=-1)
  
    out = torch.matmul(scores, value)
    return out, scores


class MultiHeadedAttention(nn.Module):

    def __init__(self, d_model, n_head):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_model = d_model
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.concat_linear = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        query = self.w_q(query)
        key = self.w_k(key)
        value = self.w_v(value)

        # same mask applied to all n_head heads.
        if mask is not None:
            mask = mask.unsqueeze(1)

        batch_size = query.size(0)
        d_key = self.d_model // self.n_head
    
        query = query.reshape(batch_size, -1, self.n_head, d_key).transpose(1, 2)
        key = key.reshape(batch_size, -1, self.n_head, d_key).transpose(1, 2)
        value = value.reshape(batch_size, -1, self.n_head, d_key).transpose(1, 2)

        out, attn = scaled_dot_product_attention(query, key, value, mask=mask)

        out = out.transpose(1, 2).reshape(batch_size, -1, self.n_head * d_key)
        out = self.concat_linear(out)
        return out
