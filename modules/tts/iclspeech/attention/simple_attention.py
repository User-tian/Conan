import torch
import torch.nn as nn
import torch.nn.functional as F


def split_heads(x, num_heads):
    """ Split heads
    :param x: A tensor with shape [batch, length, channels]
    :param num_heads: An integer
    :returns: A tensor with shape [batch, heads, length, channels / heads]
    """
    assert x.shape[-1] % num_heads == 0, str(x.shape)
    return x.reshape(x.shape[:-1] + (num_heads, x.shape[-1] // num_heads)).permute(0, 2, 1, 3)


def combine_heads(x):
    """ Combine heads
    :param x: A tensor with shape [batch, heads, length, channels]
    :returns: A tensor with shape [batch, length, heads * channels]
    """
    x = x.permute([0, 2, 1, 3])
    return x.reshape(x.shape[:-2] + (x.shape[-1] * x.shape[-2],))


class SimpleAttention(nn.Module):
    def __init__(self, query_size=192, key_size=192, value_size=192, num_heads=1, dropout_rate=0.05):
        super(SimpleAttention, self).__init__()
        self.q_transform = nn.Linear(query_size, query_size, bias=False)
        self.k_transform = nn.Linear(key_size, query_size, bias=False)
        self.v_transform = nn.Linear(value_size, query_size, bias=False)
        self.output_transform = nn.Linear(query_size, query_size, bias=False)
        self.query_size = query_size
        self.key_size = key_size
        self.value_size = value_size
        self.num_heads = num_heads
        self.attn_dropout = nn.Dropout(dropout_rate)

    def forward(self, query, key, value, attn_mask, bias=None):
        q = self.q_transform(query)
        k = self.k_transform(key)
        v = self.v_transform(value) # [1, B*T, C]
        if len(attn_mask.shape)==3:
            attn_mask = attn_mask[:, None, :, :].repeat(1, self.num_heads, 1, 1)
        elif len(attn_mask.shape)==2:
            attn_mask = attn_mask[:, None, :, None].repeat(1, self.num_heads, 1, q.size(1))

        q = split_heads(q, self.num_heads) # [1, num_heads, B*T, C//num_heads]
        k = split_heads(k, self.num_heads) # [1, num_heads, B*T, C//num_heads]
        v = split_heads(v, self.num_heads) # [1, num_heads, B*T, C//num_heads]

        logits = torch.matmul(q, k.transpose(2, 3)) # [1, num_heads, B*T, B*T]
        if bias is not None:
            logits += bias
        if logits.dtype == torch.float16:
            logits[attn_mask==0] = -5e4
        else:
            logits[attn_mask==0] = -1e9
        weights = F.softmax(logits, dim=-1)
        weights = self.attn_dropout(weights)
        out = torch.matmul(weights, v)

        out = combine_heads(out)
        out = self.output_transform(out)
        return out, weights.mean(dim=1)