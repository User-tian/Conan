import math
import torch
from torch import nn
from torch.nn import functional as F

from modules.commons.layers import Embedding
from modules.commons.rel_transformer import sequence_mask, MultiHeadAttention, FFN, LayerNorm


def init_zero_conv(m):
    classname = m.__class__.__name__
    if classname.find("Conv1d") != -1:
        torch.nn.init.zeros_(m.weight)
        torch.nn.init.zeros_(m.bias)

class Encoder(nn.Module):
    def __init__(self, hidden_channels, filter_channels, n_heads, n_layers, kernel_size=1, p_dropout=0.,
                 window_size=None, block_length=None, pre_ln=False, **kwargs):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.window_size = window_size
        self.block_length = block_length
        self.pre_ln = pre_ln

        self.drop = nn.Dropout(p_dropout)
        self.attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()
        self.attn_layers_controlnet = nn.ModuleList()
        self.norm_layers_1_controlnet = nn.ModuleList()
        self.ffn_layers_controlnet = nn.ModuleList()
        self.norm_layers_2_controlnet = nn.ModuleList()
        self.zero_convs_controlnet = nn.ModuleList()
        for i in range(self.n_layers):
            self.attn_layers.append(
                MultiHeadAttention(hidden_channels, hidden_channels, n_heads, window_size=window_size,
                                   p_dropout=p_dropout, block_length=block_length))
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(
                FFN(hidden_channels, hidden_channels, filter_channels, kernel_size, p_dropout=p_dropout))
            self.norm_layers_2.append(LayerNorm(hidden_channels))
            self.attn_layers_controlnet.append(
                MultiHeadAttention(hidden_channels, hidden_channels, n_heads, window_size=window_size,
                                   p_dropout=p_dropout, block_length=block_length))
            self.norm_layers_1_controlnet.append(LayerNorm(hidden_channels))
            self.ffn_layers_controlnet.append(
                FFN(hidden_channels, hidden_channels, filter_channels, kernel_size, p_dropout=p_dropout))
            self.norm_layers_2_controlnet.append(LayerNorm(hidden_channels))
            self.zero_convs_controlnet.append(torch.nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1))
        if pre_ln:
            self.last_ln = LayerNorm(hidden_channels)
        
        # init zero convs
        self.zero_convs_controlnet.apply(init_zero_conv)

    def forward(self, x, x_mask):
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        for i in range(self.n_layers):
            x = x * x_mask
            x_controlnet = x

            x_ = x
            if self.pre_ln:
                x = self.norm_layers_1[i](x)
            y = self.attn_layers[i](x, x, attn_mask)
            y = self.drop(y)
            x = x_ + y
            if not self.pre_ln:
                x = self.norm_layers_1[i](x)
            x_ = x
            if self.pre_ln:
                x = self.norm_layers_2[i](x)
            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = x_ + y
            if not self.pre_ln:
                x = self.norm_layers_2[i](x)
            
            # ControlNet
            x_controlnet_ = x_controlnet
            if self.pre_ln:
                x_controlnet = self.norm_layers_1_controlnet[i](x_controlnet)
            y_controlnet = self.attn_layers_controlnet[i](x_controlnet, x_controlnet, attn_mask)
            y_controlnet = self.drop(y_controlnet)
            x_controlnet = x_controlnet_ + y_controlnet
            if not self.pre_ln:
                x_controlnet = self.norm_layers_1_controlnet[i](x_controlnet)
            x_controlnet_ = x_controlnet
            if self.pre_ln:
                x_controlnet = self.norm_layers_2_controlnet[i](x_controlnet)
            y_controlnet = self.ffn_layers_controlnet[i](x_controlnet, x_mask)
            y_controlnet = self.drop(y_controlnet)
            x_controlnet = x_controlnet_ + y_controlnet
            if not self.pre_ln:
                x_controlnet = self.norm_layers_2_controlnet[i](x_controlnet)
            x_controlnet = self.zero_convs_controlnet[i](x_controlnet)

            x = x + x_controlnet

        if self.pre_ln:
            x = self.last_ln(x)
        x = x * x_mask
        return x


class ConvReluNorm(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, kernel_size, n_layers, p_dropout):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.p_dropout = p_dropout
        assert n_layers > 1, "Number of layers should be larger than 0."

        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.conv_layers.append(nn.Conv1d(in_channels, hidden_channels, kernel_size, padding=kernel_size // 2))
        self.norm_layers.append(LayerNorm(hidden_channels))
        self.conv_layers_controlnet = nn.ModuleList()
        self.norm_layers_controlnet = nn.ModuleList()
        self.zero_convs_controlnet = nn.ModuleList()
        self.conv_layers_controlnet.append(nn.Conv1d(in_channels, hidden_channels, kernel_size, padding=kernel_size // 2))
        self.norm_layers_controlnet.append(LayerNorm(hidden_channels))
        self.zero_convs_controlnet.append(torch.nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1))
        self.relu_drop = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(p_dropout))
        for _ in range(n_layers - 1):
            self.conv_layers.append(nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size // 2))
            self.norm_layers.append(LayerNorm(hidden_channels))
            self.conv_layers_controlnet.append(nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size // 2))
            self.norm_layers_controlnet.append(LayerNorm(hidden_channels))
            self.zero_convs_controlnet.append(torch.nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1))
        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

        # init zero convs
        self.zero_convs_controlnet.apply(init_zero_conv)

    def forward(self, x, x_mask):
        x_org = x
        for i in range(self.n_layers):
            x_controlnet = x

            x = self.conv_layers[i](x * x_mask)
            x = self.norm_layers[i](x)
            x = self.relu_drop(x)

            x_controlnet = self.conv_layers_controlnet[i](x_controlnet * x_mask)
            x_controlnet = self.norm_layers_controlnet[i](x_controlnet)
            x_controlnet = self.relu_drop(x_controlnet)
            x_controlnet = self.zero_convs_controlnet[i](x_controlnet)

            x = x + x_controlnet
        x = x_org + self.proj(x)
        return x * x_mask


class RelTransformerEncoder_ControlNet(nn.Module):
    def __init__(self,
                 n_vocab,
                 out_channels,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout=0.0,
                 window_size=4,
                 block_length=None,
                 prenet=True,
                 pre_ln=True,
                 ):

        super().__init__()

        self.n_vocab = n_vocab
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.window_size = window_size
        self.block_length = block_length
        self.prenet = prenet
        if n_vocab > 0:
            self.emb = Embedding(n_vocab, hidden_channels, padding_idx=0)

        if prenet:
            self.pre = ConvReluNorm(hidden_channels, hidden_channels, hidden_channels,
                                    kernel_size=5, n_layers=3, p_dropout=0)
        self.encoder = Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            window_size=window_size,
            block_length=block_length,
            pre_ln=pre_ln,
        )

    def forward(self, x, x_mask=None):
        if self.n_vocab > 0:
            x_lengths = (x > 0).long().sum(-1)
            x = self.emb(x) * math.sqrt(self.hidden_channels)  # [b, t, h]
        else:
            x_lengths = (x.abs().sum(-1) > 0).long().sum(-1)
        x = torch.transpose(x, 1, -1)  # [b, h, t]
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)

        if self.prenet:
            x = self.pre(x, x_mask)
        x = self.encoder(x, x_mask)
        return x.transpose(1, 2)
