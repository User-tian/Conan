import torch
from torch import nn


def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts


class WN(torch.nn.Module):
    def __init__(self, hidden_size, kernel_size, dilation_rate, n_layers, c_cond=0,
                 p_dropout=0, share_cond_layers=False, is_BTC=False):
        super(WN, self).__init__()
        assert (kernel_size % 2 == 1)
        assert (hidden_size % 2 == 0)
        self.is_BTC = is_BTC
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = c_cond
        self.p_dropout = p_dropout
        self.share_cond_layers = share_cond_layers

        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()
        self.drop = nn.Dropout(p_dropout)

        if c_cond != 0 and not share_cond_layers:
            cond_layer = torch.nn.Conv1d(c_cond, 2 * hidden_size * n_layers, 1)
            self.cond_layer = torch.nn.utils.weight_norm(cond_layer, name='weight')

        for i in range(n_layers):
            dilation = dilation_rate ** i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = torch.nn.Conv1d(hidden_size, 2 * hidden_size, kernel_size,
                                       dilation=dilation, padding=padding)
            in_layer = torch.nn.utils.weight_norm(in_layer, name='weight')
            self.in_layers.append(in_layer)

            # last one is not necessary
            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_size
            else:
                res_skip_channels = hidden_size

            res_skip_layer = torch.nn.Conv1d(hidden_size, res_skip_channels, 1)
            res_skip_layer = torch.nn.utils.weight_norm(res_skip_layer, name='weight')
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, x, nonpadding=None, cond=None):
        if self.is_BTC:
            x = x.transpose(1, 2)
            cond = cond.transpose(1, 2) if cond is not None else None
            nonpadding = nonpadding.transpose(1, 2) if nonpadding is not None else None
        if nonpadding is None:
            nonpadding = 1
        output = torch.zeros_like(x)
        n_channels_tensor = torch.IntTensor([self.hidden_size])

        if cond is not None and not self.share_cond_layers:
            cond = self.cond_layer(cond)

        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            x_in = self.drop(x_in)
            if cond is not None:
                cond_offset = i * 2 * self.hidden_size
                cond_l = cond[:, cond_offset:cond_offset + 2 * self.hidden_size, :]
            else:
                cond_l = torch.zeros_like(x_in)

            acts = fused_add_tanh_sigmoid_multiply(x_in, cond_l, n_channels_tensor)

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                x = (x + res_skip_acts[:, :self.hidden_size, :]) * nonpadding
                output = output + res_skip_acts[:, self.hidden_size:, :]
            else:
                output = output + res_skip_acts
        output = output * nonpadding
        if self.is_BTC:
            output = output.transpose(1, 2)
        return output

    def remove_weight_norm(self):
        def remove_weight_norm(m):
            try:
                nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(remove_weight_norm)

import torch.nn.functional as F

def kaiming_init(module):
    if isinstance(module, nn.Conv1d):
        nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    return module


class CausalConv1d(nn.Module):
    """
    只在左侧做 padding 的 1-D dilated 卷积:
    输出第 t 帧只依赖于 ≤t 的输入。
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 dilation: int = 1,
                 **kwargs):
        super().__init__()
        self.conv = kaiming_init(
            nn.Conv1d(in_channels,
                      out_channels,
                      kernel_size,
                      padding=0,
                      dilation=dilation,
                      **kwargs)
        )
        # inline weight-norm；保持与旧版一致
        self.conv = nn.utils.weight_norm(self.conv, name='weight')
        # 需要在左侧补多少帧
        self.left_pad = dilation * (kernel_size - 1)

    def forward(self, x):
        # 只 pad 左边
        x = F.pad(x, (self.left_pad, 0))
        return self.conv(x)

class CausalWN(nn.Module):
    """
    Causal WaveNet residual block.
    接口与原实现保持一致，可直接替换。
    """
    def __init__(self, hidden_size, kernel_size, dilation_rate, n_layers,
                 c_cond=0, p_dropout=0, share_cond_layers=False, is_BTC=False):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size 必须为奇数"
        assert hidden_size % 2 == 0, "hidden_size 必须为偶数"

        self.is_BTC = is_BTC
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = c_cond
        self.p_dropout = p_dropout
        self.share_cond_layers = share_cond_layers

        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()
        self.drop = nn.Dropout(p_dropout)

        # 条件卷积层（kernel_size=1 本身就是 causal）
        if c_cond != 0 and not share_cond_layers:
            cond_layer = nn.Conv1d(c_cond, 2 * hidden_size * n_layers, 1)
            self.cond_layer = nn.utils.weight_norm(cond_layer, name='weight')

        # 堆叠 dilated causal conv
        for i in range(n_layers):
            dilation = dilation_rate ** i
            in_layer = CausalConv1d(hidden_size,
                                    2 * hidden_size,
                                    kernel_size,
                                    dilation=dilation)
            self.in_layers.append(in_layer)

            # 最后一层只输出 skip，不再输出 residual
            res_skip_channels = 2 * hidden_size if i < n_layers - 1 else hidden_size
            res_skip_layer = nn.Conv1d(hidden_size, res_skip_channels, 1)
            res_skip_layer = nn.utils.weight_norm(res_skip_layer, name='weight')
            self.res_skip_layers.append(res_skip_layer)

    # --------------------------------------------------
    #  前向传播：逻辑保持不变
    # --------------------------------------------------
    def forward(self, x, nonpadding=None, cond=None):
        if self.is_BTC:          # [B,T,C] → [B,C,T]
            x = x.transpose(1, 2)
            cond = cond.transpose(1, 2) if cond is not None else None
            nonpadding = nonpadding.transpose(1, 2) if nonpadding is not None else None

        if nonpadding is None:
            nonpadding = 1

        output = torch.zeros_like(x)
        n_channels_tensor = torch.IntTensor([self.hidden_size])

        if cond is not None and not self.share_cond_layers:
            cond = self.cond_layer(cond)

        for i in range(self.n_layers):
            x_in = self.drop(self.in_layers[i](x))

            cond_l = (cond[:, i * 2 * self.hidden_size:(i + 1) * 2 * self.hidden_size, :]
                      if cond is not None else torch.zeros_like(x_in))

            acts = fused_add_tanh_sigmoid_multiply(x_in, cond_l, n_channels_tensor)

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                x = (x + res_skip_acts[:, :self.hidden_size, :]) * nonpadding
                output = output + res_skip_acts[:, self.hidden_size:, :]
            else:
                output = output + res_skip_acts

        output = output * nonpadding
        if self.is_BTC:
            output = output.transpose(1, 2)   # [B,C,T] → [B,T,C]
        return output

    # 方便导出推理模型
    def remove_weight_norm(self):
        def _remove(m):
            try:
                nn.utils.remove_weight_norm(m)
            except ValueError:
                pass
        self.apply(_remove)