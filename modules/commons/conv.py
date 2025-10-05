import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.commons.layers import LayerNorm, Embedding


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


def init_weights_func(m):
    classname = m.__class__.__name__
    if classname.find("Conv1d") != -1:
        torch.nn.init.xavier_uniform_(m.weight)

def get_norm_builder(norm_type, channels, ln_eps=1e-6):
    if norm_type == 'bn':
        norm_builder = lambda: nn.BatchNorm1d(channels)
    elif norm_type == 'in':
        norm_builder = lambda: nn.InstanceNorm1d(channels, affine=True)
    elif norm_type == 'gn':
        norm_builder = lambda: nn.GroupNorm(8, channels)
    elif norm_type == 'ln':
        norm_builder = lambda: LayerNorm(channels, dim=1, eps=ln_eps)
    else:
        norm_builder = lambda: nn.Identity()
    return norm_builder

def get_act_builder(act_type):
    if act_type == 'gelu':
        act_builder = lambda: nn.GELU()
    elif act_type == 'relu':
        act_builder = lambda: nn.ReLU(inplace=True)
    elif act_type == 'leakyrelu':
        act_builder = lambda: nn.LeakyReLU(negative_slope=0.01, inplace=True)
    elif act_type == 'swish':
        act_builder = lambda: nn.SiLU(inplace=True)
    else:
        act_builder = lambda: nn.Identity()
    return act_builder

class ResidualBlock(nn.Module):
    """Implements conv->PReLU->norm n-times"""

    def __init__(self, channels, kernel_size, dilation, n=2, norm_type='bn', dropout=0.0,
                 c_multiple=2, ln_eps=1e-12, act_type='gelu'):
        super(ResidualBlock, self).__init__()

        norm_builder = get_norm_builder(norm_type, channels, ln_eps)
        act_builder = get_act_builder(act_type)
        self.blocks = [
            nn.Sequential(
                norm_builder(),
                nn.Conv1d(channels, c_multiple * channels, kernel_size, dilation=dilation,
                          padding=(dilation * (kernel_size - 1)) // 2),
                LambdaLayer(lambda x: x * kernel_size ** -0.5),
                act_builder(),
                nn.Conv1d(c_multiple * channels, channels, 1, dilation=dilation),
            )
            for i in range(n)
        ]

        self.blocks = nn.ModuleList(self.blocks)
        self.dropout = dropout

    def forward(self, x):
        nonpadding = (x.abs().sum(1) > 0).float()[:, None, :]
        for b in self.blocks:
            x_ = b(x)
            if self.dropout > 0 and self.training:
                x_ = F.dropout(x_, self.dropout, training=self.training)
            x = x + x_
            x = x * nonpadding
        return x


class ConvBlocks(nn.Module):
    """Decodes the expanded phoneme encoding into spectrograms"""

    def __init__(self, hidden_size, out_dims, dilations, kernel_size,
                 norm_type='ln', layers_in_block=2, c_multiple=2,
                 dropout=0.0, ln_eps=1e-5,
                 init_weights=True, is_BTC=True, num_layers=None, post_net_kernel=3, act_type='gelu'):
        super(ConvBlocks, self).__init__()
        self.is_BTC = is_BTC
        if num_layers is not None:
            dilations = [1] * num_layers
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hidden_size, kernel_size, d,
                            n=layers_in_block, norm_type=norm_type, c_multiple=c_multiple,
                            dropout=dropout, ln_eps=ln_eps, act_type=act_type)
              for d in dilations],
        )
        norm = get_norm_builder(norm_type, hidden_size, ln_eps)()
        self.last_norm = norm
        self.post_net1 = nn.Conv1d(hidden_size, out_dims, kernel_size=post_net_kernel,
                                   padding=post_net_kernel // 2)
        if init_weights:
            self.apply(init_weights_func)

    def forward(self, x, nonpadding=None):
        """

        :param x: [B, T, H]
        :return:  [B, T, H]
        """
        if self.is_BTC:
            x = x.transpose(1, 2)
        if nonpadding is None:
            nonpadding = (x.abs().sum(1) > 0).float()[:, None, :]
        elif self.is_BTC:
            nonpadding = nonpadding.transpose(1, 2)
        x = self.res_blocks(x) * nonpadding
        x = self.last_norm(x) * nonpadding
        x = self.post_net1(x) * nonpadding
        if self.is_BTC:
            x = x.transpose(1, 2)
        return x

class CausalResidualBlock(nn.Module):
    """Causal convolution residual block (norm→causal conv→act→1×1 conv)×n."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        *,
        n: int = 2,
        norm_type: str = "bn",
        dropout: float = 0.0,
        c_multiple: int = 2,
        ln_eps: float = 1e-12,
        act_type: str = "gelu",
    ) -> None:
        super().__init__()

        norm_builder = get_norm_builder(norm_type, channels, ln_eps)
        act_builder = get_act_builder(act_type)

        def _causal_pad(inp, k: int = kernel_size, d: int = dilation):
            left_pad = d * (k - 1)
            return F.pad(inp, (left_pad, 0))

        self.blocks = nn.ModuleList([
            nn.Sequential(
                norm_builder(),
                LambdaLayer(_causal_pad),
                nn.Conv1d(
                    channels,
                    c_multiple * channels,
                    kernel_size,
                    padding=0,
                    dilation=dilation,
                ),
                LambdaLayer(lambda x, k=kernel_size: x * k ** -0.5),
                act_builder(),
                nn.Conv1d(c_multiple * channels, channels, 1),
            )
            for _ in range(n)
        ])
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        nonpadding = (x.abs().sum(1) > 0).float()[:, None, :]
        for blk in self.blocks:
            x_ = blk(x)
            if self.dropout > 0 and self.training:
                x_ = F.dropout(x_, self.dropout, self.training)
            x = (x + x_) * nonpadding  # 保持掩码对齐
        return x


class CausalConvBlocks(nn.Module):
    """Stack of CausalResidualBlock + causal post‑net conv.

    Args:
        hidden_size: 通道数 (d_model)
        out_dims: 输出通道数
        dilations: 每个 residual block 的 dilation 序列
        kernel_size: 卷积核大小
        is_BTC: True 则输入 [B, T, C]，否则 [B, C, T]
    """

    def __init__(
        self,
        hidden_size: int,
        out_dims: int,
        dilations: list[int] | tuple[int, ...],
        kernel_size: int = 5,
        *,
        norm_type: str = "ln",
        layers_in_block: int = 2,
        c_multiple: int = 2,
        dropout: float = 0.0,
        ln_eps: float = 1e-5,
        init_weights: bool = True,
        is_BTC: bool = True,
        num_layers: int | None = None,
        post_net_kernel: int = 3,
        act_type: str = "gelu",
    ) -> None:
        super().__init__()
        self.is_BTC = is_BTC

        if num_layers is not None:
            dilations = [1] * num_layers

        self.res_blocks = nn.Sequential(
            *[
                CausalResidualBlock(
                    hidden_size,
                    kernel_size,
                    d,
                    n=layers_in_block,
                    norm_type=norm_type,
                    c_multiple=c_multiple,
                    dropout=dropout,
                    ln_eps=ln_eps,
                    act_type=act_type,
                )
                for d in dilations
            ]
        )

        self.last_norm = get_norm_builder(norm_type, hidden_size, ln_eps)()

        self.post_net1 = nn.Sequential(
            LambdaLayer(lambda x, k=post_net_kernel: F.pad(x, (k - 1, 0))),
            nn.Conv1d(hidden_size, out_dims, post_net_kernel, padding=0),
        )

        if init_weights:
            self.apply(init_weights_func)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, x: torch.Tensor, nonpadding: torch.Tensor | None = None
    ) -> torch.Tensor:  # type: ignore[override]
        """x: [B, T, H] if is_BTC else [B, H, T]"""
        if self.is_BTC:
            x = x.transpose(1, 2)
        if nonpadding is None:
            nonpadding = (x.abs().sum(1) > 0).float()[:, None, :]
        elif self.is_BTC:
            nonpadding = nonpadding.transpose(1, 2)

        x = self.res_blocks(x) * nonpadding
        x = self.last_norm(x) * nonpadding
        x = self.post_net1(x) * nonpadding

        if self.is_BTC:
            x = x.transpose(1, 2)
        return x


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class CausalFM(nn.Module):
    """
    Stack of CausalResidualBlock + causal post‑net conv,
    增加 diffusion_step 和 cond 条件输入，补偿因果卷积带来的时延
    并保证 cond 输入也是因果对齐
    """
    def __init__(
        self,
        hidden_size: int,
        out_dims: int,
        dilations: list[int] | tuple[int, ...],
        kernel_size: int = 5,
        *,
        cond_dims: int,
        norm_type: str = "ln",
        layers_in_block: int = 2,
        c_multiple: int = 2,
        dropout: float = 0.0,
        ln_eps: float = 1e-5,
        init_weights: bool = True,
        is_BTC: bool = True,
        num_layers: int | None = None,
        post_net_kernel: int = 3,
        act_type: str = "gelu",
    ) -> None:
        super().__init__()
        self.is_BTC = is_BTC

        # diffusion_step 嵌入和 MLP
        self.diff_emb = SinusoidalPosEmb(hidden_size)
        self.diff_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

        # cond 投影
        self.cond_proj = nn.Conv1d(cond_dims, hidden_size, 1)
        nn.init.kaiming_normal_(self.cond_proj.weight, nonlinearity="linear")

        # 如果指定 num_layers，则统一 dilations
        if num_layers is not None:
            dilations = [1] * num_layers
        self.dilations = dilations
        self.kernel_size = kernel_size

        # 原有残差块
        self.res_blocks = nn.Sequential(
            *[
                CausalResidualBlock(
                    hidden_size,
                    kernel_size,
                    d,
                    n=layers_in_block,
                    norm_type=norm_type,
                    c_multiple=c_multiple,
                    dropout=dropout,
                    ln_eps=ln_eps,
                    act_type=act_type,
                )
                for d in self.dilations
            ]
        )

        self.last_norm = get_norm_builder(norm_type, hidden_size, ln_eps)()
        self.post_net1 = nn.Sequential(
            LambdaLayer(lambda x, k=post_net_kernel: F.pad(x, (k - 1, 0))),
            nn.Conv1d(hidden_size, out_dims, post_net_kernel, padding=0),
        )

        # 计算因果卷积时延
        self.receptive_field = 1 + sum((kernel_size - 1) * d for d in self.dilations)
        self.delay = self.receptive_field - 1  # 需要补偿的帧数

        if init_weights:
            self.apply(init_weights_func)

    def forward(
        self,
        x: torch.Tensor,
        diffusion_step: torch.Tensor,
        cond: torch.Tensor,
        nonpadding: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, H] if is_BTC else [B, H, T]
            diffusion_step: [B, 1]
            cond: [B, T, cond_dims] if is_BTC else [B, cond_dims, T]
        """
        # 转到 [B, H, T]
        if self.is_BTC:
            x = x.transpose(1, 2)
            cond = cond.transpose(1, 2)

        # 推理模式下补偿输入端时延：对 x 和 cond 都进行因果 pad
        if not self.training:
            # pad x
            first_x = x[:, :, :1]                           # [B, H, 1]
            pad_x   = first_x.expand(-1, -1, self.delay)    # [B, H, delay]
            x = torch.cat([pad_x, x], dim=-1)               # [B, H, T+delay]
            # pad cond
            first_c = cond[:, :, :1]                        # [B, C, 1]
            pad_c   = first_c.expand(-1, -1, self.delay)    # [B, C, delay]
            cond = torch.cat([pad_c, cond], dim=-1)         # [B, C, T+delay]

        # 构建 nonpadding mask
        if nonpadding is None:
            nonpadding = (x.abs().sum(1) > 0).float()[:, None, :]
        elif self.is_BTC:
            nonpadding = nonpadding.transpose(1, 2)

        # cond 投影
        cond_feat = self.cond_proj(cond)

        # diffusion_step 嵌入
        diff = self.diff_emb(diffusion_step)  # [B, hidden_size]
        diff = self.diff_mlp(diff)           # [B, hidden_size]
        diff = diff.unsqueeze(-1)            # [B, hidden_size, 1]

        # 条件注入
        x = x + cond_feat + diff

        # 主干计算
        x = self.res_blocks(x) * nonpadding
        x = self.last_norm(x) * nonpadding
        x = self.post_net1(x) * nonpadding

        # 推理模式下输出端裁剪掉补帧
        if not self.training:
            x = x[:, :, self.delay:]

        # 转回原始布局
        if self.is_BTC:
            x = x.transpose(1, 2)
        return x


class TextConvEncoder(ConvBlocks):
    def __init__(self, dict_size, hidden_size, out_dims, dilations, kernel_size,
                 norm_type='ln', layers_in_block=2, c_multiple=2,
                 dropout=0.0, ln_eps=1e-5, init_weights=True, num_layers=None, post_net_kernel=3):
        super().__init__(hidden_size, out_dims, dilations, kernel_size,
                         norm_type, layers_in_block, c_multiple,
                         dropout, ln_eps, init_weights, num_layers=num_layers,
                         post_net_kernel=post_net_kernel)
        self.embed_tokens = Embedding(dict_size, hidden_size, 0)
        self.embed_scale = math.sqrt(hidden_size)

    def forward(self, txt_tokens):
        """

        :param txt_tokens: [B, T]
        :return: {
            'encoder_out': [B x T x C]
        }
        """
        x = self.embed_scale * self.embed_tokens(txt_tokens)
        return super().forward(x)


class ConditionalConvBlocks(ConvBlocks):
    def __init__(self, hidden_size, c_cond, c_out, dilations, kernel_size,
                 norm_type='ln', layers_in_block=2, c_multiple=2,
                 dropout=0.0, ln_eps=1e-5, init_weights=True, is_BTC=True, num_layers=None):
        super().__init__(hidden_size, c_out, dilations, kernel_size,
                         norm_type, layers_in_block, c_multiple,
                         dropout, ln_eps, init_weights, is_BTC=False, num_layers=num_layers)
        self.g_prenet = nn.Conv1d(c_cond, hidden_size, 3, padding=1)
        self.is_BTC_ = is_BTC
        if init_weights:
            self.g_prenet.apply(init_weights_func)

    def forward(self, x, cond, nonpadding=None):
        if self.is_BTC_:
            x = x.transpose(1, 2)
            cond = cond.transpose(1, 2)
            if nonpadding is not None:
                nonpadding = nonpadding.transpose(1, 2)
        if nonpadding is None:
            nonpadding = x.abs().sum(1)[:, None]
        x = x + self.g_prenet(cond)
        x = x * nonpadding
        x = super(ConditionalConvBlocks, self).forward(x)  # input needs to be BTC
        if self.is_BTC_:
            x = x.transpose(1, 2)
        return x
