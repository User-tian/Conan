# 文件名: verify_causal_upsample_block.py
# 目的: 验证 HiFiGAN Generator 及其组件的因果性 (使用 CausalUpsampleBlock)

import math
import torch
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
import numpy as np
import copy
import traceback # 用于打印详细错误
from modules.vocoder.hifigan.mel_utils import cal_mel_spec

# 确定性设置
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True
# torch.use_deterministic_algorithms(True)

LRELU_SLOPE = 0.1

def init_weights(m, mean: float = 0.0, std: float = 0.01):
    """Initializes weights for Conv1d and ConvTranspose1d layers."""
    if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
        nn.init.normal_(m.weight, mean, std)

# -----------------------------
# Causal Convolution Blocks
# -----------------------------
class CausalConv1d(nn.Module):
    """
    Causal 1D Convolution.

    Ensures that the output at time t only depends on input at or before time t.
    Implements both parallel forward pass for training and streaming inference.
    """
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        # Calculate padding required on the left for causality
        self.left_pad = (kernel_size - 1) * dilation
        # Use weight normalization for the convolution layer
        self.conv = weight_norm(nn.Conv1d(in_ch, out_ch, kernel_size,
                                          dilation=dilation, padding=0)) # No built-in padding
        # Buffer for streaming inference cache
        # self.register_buffer('_cache', torch.zeros(1, in_ch, self.left_pad))
        init_weights(self.conv) # Initialize weights

    def forward(self, x):
        """Parallel forward pass for training."""
        # Manually pad on the left
        x_pad = F.pad(x, (self.left_pad, 0))
        return self.conv(x_pad)

    def remove_weight_norm(self):
        """Removes weight normalization from the convolution layer."""
        remove_weight_norm(self.conv)

class CausalUpsampleBlock1(nn.Module):
    """
    Strictly Causal Transposed Convolution 1D.

    Uses manual padding and output trimming to ensure causality.
    Kernel size must be even, stride >= 2.
    Forward and stream methods behave identically.
    Output length: T_out = T_in * stride
    """
    def __init__(self, in_ch, out_ch, conv_kernel_size, stride):
        super().__init__()
        assert conv_kernel_size % 2 == 0, "kernel_size must be even for CausalConvTranspose1d"
        assert stride >= 2, "stride must be >= 2 for this CausalConvTranspose1d implementation"
        self.stride = stride
        self.kernel_size = conv_kernel_size

        # Manual left padding required for causal input history
        # Derived from standard ConvTranspose1d padding logic for causality
        self.manual_left_pad = self.kernel_size // 2 - 1

        # ConvTranspose1d parameters: disable built-in padding
        convt_padding = 0
        # Choose output_padding to simplify length calculations when convt_padding=0
        # Standard formula: L_out = (L_in - 1)*s - 2*p + d*(k-1) + op + 1
        # With p=0, d=1: L_out = (L_in - 1)*s + k + op
        # For our padded input L_in' = L_in + P_manual:
        # L_out_full = (L_in' - 1)*s + k + op
        # We want the final causal output L_out = L_in * s
        # Choosing op = s - 1 simplifies trimming calculation later.
        convt_output_padding = stride - 1

        self.deconv = weight_norm(nn.ConvTranspose1d(
            in_ch, out_ch,
            kernel_size=self.kernel_size,
            stride=stride,
            padding=convt_padding,          # Disable built-in padding
            output_padding=convt_output_padding # Set calculated output padding
        ))
        init_weights(self.deconv) # Initialize weights

        # Calculate the amount to trim from the left of the raw deconv output
        # This removes the transient effect of the manual left padding.
        # L_out_full = (L_in' - 1)*s + k + op
        #            = (L_in + P_manual - 1)*s + k + (s-1)
        #            = L_in*s + P_manual*s - s + k + s - 1
        #            = L_in*s + P_manual*s + k - 1
        # We want L_out = L_in * s.
        # Amount to trim N = L_out_full - L_out
        #                 = (L_in*s + P_manual*s + k - 1) - L_in*s
        #                 = P_manual*s + k - 1
        self.output_trim_left = self.manual_left_pad * stride + self.kernel_size - 1

        # Cache for streaming inference
        self.register_buffer('_cache', torch.zeros(1, in_ch, self.manual_left_pad))

    # —— Offline forward pass (strictly causal) —— #
    def forward(self, x): # Input: (B, C, T_in) -> Output: (B, C_out, T_in * stride)
        # 1. Apply manual left padding
        x_pad = F.pad(x, (self.manual_left_pad, 0)) # Shape: (B, C, P_manual + T_in)

        # 2. Perform transposed convolution (with padding=0)
        y_full = self.deconv(x_pad)
        # Expected full output length: L_out_full = (P_manual + T_in - 1)*s + k + s - 1

        # 3. Trim the left part of the output to remove padding artifacts and ensure causality
        y_causal = y_full[:, :, self.output_trim_left:]
        # Expected causal output length: T_in * stride
        # Shape: (B, C_out, T_in * stride)

        # Optional check for output length correctness
        expected_len = x.size(2) * self.stride
        if y_causal.size(2) != expected_len:
             # Handle mismatch, e.g., pad or truncate if necessary, or raise error
             print(f"Warning: CausalConvTranspose1d forward output length mismatch. "
                   f"Expected {expected_len}, got {y_causal.size(2)}. Trimming/Padding...")
             if y_causal.size(2) > expected_len:
                 y_causal = y_causal[:, :, :expected_len]
             else:
                 padding_needed = expected_len - y_causal.size(2)
                 y_causal = F.pad(y_causal, (0, padding_needed))

        return y_causal

    def remove_weight_norm(self):
        """Removes weight normalization from the deconvolution layer."""
        remove_weight_norm(self.deconv)

# -----------------------------
# 2222
# -----------------------------

class CausalUpsampleBlock2(nn.Module):
    """使用 手动插零 + 因果卷积 实现因果上采样"""
    def __init__(self, in_channels, out_channels, stride, conv_kernel_size=5):
        super().__init__()
        self.stride = stride
        self.conv = CausalConv1d(in_channels, out_channels, kernel_size=conv_kernel_size, dilation=1)
        # print(f"[CausalUpsampleBlock Init] k={conv_kernel_size}, s={stride}, in_ch={in_channels}, out_ch={out_channels}")
    def forward(self, x):
        B, C, T_in = x.shape; T_out = T_in * self.stride
        x_upsampled = torch.zeros(B, C, T_out, device=x.device, dtype=x.dtype)
        x_upsampled[:, :, ::self.stride] = x
        output = self.conv(x_upsampled)
        if output.shape[2] != T_out: print(f"警告: CausalUpsampleBlock T={output.shape[2]} != Expected T={T_out}!")
        return output
    def remove_weight_norm(self): self.conv.remove_weight_norm()

# -----------------------------
# 3333
# -----------------------------

class CausalPixelShuffle1d(nn.Module):
    """
    1D adaptation of PixelShuffle for causal upsampling.
    """
    def __init__(self, upscale_factor):
        super().__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        # x shape: (B, C * r, T) where r is upscale_factor
        B, C_r, T = x.shape
        assert C_r % self.upscale_factor == 0
        C = C_r // self.upscale_factor
        r = self.upscale_factor

        x = x.view(B, C, r, T)  # (B, C, r, T)
        x = x.permute(0, 1, 3, 2)  # (B, C, T, r) - 把 r 放到最后
        x = x.reshape(B, C, T * r) # (B, C, T * r) - 展开 T*r 维度
        return x

class CausalUpsampleBlock3(nn.Module):
    """
    Upsampling block using CausalConv1d followed by PixelShuffle1d.
    """
    def __init__(self, in_channels, out_channels, stride, conv_kernel_size=3):
         super().__init__()
         self.stride = stride
         # Convolution produces stride * out_channels
         intermediate_channels = out_channels * stride
         # 使用 CausalConv1d 保证卷积部分的因果性
         self.conv = CausalConv1d(in_channels, intermediate_channels, kernel_size=conv_kernel_size)
         self.shuffle = CausalPixelShuffle1d(stride)
         # print(f"[CausalUpsampleSubPixel] k={conv_kernel_size}, s={stride}, in={in_channels}, mid={intermediate_channels}, out={out_channels}")

    def forward(self, x):
        x = self.conv(x) # (B, out_channels * stride, T)
        x = self.shuffle(x) # (B, out_channels, T * stride)
        return x

    def remove_weight_norm(self):
         if hasattr(self, 'conv') and hasattr(self.conv, 'remove_weight_norm'):
            self.conv.remove_weight_norm()

# -----------------------------
# Residual Blocks (Causal)
# -----------------------------
class ResBlock1(nn.Module):
    """Causal Residual Block type 1 (from original HiFi-GAN)."""
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList([
            CausalConv1d(channels, channels, kernel_size, d)
            for d in dilation
        ])
        self.convs2 = nn.ModuleList([
            CausalConv1d(channels, channels, kernel_size, 1) # Dilation reset to 1 for second conv
            for _ in dilation
        ])

    def forward(self, x):
        """Parallel forward pass for training."""
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = x + xt # Residual connection
        return x

    def remove_weight_norm(self):
        """Removes weight normalization from all conv layers."""
        for m in (*self.convs1, *self.convs2):
            if hasattr(m, 'remove_weight_norm'):
                m.remove_weight_norm()

class ResBlock2(nn.Module):
    """Causal Residual Block type 2 (simpler variant)."""
    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        super().__init__()
        self.convs = nn.ModuleList([
            CausalConv1d(channels, channels, kernel_size, d)
            for d in dilation
        ])

    def forward(self, x):
        """Parallel forward pass for training."""
        for conv in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = conv(xt)
            x = x + xt # Residual connection
        return x

    def remove_weight_norm(self):
        """Removes weight normalization from all conv layers."""
        for m in self.convs:
             if hasattr(m, 'remove_weight_norm'):
                m.remove_weight_norm()
                
class HifiGanGenerator(nn.Module):
    """
    HiFi-GAN Generator with causal components for streaming inference.
    """
    def __init__(self, hparams):
        super().__init__()
        self.h = hparams # Store hyperparameters
        up_init_ch = hparams.get('upsample_initial_channel', 512) # Default if not in hparams
        in_ch = hparams.get('num_mels', 80) # Number of mel channels

        # Initial convolution (causal)
        self.conv_pre = CausalConv1d(in_ch, up_init_ch, kernel_size=7)

        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        current_ch = up_init_ch
        for i, (u, k) in enumerate(zip(hparams['upsample_rates'], hparams['upsample_kernel_sizes'])):
            out_ch = current_ch // 2
            # Causal Transposed Convolution for upsampling
            if hparams['upsample']=='nn':
                upsample_layer = CausalUpsampleBlock1(current_ch, out_ch, conv_kernel_size=k, stride=u)
            elif hparams['upsample']=='zero':
                upsample_layer = CausalUpsampleBlock2(current_ch, out_ch, conv_kernel_size=k, stride=u)
            elif hparams['upsample']=='shuffle':
                upsample_layer = CausalUpsampleBlock3(current_ch, out_ch, conv_kernel_size=k, stride=u)
            self.ups.append(upsample_layer)

            # Add residual blocks after each upsampling layer
            resblock_ch = out_ch
            for j, (res_k, res_d) in enumerate(zip(hparams['resblock_kernel_sizes'], hparams['resblock_dilation_sizes'])):
                resblock_type = hparams.get('resblock', '1') # Default to type '1'
                if resblock_type == '1':
                    block = ResBlock1(resblock_ch, res_k, res_d)
                else:
                    block = ResBlock2(resblock_ch, res_k, res_d)
                self.resblocks.append(block) # Append flattened list of resblocks

            current_ch = out_ch # Update channel size for next layer

        # Final convolution (causal)
        self.conv_post = CausalConv1d(current_ch, 1, kernel_size=7)

        # Apply weight initialization
        self.apply(init_weights)

    def forward(self, x, f0=None):
        """Parallel forward pass for training."""
        # Input x: Mel-spectrogram (B, num_mels, T_mel)
        # f0 is not used in this basic HiFiGAN generator structure
        x = self.conv_pre(x) # (B, up_init_ch, T_mel)
        resblock_idx = 0
        for i in range(len(self.ups)):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x) # Upsample (B, C_out, T_mel * stride)
            # Apply corresponding residual blocks
            xs = 0 # Sum of outputs from parallel residual blocks for this scale
            num_resblocks_per_scale = len(self.h['resblock_kernel_sizes'])
            for _ in range(num_resblocks_per_scale):
                xs += self.resblocks[resblock_idx](x)
                resblock_idx += 1
            x = xs / num_resblocks_per_scale # Average the outputs

        x = F.leaky_relu(x, LRELU_SLOPE)
        x = self.conv_post(x) # (B, 1, T_audio)
        return torch.tanh(x) # Output waveform

    def remove_weight_norm(self):
        print("正在移除 Generator 中的 weight_norm...")
        self.conv_pre.remove_weight_norm()
        for up_layer in self.ups: up_layer.remove_weight_norm()
        for res_block in self.resblocks: res_block.remove_weight_norm()
        self.conv_post.remove_weight_norm()
        print("Generator weight_norm 移除完成.")


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)

class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False, use_cond=False, c_in=1):
        super(DiscriminatorP, self).__init__()
        self.use_cond = use_cond
        if use_cond:
            from utils.hparams import hparams
            t = hparams['hop_size']
            self.cond_net = torch.nn.ConvTranspose1d(80, 1, t * 2, stride=t, padding=t // 2)
            c_in = 2

        self.period = period
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv2d(c_in, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x, mel=None):
        fmap = []
        if self.use_cond:
            x_mel = self.cond_net(mel)
            x = torch.cat([x_mel, x], 1)
        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, use_cond=False, c_in=1):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorP(2, use_cond=use_cond, c_in=c_in),
            DiscriminatorP(3, use_cond=use_cond, c_in=c_in),
            DiscriminatorP(5, use_cond=use_cond, c_in=c_in),
            DiscriminatorP(7, use_cond=use_cond, c_in=c_in),
            DiscriminatorP(11, use_cond=use_cond, c_in=c_in),
        ])

    def forward(self, y, y_hat, mel=None):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y, mel)
            y_d_g, fmap_g = d(y_hat, mel)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False, use_cond=False, upsample_rates=None, c_in=1):
        super(DiscriminatorS, self).__init__()
        self.use_cond = use_cond
        if use_cond:
            t = np.prod(upsample_rates)
            self.cond_net = torch.nn.ConvTranspose1d(80, 1, t * 2, stride=t, padding=t // 2)
            c_in = 2
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv1d(c_in, 128, 15, 1, padding=7)),
            norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x, mel):
        if self.use_cond:
            x_mel = self.cond_net(mel)
            x = torch.cat([x_mel, x], 1)
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(self, use_cond=False, c_in=1):
        super(MultiScaleDiscriminator, self).__init__()
        from utils.hparams import hparams
        self.discriminators = nn.ModuleList([
            DiscriminatorS(use_spectral_norm=True, use_cond=use_cond,
                           upsample_rates=[4, 4, hparams['hop_size'] // 16],
                           c_in=c_in),
            DiscriminatorS(use_cond=use_cond,
                           upsample_rates=[4, 4, hparams['hop_size'] // 32],
                           c_in=c_in),
            DiscriminatorS(use_cond=use_cond,
                           upsample_rates=[4, 4, hparams['hop_size'] // 64],
                           c_in=c_in),
        ])
        self.meanpools = nn.ModuleList([
            AvgPool1d(4, 2, padding=1),
            AvgPool1d(4, 2, padding=1)
        ])

    def forward(self, y, y_hat, mel=None):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i - 1](y)
                y_hat = self.meanpools[i - 1](y_hat)
            y_d_r, fmap_r = d(y, mel)
            y_d_g, fmap_g = d(y_hat, mel)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss * 2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    r_losses = 0
    g_losses = 0
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg ** 2)
        r_losses += r_loss
        g_losses += g_loss
    r_losses = r_losses / len(disc_real_outputs)
    g_losses = g_losses / len(disc_real_outputs)
    return r_losses, g_losses


def cond_discriminator_loss(outputs):
    loss = 0
    for dg in outputs:
        g_loss = torch.mean(dg ** 2)
        loss += g_loss
    loss = loss / len(outputs)
    return loss


def generator_loss(disc_outputs):
    loss = 0
    for dg in disc_outputs:
        l = torch.mean((1 - dg) ** 2)
        loss += l
    loss = loss / len(disc_outputs)
    return loss

def mel_loss(y,y_,hparams):
    loss = 0
    for nfft,hop,win in zip(hparams['mel_loss_param']['fft_sizes'],hparams['mel_loss_param']['hop_sizes'],hparams['mel_loss_param']['win_lengths']):

        y_mel = cal_mel_spec(y.squeeze(1),n_fft=nfft, num_mels=hparams['mel_loss_param']['mel_bin'],
                             sampling_rate=hparams['audio_sample_rate'], hop_size=hop, win_size=win,
                             fmin=0, fmax=hparams['audio_sample_rate']//2).transpose(1, 2)
        y_hat_mel = cal_mel_spec(y_.squeeze(1),n_fft=nfft, num_mels=hparams['mel_loss_param']['mel_bin'],
                                 sampling_rate=hparams['audio_sample_rate'], hop_size=hop, win_size=win,
                                 fmin=0, fmax=hparams['audio_sample_rate']//2).transpose(1, 2)
        loss+= F.l1_loss(y_hat_mel, y_mel) * hparams['lambda_mel']

    return loss

# -----------------------------
# Causality Verification Function (修正版)
# -----------------------------
def verify_causality(module: nn.Module, input_shape: tuple, module_name: str = "Module",
                     perturb_magnitude: float = 1e-3, tolerance: float = 1e-6, device: str = 'cpu',
                     stride: int = 1, total_stride: int = 1 ):
    module.eval(); module.to(device)
    is_causal = True
    B, C, T = input_shape
    if T <= 1: print(f"[{module_name}] 输入时间步过短 ({T})，跳过。"); return True

    print(f"--- 正在验证 {module_name} 的因果性 ---")
    x = torch.randn(B, C, T, device=device)
    with torch.no_grad():
        try:
             print(f"[{module_name}] Running initial forward pass...")
             output_orig = module(x)
             print(f"[{module_name}] Initial forward pass successful. Output shape: {output_orig.shape}")
        except Exception as e:
             print(f"[{module_name}] 初始前向传播出错: {e}"); traceback.print_exc(); return False
        output_len = output_orig.shape[2]

        for t in range(T - 1):
            x_pert = x.clone()
            perturbation = torch.randn_like(x_pert[:, :, t + 1:]) * perturb_magnitude
            x_pert[:, :, t + 1:] = x_pert[:, :, t + 1:] + perturbation
            try:
                output_pert = module(x_pert)
            except Exception as e:
                print(f"[{module_name}] 在 t={t} 进行扰动前向传播时出错: {e}"); traceback.print_exc(); is_causal = False; break

            check_len_out = 0
            # --- 修正: 检查 CausalUpsampleBlock ---
            if isinstance(module, (CausalUpsampleBlock1, CausalUpsampleBlock2,CausalUpsampleBlock3)):
                 check_len_out = (t + 1) * stride
            # -------------------------------------
            elif isinstance(module, HifiGanGenerator):
                 check_len_out = (t + 1) * total_stride
            else:
                 check_len_out = t + 1
            check_len_out = min(check_len_out, output_len, output_pert.shape[2])

            slice_orig = output_orig[:, :, :check_len_out]
            slice_pert = output_pert[:, :, :check_len_out]
            if not torch.allclose(slice_orig, slice_pert, atol=tolerance):
                is_causal = False; diff = torch.abs(slice_orig - slice_pert).max()
                print(f"[{module_name}] 在输入时间 t={t} 处因果性检查失败")
                print(f"  输出在计算出的输出索引 {check_len_out-1} 之前就已不同。")
                print(f"  最大绝对差值: {diff.item()}"); break

    if is_causal: print(f"[{module_name}] 通过因果性检查。")
    print(f"--- {module_name} 验证结束 ---")
    return is_causal

# 在脚本中添加这个新函数 (可以放在 verify_causality 后面)

def verify_prefix_consistency(generator: nn.Module, hparams: dict, device: str,
                              t1_frames: int = 8,    # 短输入的帧数 (对应 80ms)
                              t2_frames: int = 16,   # 长输入的帧数 (对应 160ms)
                              batch_size: int = 1,
                              tolerance: float = 1e-6):
    """
    验证生成器对于不同长度但具有相同前缀的输入，其输出前缀是否一致。
    """
    module_name = "Generator Prefix Consistency"
    print(f"--- 正在验证 {module_name} ---")
    print(f"    短输入帧数 T1 = {t1_frames}, 长输入帧数 T2 = {t2_frames}")

    if t2_frames <= t1_frames:
        print(f"错误: t2_frames ({t2_frames}) 必须大于 t1_frames ({t1_frames})")
        return False

    generator.eval()
    generator.to(device)

    num_mels = hparams.get('num_mels', 80)
    total_stride = np.prod(hparams.get('upsample_rates', []))
    if total_stride == 0:
        print("错误: 未能计算总上采样率 (upsample_rates 可能为空?)")
        return False

    # 预期输出长度
    expected_len_short = t1_frames * total_stride
    expected_len_long = t2_frames * total_stride
    print(f"    预期短输出样本数 = {expected_len_short} (约 {t1_frames * hparams['hop_size'] / hparams['audio_sample_rate'] * 1000:.1f} ms)")
    print(f"    预期长输出样本数 = {expected_len_long}")

    is_consistent = False
    with torch.no_grad():
        try:
            # 1. 创建输入
            # 短输入
            mel_short = torch.randn(batch_size, num_mels, t1_frames, device=device)
            # 长输入 (前缀与短输入相同)
            mel_long_suffix = torch.randn(batch_size, num_mels, t2_frames - t1_frames, device=device)
            mel_long = torch.cat([mel_short, mel_long_suffix], dim=2)

            # 2. 生成输出
            print("    生成短输入对应的波形...")
            wav_short = generator(mel_short)
            print(f"    生成长输入对应的波形...")
            wav_long = generator(mel_long)

            # 3. 检查输出形状
            if wav_short.shape[2] != expected_len_short:
                print(f"警告: wav_short 长度 ({wav_short.shape[2]}) 与预期 ({expected_len_short}) 不符!")
            if wav_long.shape[2] != expected_len_long:
                 print(f"警告: wav_long 长度 ({wav_long.shape[2]}) 与预期 ({expected_len_long}) 不符!")

            # 确保比较长度不超过实际得到的长度
            compare_len = min(expected_len_short, wav_short.shape[2], wav_long.shape[2])
            if compare_len != expected_len_short:
                 print(f"警告: 实际比较长度为 {compare_len}, 而非预期的 {expected_len_short}")

            # 4. 比较前缀
            wav_short_prefix = wav_short[:, :, :compare_len]
            wav_long_prefix = wav_long[:, :, :compare_len]

            is_consistent = torch.allclose(wav_short_prefix, wav_long_prefix, atol=tolerance)

            if is_consistent:
                print(f"[{module_name}] 通过前缀一致性检查。")
            else:
                diff = torch.abs(wav_short_prefix - wav_long_prefix).max()
                print(f"[{module_name}] 未通过前缀一致性检查！")
                print(f"  最大绝对差值: {diff.item()}")

        except Exception as e:
            print(f"[{module_name}] 测试过程中发生错误: {e}")
            traceback.print_exc()
            is_consistent = False # 标记为失败

    print(f"--- {module_name} 验证结束 ---")
    return is_consistent

# -----------------------------
# Test Execution (使用 CausalUpsampleBlock)
# -----------------------------
if __name__ == "__main__":
    print("开始验证模块的因果性（使用 CausalUpsampleBlock）...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    # --- 组件测试参数 ---
    B = 2; T_in_comp = 40; C_low = 16; C_high = 64; K = 3; D = 3

    # --- 1. 测试 CausalConv1d ---
    print("\n--- 开始测试独立的 CausalConv1d ---")
    causal_conv = CausalConv1d(C_low, C_low, kernel_size=K, dilation=D).to(device)
    verify_causality(causal_conv, (B, C_low, T_in_comp), "CausalConv1d", device=device)
    print("--- 独立 CausalConv1d 测试结束 ---")
    print("-" * 20)

    # --- 定义 hparams (添加新参数) ---
    hparams_from_config = {
        'upsample_rates': [4, 5, 4, 2],
        'upsample_kernel_sizes': [8, 10, 8, 4],
        'audio_sample_rate': 16000, 'fmin': 80, 'fmax': 7600, 'fft_size': 1024,
        'hop_size': 160, 'win_size': 1024,
        'mel_loss_param': {'mel_bin': 320, 'fft_sizes': [512, 1024, 2048],
                           'hop_sizes': [160, 160, 160], 'win_lengths': [512, 1024, 2048]},
        'lambda_mel': 45,
        'num_mels': 80, 'upsample_initial_channel': 512, 'resblock': '1',
        'resblock_kernel_sizes': [3, 7, 11],
        'resblock_dilation_sizes': [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        'upsample': 'shuffle'
    }
    # -----------------------------------------------

    # --- 2. 测试 CausalUpsampleBlock ---
    print("\n--- 开始测试独立的 CausalUpsampleBlock ---")
    first_up_rate = hparams_from_config['upsample_rates'][0]
    initial_channel = hparams_from_config['upsample_initial_channel']
    upsample_conv_k = hparams_from_config['upsample_kernel_sizes'][0]
    if hparams_from_config['upsample'] == 'nn':
        causal_upsample_test = CausalUpsampleBlock1(initial_channel, initial_channel // 2,
                                           stride=first_up_rate,
                                           conv_kernel_size=upsample_conv_k).to(device)
    elif hparams_from_config['upsample'] == 'zero':
        causal_upsample_test = CausalUpsampleBlock2(initial_channel, initial_channel // 2,
                                           stride=first_up_rate,
                                           conv_kernel_size=upsample_conv_k).to(device)
    elif hparams_from_config['upsample'] == 'shuffle':
        causal_upsample_test = CausalUpsampleBlock3(initial_channel, initial_channel // 2,
                                           stride=first_up_rate,
                                           conv_kernel_size=upsample_conv_k).to(device)
    verify_causality(causal_upsample_test, (B, initial_channel, T_in_comp),
                     f"CausalUpsampleBlock (s={first_up_rate}, conv_k={upsample_conv_k})",
                     stride=first_up_rate, device=device) # 传入 stride
    print("--- 独立 CausalUpsampleBlock 测试结束 ---")
    print("-" * 20)

    # --- 3. 测试 ResBlock ---
    print("\n--- 开始测试 ResBlock ---")
    res_block_to_test = None; res_block_name = ""
    if hparams_from_config['resblock'] == '1':
         res_channel_after_first_up = initial_channel // 2
         res_kernels = hparams_from_config['resblock_kernel_sizes']
         res_dilations = hparams_from_config['resblock_dilation_sizes']
         res_block_to_test = ResBlock1(res_channel_after_first_up, kernel_size=res_kernels[0], dilation=res_dilations[0]).to(device)
         res_block_name = f"ResBlock1 (k={res_kernels[0]}, d={res_dilations[0]})"
    elif hparams_from_config['resblock'] == '2':
         res_channel_after_first_up = initial_channel // 2
         res_kernels = hparams_from_config['resblock_kernel_sizes']
         res_dilations = hparams_from_config['resblock_dilation_sizes']
         res_block_to_test = ResBlock2(res_channel_after_first_up, kernel_size=res_kernels[0], dilation=res_dilations[0]).to(device)
         res_block_name = f"ResBlock2 (k={res_kernels[0]}, d={res_dilations[0]})"
    if res_block_to_test: verify_causality(res_block_to_test, (B, res_channel_after_first_up, T_in_comp), res_block_name, device=device)
    else: print("未配置有效的 ResBlock 类型，跳过测试。")
    print("--- ResBlock 测试结束 ---")
    print("-" * 20)

# --- 4. 测试使用 CausalUpsampleBlock 的 HifiGanGenerator ---
    print("\n--- 开始测试完整的 HifiGanGenerator (Causal Upsample) ---")
    total_generator_stride = np.prod(hparams_from_config['upsample_rates'])
    print(f"计算得到的生成器总上采样步幅: {total_generator_stride}")

    print(">>> 即将实例化 HifiGanGenerator...")
    generator = None # 先置为 None
    try:
        generator = HifiGanGenerator(hparams_from_config).to(device)
        print(">>> HifiGanGenerator 实例化并移至设备成功！")
    except Exception as e:
        print(f">>> HifiGanGenerator 实例化或移至设备时出错: {e}")
        traceback.print_exc()
    # ------------------------------

    if generator is not None: # 仅在实例化成功后才进行测试
        T_mel = 25 # 这个 T_mel 是给 verify_causality 用的
        gen_input_shape = (B, hparams_from_config['num_mels'], T_mel)

        # --- 首先进行原来的逐样本因果性测试 (如果需要取消注释) ---
        # print(">>> 即将调用 verify_causality 测试 Generator...")
        # causal_test_passed = verify_causality(generator, gen_input_shape, "HifiGanGenerator (Causal Upsample)",
        #                                       total_stride=total_generator_stride, device=device)
        # print(f">>> Generator verify_causality 测试结果: {'通过' if causal_test_passed else '失败'}")
        # print("-" * 10) # 分隔符
        # ----------------------------------------------------

        # --- 然后进行新的前缀一致性测试 ---
        print(">>> 即将调用 verify_prefix_consistency 测试 Generator...")
        prefix_test_passed = verify_prefix_consistency(generator, hparams_from_config, device)
        print(f">>> Generator verify_prefix_consistency 测试结果: {'通过' if prefix_test_passed else '失败'}")
        # ---------------------------------

        print("--- 完整 HifiGanGenerator 测试结束 ---")
    else:
        print(">>> Generator 实例化失败，跳过因果性测试。")

    print("-" * 20)
    print("因果性验证完成.")