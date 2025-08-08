from inspect import isfunction
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from utils.commons.hparams import hparams
from torchdyn.core import NeuralODE


# ------------------ 工具函数 ------------------ #
def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(shape, device, repeat=False):
    repeat_noise = (
        lambda: torch.randn((1, *shape[1:]), device=device).repeat(
            shape[0], *((1,) * (len(shape) - 1))
        )
    )
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


# ------------------ ODE 包装 ------------------ #
class Wrapper(nn.Module):
    def __init__(self, net, cond, num_timesteps):
        super().__init__()
        self.net = net
        self.cond = cond
        self.num_timesteps = num_timesteps

    def forward(self, t, x, args):
        t = torch.tensor([t * self.num_timesteps] * x.shape[0], device=t.device).long()
        return self.net.denoise_fn(x, t, self.cond)


class Wrapper_CFG(nn.Module):
    def __init__(self, net, cond, ucond, cfg_scale, num_timesteps):
        super().__init__()
        self.net = net
        self.cond = cond
        self.ucond = ucond
        self.cfg_scale = cfg_scale
        self.num_timesteps = num_timesteps

    def forward(self, t, x, args):
        t = torch.tensor([t * self.num_timesteps] * x.shape[0], device=t.device).long()
        cond_in = torch.cat([self.ucond, self.cond])
        t_in = torch.cat([t] * 2)
        x_in = torch.cat([x] * 2)

        v_uncond, v_cond = self.net.denoise_fn(x_in, t_in, cond_in).chunk(2)
        return v_uncond + self.cfg_scale * (v_cond - v_uncond)


# ------------------ 主模型 ------------------ #
class FlowMel(nn.Module):
    def __init__(
        self,
        out_dims,
        denoise_fn,
        timesteps=1000,
        K_step=1000,
        loss_type=hparams.get("flow_loss_type", "l1"),
        spec_min=None,
        spec_max=None,
    ):
        super().__init__()
        self.denoise_fn = denoise_fn
        self.mel_bins = out_dims
        self.num_timesteps = int(timesteps)
        self.K_step = K_step
        self.loss_type = loss_type

        self.register_buffer(
            "spec_min", torch.FloatTensor(spec_min)[None, None, : hparams["keep_bins"]]
        )
        self.register_buffer(
            "spec_max", torch.FloatTensor(spec_max)[None, None, : hparams["keep_bins"]]
        )

    # ---------- 前向 ---------- #
    def forward(
        self,
        cond,
        gt_mels,
        coarse_mels,
        ret,
        infer,
        ucond=None,
        noise=None,
        cfg_scale=1.0,
        solver="euler",
    ):
        b, *_, device = *cond.shape, cond.device

        cond = cond.transpose(1, 2)
        fs2_mels = coarse_mels

        # 统一掩码形状：(B, T)
        # nonpadding = ret["tgt_nonpadding"]
        nonpadding=None
        if nonpadding!=None and nonpadding.dim() == 3 and nonpadding.size(-1) == 1:
            nonpadding = nonpadding.squeeze(-1)

        if not infer:  # ---------- 训练 ---------- #
            t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
            x = self.norm_spec(gt_mels).transpose(1, 2)[:, None, :, :]  # (B,1,M,T)
            ret["flow"] = self.p_losses(x, t, cond, nonpadding=nonpadding)
        else:  # ---------- 推理 ---------- #
            x0 = noise
            if x0 is None:
                x0 = (
                    default(noise, lambda: torch.randn_like(fs2_mels))
                    .transpose(1, 2)[:, None, :, :]
                )

            if ucond is not None:  # CFG
                ucond = ucond.transpose(1, 2)
                ode_func = self.ode_wrapper_cfg(
                    cond, ucond, cfg_scale, self.num_timesteps
                )
            else:  # 普通
                ode_func = self.ode_wrapper(cond, self.num_timesteps)

            neural_ode = NeuralODE(
                ode_func,
                solver=solver,
                sensitivity="adjoint",
                atol=1e-4,
                rtol=1e-4,
            )
            t_span = torch.linspace(0, 1, self.K_step + 1)
            _, traj = neural_ode(x0, t_span)
            x = traj[-1][:, 0].transpose(1, 2)  # (B,T,M)
            ret["mel_out"] = self.denorm_spec(x) 
            # * nonpadding.unsqueeze(-1).float()
            ret["flow"] = 0.0
        return ret

    # ---------- Diffusion / ODE 包装 ---------- #
    def ode_wrapper(self, cond, num_timesteps):
        return Wrapper(self, cond, num_timesteps)

    def ode_wrapper_cfg(self, cond, ucond, cfg_scale, num_timesteps):
        return Wrapper_CFG(self, cond, ucond, cfg_scale, num_timesteps)

    # ---------- 扩散核心 ---------- #
    @staticmethod
    def q_sample(x_start, t, noise, num_timesteps):
        t_frac = (
            t.unsqueeze(1).unsqueeze(1).unsqueeze(1).float() / num_timesteps
        )  # (B,1,1,1)
        return t_frac * x_start + (1.0 - t_frac) * noise

    def p_losses(self, x_start, t, cond, noise=None, nonpadding=None):
        if noise is None:
            noise = default(noise, lambda: torch.randn_like(x_start))

        xt = self.q_sample(x_start, t, noise, self.num_timesteps)
        v_pred = self.denoise_fn(xt, t, cond)
        ut = x_start - noise  # 真实残差

        if nonpadding is not None:
            # (B, T) -> (B,1,1,T)
            mask = nonpadding.unsqueeze(1).unsqueeze(2).float()
        # ---------- L1 / L2 ---------- #
        if self.loss_type == "l1":
            if nonpadding is None:
                loss = (ut - v_pred).abs().mean()
            else:
                loss = ((ut - v_pred).abs() * mask).sum() / (mask.sum() + 1e-8)
        elif self.loss_type == "l2":
            if nonpadding is None:
                loss = F.mse_loss(ut, v_pred)
            else:
                loss = (
                    F.mse_loss(ut, v_pred, reduction="none") * mask
                ).sum() / (mask.sum() + 1e-8)
        else:
            raise NotImplementedError(f"unknown loss type: {self.loss_type}")
        return loss

    # ---------- 归一化 / 反归一化 ---------- #
    def norm_spec(self, x):
        return (x - self.spec_min) / (self.spec_max - self.spec_min) * 2 - 1

    def denorm_spec(self, x):
        return (x + 1) / 2 * (self.spec_max - self.spec_min) + self.spec_min
