from inspect import isfunction
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from utils.commons.hparams import hparams
from torchdyn.core import NeuralODE
sigma = 1e-4

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()

class Wrapper(nn.Module):
    def __init__(self, net, cond, num_timesteps, dyn_clip):
        super(Wrapper, self).__init__()
        self.net = net
        self.cond = cond
        self.num_timesteps = num_timesteps
        self.dyn_clip = dyn_clip

    def forward(self, t, x, args):
        t = torch.tensor([t * self.num_timesteps], device=t.device).long()
        ut = self.net.denoise_fn(x, t, self.cond)
        if hparams['f0_sample_clip']:
            x_recon = (1 - t / self.num_timesteps) * ut + x
            if self.dyn_clip is not None:
                x_recon.clamp_(self.dyn_clip[0].unsqueeze(1), self.dyn_clip[1].unsqueeze(1))
            else:
                x_recon.clamp_(-1., 1.)
            ut = (x_recon - x) / (1 - t / self.num_timesteps)
        return ut

class ReflowF0(nn.Module):
    def __init__(self, out_dims, denoise_fn, timesteps=1000, f0_K_step=1000, loss_type='l1'):
        super().__init__()
        self.denoise_fn = denoise_fn
        self.mel_bins = out_dims
        self.K_step = f0_K_step
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

    def q_sample(self, x_start, t, noise=None):
        if noise==None:
            noise = default(noise, lambda: torch.randn_like(x_start))
        x1 = x_start
        x0 = noise
        t_unsqueeze = t.unsqueeze(1).unsqueeze(1).unsqueeze(1).float() / self.num_timesteps

        if hparams['flow_qsample'] == 'sig':
            epsilon = torch.randn_like(x0)
            xt = t_unsqueeze * x1 + (1. - t_unsqueeze) * x0 + sigma * epsilon
        else:
            xt = t_unsqueeze * x1 + (1. - t_unsqueeze) * x0 
        return xt

    def p_losses(self, x_start, t, cond, noise=None, nonpadding=None):
        if noise==None:
            noise = default(noise, lambda: torch.randn_like(x_start))
        xt = self.q_sample(x_start=x_start, t=t, noise=noise)
        x1 = x_start
        x0 = noise
        
        v_pred = self.denoise_fn(xt, t, cond)
        ut = x1 - x0 
        if self.loss_type == 'l1':
            if nonpadding is not None:
                loss = ((ut - v_pred).abs() * nonpadding.unsqueeze(1)).sum() / (nonpadding.unsqueeze(1) + 1e-8).sum()
            else:
                loss = ((ut - v_pred).abs()).mean()
        elif self.loss_type == 'l2':
            if nonpadding is not None:
                loss = (F.mse_loss(ut, v_pred,  reduction='none') * nonpadding.unsqueeze(1)).sum() / (nonpadding.unsqueeze(1) + 1e-8).sum()
            else:
                loss_simple = F.mse_loss(ut, v_pred,  reduction='none')
                loss = torch.mean(loss_simple)
        else:
            raise NotImplementedError()
        
        return loss

    def forward(self, cond, f0=None, nonpadding=None, ret=None, infer=False, dyn_clip=None, solver='euler', initial_noise=None):
        b = cond.shape[0]
        device = cond.device
        if not infer:
            # --- 训练部分 (保持不变) ---
            t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
            # 注意: 训练时 f0 可能是 [B, T], 需要 unsqueeze 成 [B, 1, 1, T] 或类似形状
            # 假设 f0 输入是 [B, T]
            if f0.ndim == 2:
                 # self.mel_bins 应该是 F0 的维度，通常是 1
                 x = f0.unsqueeze(1).unsqueeze(1) # -> [B, 1, 1, T]
                 if x.shape[2] != self.mel_bins:
                      # 如果 self.mel_bins 不是 1, 可能需要调整
                      print(f"警告: 训练 f0 形状调整可能需要。 x shape:{x.shape}, mel_bins:{self.mel_bins}")
            elif f0.ndim == 4: # 可能已经是 [B, 1, D, T]
                 x = f0
            else:
                 raise ValueError(f"ReflowF0 训练中未预期的 f0 ndim: {f0.ndim}")

            return self.p_losses(x, t, cond, nonpadding=nonpadding)
        else:
            # --- 推理部分 ---
            # 确定噪声形状：[批量大小, 通道数=1, F0维度=mel_bins, 时间步长=cond的时间步长]
            shape = (cond.shape[0], 1, self.mel_bins, cond.shape[2])

            # 如果提供了 initial_noise，则使用它
            if initial_noise is not None:
                # print("--- 正在使用提供的初始噪声 ---") # 调试打印
                x0 = initial_noise
                # 检查形状和设备是否匹配
                if x0.shape != shape:
                    raise ValueError(f"提供的 initial_noise 形状 {x0.shape} 与期望形状 {shape} 不匹配")
                if x0.device != device:
                    x0 = x0.to(device) # 移动到正确的设备
            else:
                # print("--- 正在生成新的初始噪声 ---") # 调试打印
                x0 = torch.randn(shape, device=device) # 否则生成新的随机噪声

            # === 重要: 在 ret 字典中存储实际使用的噪声 ===
            if ret is not None:
                # 存储一个分离的克隆，防止后续计算修改它
                ret['initial_noise_used'] = x0.detach().clone()
            # ===

            # 创建 NeuralODE 实例
            neural_ode = NeuralODE(self.ode_wrapper(cond, self.num_timesteps, dyn_clip), solver=solver, sensitivity="adjoint", atol=1e-4, rtol=1e-4)
            # 定义积分时间范围 [0, 1]，步数为 K_step
            t_span = torch.linspace(0, 1, self.K_step + 1, device=device) # 确保 t_span 在正确的设备上
            # 执行 ODE 求解，确保 x0 在正确的设备上 (上面已处理)
            eval_points, traj = neural_ode(x0, t_span)
            # 获取最终在 t=1 时的状态作为预测结果
            x = traj[-1]
            # 调整输出形状，假设 F0 维度是 1，期望输出是 [B, T]
            x = x[:, 0, 0, :] # 从 [B, 1, 1, T] 提取 -> [B, T]
            # 如果你的模型期望 F0 输出是 [B, T, 1]，则使用下面这行:
            # x = x[:, 0].transpose(1, 2) # 从 [B, 1, mel_bins, T] 调整 -> [B, T, mel_bins]

        return x # 返回预测的 F0

    def ode_wrapper(self, cond, num_timesteps, dyn_clip):
        return Wrapper(self, cond, num_timesteps, dyn_clip)