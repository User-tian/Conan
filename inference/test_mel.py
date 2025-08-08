#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
绘制 Mel + F0，对齐帧数，并把横坐标统一为秒。
图片保存到当前目录，文件名 *_mel_f0.png。
"""

import os
import numpy as np
import librosa
import matplotlib.pyplot as plt

# ------------ 路径与参数 -------------
wav_path = "/home/zy/data/libritts/LibriTTS/test-clean/3570/5696/3570_5696_000004_000003.wav"
sr       = 16000
hop      = 320           # Mel 的 hop 就按题目要求 320
win      = 1024
fmin, fmax = 80, 7600
f0_min, f0_max = 50, 1000
# ------------------------------------

# ---------- 1. 读音频 ----------
y, _   = librosa.load(wav_path, sr=sr, mono=True)

# ---------- 2. 读 F0 ----------
f0_path = wav_path.replace(".wav", "_f0.npy")
f0      = np.load(f0_path).astype(np.float32)        # 613 帧
f0_len  = len(f0)

# ---------- 3. 计算 Mel ----------
mel = librosa.feature.melspectrogram(
    y=y, sr=sr, n_fft=win, win_length=win, hop_length=hop,
    fmin=fmin, fmax=fmax, power=2.0, center=False
)
mel_db = librosa.power_to_db(mel, ref=np.max)
mel_len = mel_db.shape[1]                            # 304 帧

# ---------- 4. 帧对齐 ----------
if f0_len != mel_len:
    # 线性插值把 F0 重新映射到 mel_len 帧
    f0_idx   = np.linspace(0, f0_len - 1, mel_len)
    f0_align = np.interp(f0_idx, np.arange(f0_len), np.nan_to_num(f0, nan=0.0))
else:
    f0_align = f0

# 过滤异常值
f0_align[(f0_align < f0_min) | (f0_align > f0_max)] = np.nan

# ---------- 5. 绘图 ----------
duration = len(y) / sr
times    = np.arange(mel_len) * hop / sr            # 与 Mel/F0 同步的时间轴

fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [4, 1]}
)

# ---- Mel（横坐标用秒）----
extent = [0, duration, 0, mel_db.shape[0]]          # xmin, xmax, ymin, ymax
img = ax1.imshow(mel_db, origin="lower", aspect="auto",
                 interpolation="nearest", cmap="magma", extent=extent)
ax1.set_xlim(0, duration)
ax1.set_ylabel("Mel bins")
ax1.set_title("Mel Spectrogram")
cbar = plt.colorbar(img, ax=ax1, format="%+2.0f dB")
cbar.set_label("dB")

# ---- F0 ----
ax2.plot(times, f0_align, linewidth=1.2)
ax2.set_xlim(0, duration)
ax2.set_ylim(f0_min, f0_max)
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("F0 (Hz)")
ax2.set_title("Fundamental Frequency (F0)")

plt.tight_layout()

# ---------- 6. 保存 ----------
out_png = os.path.basename(wav_path).replace(".wav", "_mel_f0.png")
plt.savefig(out_png, dpi=300)
plt.close()
print(f"✅ 图已保存到当前目录: {out_png}")

# ---------- 7. 额外排错信息 ----------
print(f"INFO | mel_len={mel_len}, f0_len={f0_len}, duration={duration:.3f}s")
