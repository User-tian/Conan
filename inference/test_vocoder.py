from tasks.tts.vocoder_infer.base_vocoder import get_vocoder_cls
from tasks.tts.vocoder_infer.base_vocoder import BaseVocoder
from utils.commons.hparams import set_hparams,hparams
import numpy as np
import torch
import soundfile
import torch.nn.functional as F
import math
import torchaudio


def wav2spec(wav_fn, return_linear=False):
    from utils.audio import librosa_wav2spec
    wav_spec_dict = librosa_wav2spec(wav_fn, fft_size=hparams['fft_size'],
                                        hop_size=hparams['hop_size'],
                                        win_length=hparams['win_size'],
                                        num_mels=hparams['audio_num_mel_bins'],
                                        fmin=hparams['fmin'],
                                        fmax=hparams['fmax'],
                                        sample_rate=hparams['audio_sample_rate'],
                                        loud_norm=hparams['loud_norm'])
    wav = wav_spec_dict['wav']
    mel = wav_spec_dict['mel']
    return wav, mel

def synthesize_segment(mel, step=5, start=10):
    """分段合成函数"""
    # 分块处理参数
    t=0

    output_segments = []

    # 逐个处理每个块
    while t<mel.shape[0]:
        if t<start:
            chunk=mel[:t+step+step]
            y_segment = vocoder.spec2wav(chunk)  # [1, 1, samples]
            # y_segment = y_segment[:, 0]  
            y_segment = y_segment[(t)*hparams['hop_size']:(t+step)*hparams['hop_size']]
            t+=step
        else:
            chunk=mel[t-start:t+step+step]
            y_segment = vocoder.spec2wav(chunk)  # [1, 1, samples]
            # y_segment = y_segment[:, 0]  
            y_segment = y_segment[(start)*hparams['hop_size']:(start+step)*hparams['hop_size']]
            t+=step
        output_segments.append(y_segment)

    # 拼接所有音频段
    y = np.concatenate(output_segments)
    return y

def synthesize_with_cola(mel, total_samples, frame_length_samples, frame_step_samples):
    hop = hparams['hop_size']
    # √Hann 窗
    synth_win = torch.sqrt(torch.hann_window(frame_length_samples)).to(mel.device)

    output     = torch.zeros(total_samples).to(mel.device)
    sum_window = torch.zeros_like(output)

    T, C = mel.shape

    for start in range(0, total_samples - frame_length_samples + 1, frame_step_samples):
        end_frame = (start + frame_length_samples) // hop + 1

        # 用全部历史上下文
        if end_frame <= T:
            mel_input = mel[:end_frame]
        else:
            pad_n    = end_frame - T
            last_row = mel[-1:].expand(pad_n, C)
            mel_input = torch.cat([mel, last_row], dim=0)

        # 边缘用 reflect pad（更平滑）
        pad_l = max(0, - (start // hop))
        pad_r = max(0, end_frame - T)
        if pad_l or pad_r:
            mel_input = torch.nn.functional.pad(
                mel_input.transpose(0,1),
                (pad_l, pad_r),
                mode='reflect'
            ).transpose(0,1)

        # 传 2D mel [frames, C] 给 spec2wav，拿回 1D waveform
        audio_full = torch.FloatTensor(vocoder.spec2wav(mel_input))

        # 切当前窗对应的 segment
        seg_start = end_frame * hop - frame_length_samples
        seg = audio_full[seg_start : seg_start + frame_length_samples]

        # 加窗叠加
        wseg = seg * synth_win
        s, e = start, start + frame_length_samples
        output[s:e]     += wseg
        sum_window[s:e] += synth_win
        
    nonzero = sum_window > 1e-8     # 避免除 0
    output[nonzero] /= sum_window[nonzero]

    # 可选：防削波，再做一个软限制
    output = torch.clamp(output, -1.0, 1.0)
    output = torchaudio.functional.highpass_biquad(output, hparams['audio_sample_rate'], cutoff_freq=20)

    return output

def synthesize_with_turkey(mel, total_samples, frame_length_samples, frame_step_samples):
    """COLA 合成——Tukey 窗 (前后25% fade in/out，中间50% flat)"""
    hop = hparams['hop_size']
    T, C = mel.shape

    # 生成 Tukey 窗，alpha=0.5
    N = frame_length_samples
    n = torch.arange(N, device=mel.device, dtype=torch.float32)
    alpha = 0.25
    edge = alpha * (N - 1) / 2
    window = torch.ones(N, device=mel.device, dtype=torch.float32)
    # 前 25% 渐入
    idx1 = n < edge
    window[idx1] = 0.5 * (1 - torch.cos(math.pi * n[idx1] / edge))
    # 后 25% 渐出
    idx2 = n > (N - 1 - edge)
    window[idx2] = 0.5 * (1 - torch.cos(math.pi * (N - 1 - n[idx2]) / edge))

    output     = torch.zeros(total_samples, device=mel.device)
    sum_window = torch.zeros_like(output)

    for start in range(0, total_samples - N + 1, frame_step_samples):
        # 取到当前窗结束帧（全部历史上下文）
        end_frame = (start + N) // hop + 1
        if end_frame <= T:
            mel_input = mel[:end_frame]
        else:
            pad_cnt  = end_frame - T
            last_row = mel[-1:].expand(pad_cnt, C)
            mel_input = torch.cat([mel, last_row], dim=0)

        # 只对 time 轴做平滑反射填充，以防边界突变
        pad_l = max(0, - (start // hop))
        pad_r = max(0, end_frame - T)
        if pad_l or pad_r:
            mel_input = F.pad(
                mel_input.transpose(0,1),
                (pad_l, pad_r),
                mode='reflect'
            ).transpose(0,1)

        # 生成完整到 end_frame 的波形，然后截取本窗口段
        audio_full = torch.FloatTensor(vocoder.spec2wav(mel_input))
        seg_start  = end_frame * hop - N
        seg = audio_full[seg_start : seg_start + N]

        # 重叠加权
        wseg = seg * window
        output[start:start+N]     += wseg
        sum_window[start:start+N] += window

    # 归一化
    return output / (sum_window + 1e-8)

config_path = f'/home/zy/VC/egs/stage1.yaml'
hparams = set_hparams(config_path)
vocoder=get_vocoder_cls(hparams["vocoder"])()

wav_fn='/home/zy/data/libritts/LibriTTS/dev-clean/84/121123/84_121123_000007_000001.wav'
wav, mel = wav2spec(wav_fn)

mel=torch.FloatTensor(mel)

# y = synthesize_segment(mel, step=8, start=50)

# assert False,f'{mel.shape}'
# print(wav.shape, mel.shape)

# mel = mel.transpose(1, 0)

import time

start_time = time.time()
y = vocoder.spec2wav(mel)
end_time = time.time()
infer_time = end_time - start_time

output_duration_sec = len(y) / hparams['audio_sample_rate']
rtf_value = infer_time / output_duration_sec

print(f'Vocoder Inference time: {infer_time:.2f} seconds, RTF: {rtf_value:.2f},output duration: {output_duration_sec:.2f} seconds')

# assert False,f'{y.shape}'

# frame_length_frames = 8
# frame_length_samples = frame_length_frames * hparams['hop_size']
# frame_step_samples = frame_length_samples *7//8

# y = synthesize_with_cola(mel, mel.shape[0] * hparams['hop_size'], frame_length_samples, frame_step_samples)

# y = synthesize_with_turkey(mel, mel.shape[0] * hparams['hop_size'], frame_length_samples, frame_step_samples)

soundfile.write('single_channel.wav', y, 16000, 'PCM_16')