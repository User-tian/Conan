import os
import torch
import numpy as np
from typing import Dict, List
import json
import time
from datetime import datetime

from utils.commons.hparams import hparams, set_hparams
from utils.commons.ckpt_utils import load_ckpt, load_ckpt_emformer
from utils.audio import librosa_wav2spec
from utils.audio.io import save_wav
from tasks.tts.vocoder_infer.base_vocoder import get_vocoder_cls

from modules.Conan.Conan import Conan
# Emformer feature extractor
from modules.Emformer.emformer import EmformerDistillModel
__all__ = ["StreamingVoiceConversion"]

class StreamingVoiceConversion:
    """
    Streaming style-transfer inference using Emformer for feature extraction.
    """
    tokens_per_chunk: int = 4  # 4 HuBERT tokens ≈ 80 ms
    
    def __init__(self, hp: Dict):
        self.hparams = hp
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self._build_model()
        self.vocoder = self._build_vocoder()
        self.emformer = self._build_emformer()
        self._vocoder_warm_zero()

    def _build_model(self):
        m = Conan(0, self.hparams)
        m.eval()
        load_ckpt(m, self.hparams["work_dir"], strict=False)
        return m.to(self.device)

    def _build_vocoder(self):

        vocoder_cls = get_vocoder_cls(self.hparams["vocoder"])
        if vocoder_cls is None:
            raise ValueError(f"Vocoder '{self.hparams['vocoder']}' is not registered. Check vocoder name and registration.")
        return vocoder_cls()

    def _build_emformer(self):
        emformer = EmformerDistillModel(self.hparams, output_dim=100)
        # load checkpoint
        load_ckpt(emformer, self.hparams["emformer_ckpt"], strict=False)
        emformer.eval()
        return emformer.to(self.device)

    def _vocoder_warm_zero(self):
        _ = self.vocoder.spec2wav(np.zeros((4, 80), dtype=np.float32))
        
    @staticmethod
    def _wav_to_mel(path: str) -> np.ndarray:
        mel = librosa_wav2spec(
            path,
            fft_size=hparams["fft_size"],
            hop_size=hparams["hop_size"],
            win_length=hparams["win_size"],
            num_mels=hparams["audio_num_mel_bins"],
            fmin=hparams["fmin"],
            fmax=hparams["fmax"],
            sample_rate=hparams["audio_sample_rate"],
            loud_norm=hparams["loud_norm"],
        )["mel"]
        return np.clip(mel, hparams["mel_vmin"], hparams["mel_vmax"])

    def infer_once(self, inp: Dict, spk_emb=None):
        if spk_emb is not None:
            spk_emb = torch.from_numpy(spk_emb).float().to(self.device)
        # 1. Load reference mel
        ref_mel_np = self._wav_to_mel(inp["ref_wav"])
        ref_mel = torch.from_numpy(ref_mel_np).float().to(self.device)

        # 2. Load src mel
        src_mel_np = self._wav_to_mel(inp["src_wav"])
        src_mel = torch.from_numpy(src_mel_np).unsqueeze(0).to(self.device)  # [1, T, 80]
        total_frames = src_mel.shape[1]
        # 3. Streaming Emformer + RFSinger with proper state management
        chunk_size = self.hparams["chunk_size"] // 20  # frames per chunk (20ms per frame)
        right_context = self.hparams["right_context"]  # frames
        seg = chunk_size
        rc = right_context

        content_code_buffer = []  # list of [emit,] tensors
        mel_chunks = []
        wav_chunks = []
        prev_len = 0
        pos = 0
        state = None  # Emformer state for streaming
        

        while pos < total_frames:
            # 1) How many NEW frames do we want to emit this step?
            emit = min(seg, total_frames - pos)

            # 2) How much genuine look-ahead is still available?
            look = min(rc, total_frames - (pos + emit))
            
            # 3) Build the real chunk (emit + look) … then pad
            real_len = emit + look
            chunk = src_mel[:, pos:pos + real_len, :]  # (1, real_len, 80)

            # Pad so that len(chunk) == seg + rc, as Emformer expects
            need_pad = (seg + rc) - real_len
            if need_pad > 0:
                pad = chunk[:, -1:, :].expand(1, need_pad, src_mel.shape[2])  # repeat last frame
                chunk = torch.cat([chunk, pad], dim=1)  # (1, seg+rc, 80)

            # 4) Run one streaming step (length **includes** the right context)
            lengths = torch.full((1,), chunk.size(1), dtype=torch.long, device=self.device)
            with torch.no_grad():
                chunk_out, _, state = self.emformer.emformer.infer(chunk, lengths, state)
                # Apply projection if needed
                if self.emformer.mode == 'both':
                    chunk_out = self.emformer.proj1(chunk_out)
                else:
                    chunk_out = self.emformer.proj(chunk_out)
                
                # Convert to tokens if needed
                if chunk_out.dim() == 3 and chunk_out.shape[-1] > 1:
                    chunk_out = torch.argmax(chunk_out, dim=-1)  # [1, seg+rc]
                
            # Emformer drops the right context in its output; keep only `emit` frames
            new_codes = chunk_out[:, :emit]  # [1, emit]
            content_code_buffer.append(new_codes.squeeze(0))  # [emit]
            all_codes = torch.cat(content_code_buffer, dim=0).unsqueeze(0)  # [1, T_so_far]
            
            with torch.no_grad():
                out = self.model(
                    content=all_codes,
                    spk_embed=spk_emb,
                    target=None,
                    ref=None, # ref_mel.unsqueeze(0),
                    f0=None,
                    uv=None,
                    infer=True,
                    global_steps=200000,
                )
                mel_out = out["mel_out"][0]
            mel_new = mel_out[prev_len:]
            mel_chunks.append(mel_new)
            prev_len = mel_out.shape[0]
            pos += emit
            # collect mel from start to current pos
            mel_chunks_forvocoder = torch.cat(mel_chunks, dim=0)
            wav_chunk_vocoder = self.vocoder.spec2wav(mel_chunks_forvocoder.cpu().numpy())
            # only keep the wav generated for the current chunk
            hop = self.hparams["hop_size"]
            start_sample = max(0, (pos - emit) * hop)
            end_sample = min(len(wav_chunk_vocoder), pos * hop)
            if end_sample > start_sample:
                wav_chunk = wav_chunk_vocoder[start_sample:end_sample]
                wav_chunks.append(wav_chunk)
        mel_pred = torch.cat(mel_chunks, dim=0)
        if len(wav_chunks) > 0:
            wav_pred = np.concatenate(wav_chunks, axis=0)
        else:
            wav_pred = self.vocoder.spec2wav(mel_pred.cpu().numpy())
        # Vocoder whole sentence
        # wav_pred = self.vocoder.spec2wav(mel_pred.cpu().numpy())
        
        
        return wav_pred, mel_pred.cpu().numpy()

    def test_multiple_sentences(self, test_cases: List[Dict], spk_embs=None):
        os.makedirs("infer_out_demo", exist_ok=True)
        for i, inp in enumerate(test_cases):
            for j, emb in enumerate(spk_embs):
                wav, mel = self.infer_once(inp, emb)
                ref_name = os.path.splitext(os.path.basename(inp["ref_wav"]))[0]
                src_name = os.path.splitext(os.path.basename(inp["src_wav"]))[0]
                save_path = f"infer_out_demo/{j}_{src_name}.wav"
                save_wav(wav, save_path, self.hparams["audio_sample_rate"])
                print(f"Saved output: {save_path}")


if __name__ == "__main__":
    set_hparams()
    # Example usage: update with your own wav paths
    spk_embs = np.load("/storageSSD/huiran/src/NVAE-DarkStream/output/nvae_conan/emb.npy")
    demo = [
        # {"ref_wav": "/storageNVME/baotong/datasets/vctk-controlvc16k/wav16_silence_trimmed_padded/p226_005_mic2.wav", "src_wav": "/storageNVME/baotong/datasets/vctk-controlvc16k/wav16_silence_trimmed_padded/p236_005_mic2.wav"},
        {"ref_wav": "/storageNVME/baotong/datasets/vctk-controlvc16k/wav16_silence_trimmed_padded/p226_005_mic2.wav", "src_wav": "/storageNVME/baotong/datasets/vctk-controlvc16k/wav16_silence_trimmed_padded/p246_005_mic2.wav"},
    ]
    
    engine = StreamingVoiceConversion(hparams)
    engine.test_multiple_sentences(demo, spk_embs)
