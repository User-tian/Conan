import os
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
import json
import time
from datetime import datetime

from utils.commons.hparams import hparams, set_hparams
from utils.commons.ckpt_utils import load_ckpt, load_ckpt_emformer
from utils.audio import librosa_wav2spec
from utils.audio.io import save_wav
from tasks.tts.vocoder_infer.base_vocoder import get_vocoder_cls

# TechSinger acoustic model
from modules.Conan.Conan import Conan
# Emformer feature extractor
from modules.Emformer.emformer import EmformerDistillModel

__all__ = ["RealtimeVoiceConversion"]

class RealtimeVoiceConversion:
    """
    Real-time streaming voice conversion optimized for low-latency processing.
    Decomposes the original inference pipeline into initialization and streaming steps.
    """
    
    def __init__(self, hp: Dict):
        self.hparams = hp
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self._build_model()
        self.vocoder = self._build_vocoder()
        self.emformer = self._build_emformer()
        self._vocoder_warm_zero()
        
        # Streaming parameters
        self.chunk_size = self.hparams["chunk_size"] // 20  # frames per chunk (20ms per frame)
        self.right_context = self.hparams["right_context"]  # frames
        self.hop_size = self.hparams["hop_size"]
        
        # Reference audio state
        self.ref_mel = None
        self.is_reference_set = False
        
        # Streaming state
        self.reset_streaming_state()

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

    def set_reference_audio(self, ref_wav_path: str):
        """
        Initialize reference audio for voice conversion.
        This should be called once when the reference audio is uploaded.
        
        Args:
            ref_wav_path: Path to the reference audio file
        """
        try:
            # Load and process reference mel
            ref_mel_np = self._wav_to_mel(ref_wav_path)
            self.ref_mel = torch.from_numpy(ref_mel_np).float().to(self.device)
            self.is_reference_set = True
            print(f"Reference audio set successfully: {ref_mel_np.shape}")
        except Exception as e:
            self.is_reference_set = False
            raise ValueError(f"Failed to set reference audio: {str(e)}")
    
    def reset_streaming_state(self):
        """Reset streaming state for a new conversion session."""
        self.content_code_buffer = []  # list of [emit,] tensors
        self.mel_chunks = []
        self.prev_len = 0
        self.pos = 0
        self.emformer_state = None  # Emformer state for streaming
        
    def infer_chunk(self, src_mel_chunk: torch.Tensor, 
                   emformer_state=None):
        """
        Process a single chunk of source audio for real-time conversion.
        
        Args:
            src_mel_chunk: Source mel spectrogram chunk [1, seg+rc, 80] (padded if needed)
            emformer_state: Previous emformer state (None for first chunk)
            
        Returns:
            Tuple of (wav_chunk, new_emformer_state)
        """
        if not self.is_reference_set:
            raise ValueError("Reference audio not set. Call set_reference_audio() first.")
        
        # Determine how many frames to emit this step (always chunk_size for streaming)
        emit = self.chunk_size
        
        # Run Emformer streaming step
        lengths = torch.full((1,), src_mel_chunk.size(1), dtype=torch.long, device=self.device)
        with torch.no_grad():
            chunk_out, _, new_emformer_state = self.emformer.emformer.infer(
                src_mel_chunk, lengths, emformer_state
            )
            
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
        self.content_code_buffer.append(new_codes.squeeze(0))  # [emit]
        all_codes = torch.cat(self.content_code_buffer, dim=0).unsqueeze(0)  # [1, T_so_far]
        
        # Run Conan model
        with torch.no_grad():
            out = self.model(
                content=all_codes,
                spk_embed=None,
                target=None,
                ref=self.ref_mel.unsqueeze(0),
                f0=None,
                uv=None,
                infer=True,
                global_steps=200000,
            )
            mel_out = out["mel_out"][0]
        
        # Get new mel frames
        mel_new = mel_out[self.prev_len:]
        self.mel_chunks.append(mel_new)
        self.prev_len = mel_out.shape[0]
        
        # Generate wav for current chunk using incremental vocoding
        mel_chunks_forvocoder = torch.cat(self.mel_chunks, dim=0)
        wav_full = self.vocoder.spec2wav(mel_chunks_forvocoder.cpu().numpy())
        
        # Extract only the wav corresponding to current chunk
        start_sample = max(0, self.pos * self.hop_size)
        end_sample = min(len(wav_full), (self.pos + emit) * self.hop_size)
        
        if end_sample > start_sample:
            wav_chunk = wav_full[start_sample:end_sample]
        else:
            wav_chunk = np.array([])
        
        # Update position
        self.pos += emit
        
        return wav_chunk, new_emformer_state
    
    def prepare_chunk(self, src_mel: torch.Tensor, pos: int, total_frames: int) -> torch.Tensor:
        """
        Prepare a chunk with proper padding for streaming inference.
        
        Args:
            src_mel: Complete source mel spectrogram [1, T, 80]
            pos: Current position in frames
            total_frames: Total frames in the audio
            
        Returns:
            Prepared chunk [1, seg+rc, 80] with padding if needed
        """
        seg = self.chunk_size
        rc = self.right_context
        
        # How many NEW frames do we want to emit this step?
        emit = min(seg, total_frames - pos)
        
        # How much genuine look-ahead is still available?
        look = min(rc, total_frames - (pos + emit))
        
        # Build the real chunk (emit + look)
        real_len = emit + look
        chunk = src_mel[:, pos:pos + real_len, :]  # (1, real_len, 80)
        
        # Pad so that len(chunk) == seg + rc, as Emformer expects
        need_pad = (seg + rc) - real_len
        if need_pad > 0:
            pad = chunk[:, -1:, :].expand(1, need_pad, src_mel.shape[2])  # repeat last frame
            chunk = torch.cat([chunk, pad], dim=1)  # (1, seg+rc, 80)
        
        return chunk
    
    def infer_streaming(self, src_wav_path: str) -> np.ndarray:
        """
        Complete streaming inference for testing purposes.
        In real-time usage, use prepare_chunk() and infer_chunk() separately.
        
        Args:
            src_wav_path: Path to source audio file
            
        Returns:
            Complete converted audio
        """
        if not self.is_reference_set:
            raise ValueError("Reference audio not set. Call set_reference_audio() first.")
        
        # Reset streaming state
        self.reset_streaming_state()
        
        # Load source audio
        src_mel_np = self._wav_to_mel(src_wav_path)
        src_mel = torch.from_numpy(src_mel_np).unsqueeze(0).to(self.device)  # [1, T, 80]
        total_frames = src_mel.shape[1]
        
        wav_chunks = []
        emformer_state = None
        
        while self.pos < total_frames:
            # Prepare chunk
            chunk = self.prepare_chunk(src_mel, self.pos, total_frames)
            
            # Process chunk
            wav_chunk, emformer_state = self.infer_chunk(
                chunk, emformer_state
            )
            
            if len(wav_chunk) > 0:
                wav_chunks.append(wav_chunk)
            
            # The streaming demo handles when to stop processing
            # if is_final:
            #     break
        
        # Concatenate all chunks
        if len(wav_chunks) > 0:
            wav_pred = np.concatenate(wav_chunks, axis=0)
        else:
            # Fallback: use complete mel for vocoding
            complete_mel = torch.cat(self.mel_chunks, dim=0)
            wav_pred = self.vocoder.spec2wav(complete_mel.cpu().numpy())
        
        return wav_pred
    
    def test_streaming(self, test_cases: List[Dict]):
        """Test the streaming inference with multiple cases."""
        os.makedirs("infer_out_realtime", exist_ok=True)
        
        for i, inp in enumerate(test_cases):
            # Set reference
            self.set_reference_audio(inp["ref_wav"])
            
            # Run streaming inference
            wav = self.infer_streaming(inp["src_wav"])
            
            # Save result
            ref_name = os.path.splitext(os.path.basename(inp["ref_wav"]))[0]
            src_name = os.path.splitext(os.path.basename(inp["src_wav"]))[0]
            save_path = f"infer_out_realtime/{ref_name}_{src_name}_realtime.wav"
            save_wav(wav, save_path, self.hparams["audio_sample_rate"])
            print(f"Saved streaming output: {save_path}")


if __name__ == "__main__":
    set_hparams()
    # Example usage
    demo = [
        {"ref_wav": "/storageNVME/baotong/datasets/vctk-controlvc16k/wav16_silence_trimmed_padded/p226_005_mic2.wav", 
         "src_wav": "/storageNVME/baotong/datasets/vctk-controlvc16k/wav16_silence_trimmed_padded/p246_005_mic2.wav"},
    ]
    
    engine = RealtimeVoiceConversion(hparams)
    engine.test_streaming(demo) 