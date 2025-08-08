import os
import json
import time
from typing import Dict, List, Tuple

import torch
import numpy as np

from utils.commons.hparams import hparams, set_hparams
from utils.commons.ckpt_utils import load_ckpt
from utils.audio import librosa_wav2spec
from utils.audio.io import save_wav
from tasks.tts.vocoder_infer.base_vocoder import get_vocoder_cls

# —— TechSinger acoustic model ——
from modules.TechSinger.techsinger import RFSinger  # pylint: disable=import-error

__all__ = ["StyleTransferStreaming"]

# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def _sec_from_frames(n_frames: int) -> float:
    return n_frames * hparams["hop_size"] / hparams["audio_sample_rate"]


def _ms(sec: float) -> float:
    return sec * 1000.0

# -----------------------------------------------------------------------------
# Main class
# -----------------------------------------------------------------------------

class StyleTransferStreaming:
    """Streaming style‑transfer inference with **on‑the‑fly warm‑up**:

    * Model warm‑up:   real `ref_mel` + dummy `content=71`.
    * Vocoder warm‑up: zero 4‑frame mel.
    * Per‑chunk latency / RTF for both model & vocoder.
    """

    tokens_per_chunk: int = 4  # 4 HuBERT tokens ≈ 80 ms

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------
    def __init__(self, hp: Dict):
        self.hparams = hp
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self._build_model()
        self.vocoder = self._build_vocoder()
        # vocoder can be warmed once globally (zero‑mel)
        self._vocoder_warm_zero()

    # ------------------------------------------------------------------
    # Builders
    # ------------------------------------------------------------------
    def _build_model(self):
        m = RFSinger(0, self.hparams)
        m.eval()
        load_ckpt(m, self.hparams["work_dir"], strict=False)
        return m.to(self.device)

    def _build_vocoder(self):
        return get_vocoder_cls(self.hparams["vocoder"])()

    # ------------------------------------------------------------------
    # Vocoder warm‑up (zero mel)
    # ------------------------------------------------------------------
    def _vocoder_warm_zero(self):
        _ = self.vocoder.spec2wav(np.zeros((4, 80), dtype=np.float32))

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_hubert_string(s: str) -> List[int]:
        return [int(tok) for tok in s.strip().split() if tok.strip()]

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

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def infer_once(self, inp: Dict) -> Tuple[np.ndarray, np.ndarray]:
        meta = json.load(open("/storage/baotong/workspace/streamvc/data/processed/vc/metadata_6layer.json"))

        def _find(k):
            for it in meta:
                if k in it["item_name"]:
                    return it
            raise RuntimeError(f"{k} not found in metadata.json")

        ref_item, gen_item = _find(inp["ref_name"]), _find(inp["gen_name"])
        ref_mel_np = self._wav_to_mel(ref_item["wav_fn"])
        __import__("ipdb").set_trace()
        content_ids = self._parse_hubert_string(gen_item["hubert"])

        content_full = torch.LongTensor(content_ids).to(self.device)
        ref_mel = torch.from_numpy(ref_mel_np).float().to(self.device)

        # ------------------ model warm‑up (real ref, dummy content) ------------------
        # with torch.no_grad():
        #     dummy_content = torch.full(
        #         (1, self.tokens_per_chunk), 71, dtype=torch.long, device=self.device
        #     )
        #     _ = self.model(
        #         content=dummy_content,
        #         spk_embed=None,
        #         target=None,
        #         ref=ref_mel.unsqueeze(0),
        #         f0=None,
        #         uv=None,
        #         infer=True,
        #         global_steps=200000,
        #     )

        # ------------------ streaming loop ------------------
        total_chunks = (len(content_ids) + self.tokens_per_chunk - 1) // self.tokens_per_chunk
        mel_chunks: List[torch.Tensor] = []
        wav_chunks: List[np.ndarray] = []
        prev_len = 0
        model_tot_s = 0.0
        voc_tot_s = 0.0

        for idx in range(total_chunks):
            end = min((idx + 1) * self.tokens_per_chunk, len(content_ids))
            chunk_content = content_full[:end].unsqueeze(0)

            # model
            t0 = time.time()
            with torch.no_grad():
                out = self.model(
                    content=chunk_content,
                    spk_embed=None,
                    target=None,
                    ref=ref_mel.unsqueeze(0),
                    f0=None,
                    uv=None,
                    infer=True,
                    global_steps=200000,
                )
                mel_out = out["mel_out"][0]
            model_t = time.time() - t0
            model_tot_s += model_t

            
            mel_new = mel_out[prev_len:]
            mel_chunks.append(mel_new)
            prev_len = mel_out.shape[0]

            # # vocoder per‑chunk
            # t0 = time.time()
            # wav_new = self.vocoder.spec2wav(mel_new.cpu().numpy())
            # voc_t = time.time() - t0
            # voc_tot_s += voc_t
            # wav_chunks.append(wav_new)

            # sec = _sec_from_frames(mel_new.shape[0]) or 1e-9
            # print(
            #     f"Chunk {idx+1}/{total_chunks}: Model {_ms(model_t):.1f} ms (RTF {model_t/sec:.2f}) | "
            #     f"Vocoder {_ms(voc_t):.1f} ms (RTF {voc_t/sec:.2f})"
            # )
        # vocoder whole sentence
        mel_pred = torch.cat(mel_chunks, dim=0)
        wav_pred = self.vocoder.spec2wav(mel_pred.cpu().numpy())
        # mel_pred = torch.cat(mel_chunks, 0).cpu().numpy()
        # wav_pred = np.concatenate(wav_chunks)
        # total_sec = _sec_from_frames(mel_pred.shape[0])
        # print(
        #     f"\nTotal: Model {_ms(model_tot_s):.1f} ms (RTF {model_tot_s/total_sec:.2f}) | "
        #     f"Vocoder {_ms(voc_tot_s):.1f} ms (RTF {voc_tot_s/total_sec:.2f}) | Audio {total_sec:.2f} s"
        # )
        return wav_pred, mel_pred
    
    def test_multiple_sentences(self, test_cases: List[Dict]):
        """测试多个句子并生成报告"""
        # print(f"\n{'='*50}")
        # print(f"Starting multi-sentence test with {len(test_cases)} sentences")
        # print(f"{'='*50}\n")
        
        os.makedirs("infer_out_demo", exist_ok=True)
        
        # 处理所有句子
        for i, inp in enumerate(test_cases):
            wav, mel = self.infer_once(inp)
            save_path = f"infer_out_demo/{inp['ref_name']}_{inp['gen_name']}.wav"
            save_wav(wav, save_path, self.hparams["audio_sample_rate"])
            # print(f"Saved output: {save_path}")

# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    set_hparams()
    # demo = [
    #     {"ref_name": "p231_001_002_003", "gen_name": "1089_134686_000001_000001"},
    #     {"ref_name": "p231_001_002_003", "gen_name": "1188_133604_000004_000005"},
    #     {"ref_name": "p231_001_002_003", "gen_name": "1221_135766_000026_000005"},
    #     {"ref_name": "p231_001_002_003", "gen_name": "1284_1180_000005_000001"},
    #     {"ref_name": "p231_001_002_003", "gen_name": "1580_141083_000006_000002"},
    #     {"ref_name": "p231_001_002_003", "gen_name": "1995_1826_000005_000002"},
    #     {"ref_name": "p231_001_002_003", "gen_name": "2300_131720_000002_000003"},
    #     {"ref_name": "p231_001_002_003", "gen_name": "237_126133_000004_000000"},
    #     {"ref_name": "p231_001_002_003", "gen_name": "260_123286_000005_000001"},
    #     {"ref_name": "p231_001_002_003", "gen_name": "3729_6852_000004_000003"}
    # ]
    demo = [
        {"ref_name": "p236_005_mic2", "gen_name": "p360_057_mic2"},
        {"ref_name": "p303_025_mic2", "gen_name": "p360_057_mic2"},
        {"ref_name": "p339_345_mic2", "gen_name": "p360_057_mic2"},
    ]
    engine = StyleTransferStreaming(hparams)
    # wav, mel = engine.infer_once(demo)
    # os.makedirs("infer_out", exist_ok=True)
    # save_wav(wav, "infer_out/transfer_stream.wav", hparams["audio_sample_rate"])
    engine.test_multiple_sentences(demo)
    