
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import List, Tuple

import torch
import torchaudio
from transformers import HubertForCTC, Wav2Vec2Processor
from jiwer import wer, cer

# -------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------

def find_files_by_tag(folder: Path, tag: str) -> List[Path]:
    """Return all *.wav files in *folder* whose names contain *tag*."""
    return [
        p for p in folder.glob("*.wav") if tag in p.name
    ]


def sort_by_basename_key(files: List[Path], tag: str) -> List[Path]:
    """Sort files by their basename with *tag* removed so that matching pairs align."""
    key_func = lambda p: re.sub(re.escape(tag), "", p.stem)
    return sorted(files, key=key_func)


def load_audio(path: Path, target_sr: int = 16_000) -> torch.Tensor:
    """Load audio as mono, 16 kHz, float‐32 tensor on CPU."""
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:  # Convert to mono by averaging channels
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.squeeze(0)


# -------------------------------------------------------------
# ASR wrapper
# -------------------------------------------------------------

def transcribe_batch(model, processor, waveforms: List[torch.Tensor], device: torch.device) -> List[str]:
    """Run ASR on a list of waveforms and return decoded strings."""
    inputs = processor(waveforms, sampling_rate=16_000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(inputs.input_values.to(device), attention_mask=inputs.attention_mask.to(device)).logits
    pred_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(pred_ids, skip_special_tokens=True)


# -------------------------------------------------------------
# Main logic
# -------------------------------------------------------------

def compute_metrics(folder: Path) -> Tuple[float, float]:
    p_files = find_files_by_tag(folder, "[P]")
    g_files = find_files_by_tag(folder, "[G]")

    if not p_files:
        raise FileNotFoundError(f"在目录 `{folder}` 中未找到任何包含 \"[P]\" 的文件。")
    if not g_files:
        raise FileNotFoundError(f"在目录 `{folder}` 中未找到任何包含 \"[G]\" 的文件。")

    p_files_sorted = sort_by_basename_key(p_files, "[P]")
    g_files_sorted = sort_by_basename_key(g_files, "[G]")

    if len(p_files_sorted) != len(g_files_sorted):
        raise ValueError("[P] 与 [G] 文件数量不一致，无法一一对应。")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
    model = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft").to(device).eval()

    reference_texts, hypothesis_texts = [], []

    # — Inference in small batches to balance speed & memory —
    batch_size = 4
    for idx in range(0, len(p_files_sorted), batch_size):
        batch_p = p_files_sorted[idx: idx + batch_size]
        batch_g = g_files_sorted[idx: idx + batch_size]

        wav_ps = [load_audio(p) for p in batch_p]
        wav_gs = [load_audio(g) for g in batch_g]

        hyps = transcribe_batch(model, processor, wav_ps, device)
        refs = transcribe_batch(model, processor, wav_gs, device)

        hypothesis_texts.extend(hyps)
        reference_texts.extend(refs)

    # Concatenate all references and hypotheses to compute corpus‑level metrics
    ref_corpus = " ".join(reference_texts)
    hyp_corpus = " ".join(hypothesis_texts)

    wer_score = wer(ref_corpus, hyp_corpus)
    cer_score = cer(ref_corpus, hyp_corpus)
    return wer_score, cer_score


# -------------------------------------------------------------
# CLI entry point
# -------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compute WER & CER between matched [G] (reference) and [P] (predicted) WAV files using HuBERT-Large ASR.")
    parser.add_argument("--folder", type=Path, required=True, help="包含 [G]/[P] WAV 文件的目录")
    args = parser.parse_args()

    wer_score, cer_score = compute_metrics(args.folder)
    print(f"WER: {wer_score * 100:.2f}%")
    print(f"CER: {cer_score * 100:.2f}%")


if __name__ == "__main__":
    main()
