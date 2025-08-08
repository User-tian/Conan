import random
import numpy as np
import torch
from utils.commons.dataset_utils import BaseDataset, collate_1d_or_2d
from utils.commons.indexed_datasets import IndexedDataset
from utils.audio.pitch.utils import norm_interp_f0


class BaseSpeechDataset(BaseDataset):
    """Dataset that always draws a *reference mel* from the same speaker via
    ``spk_id`` while keeping the original public interfaces intact (notably
    ``_get_item``).

    * `spk_id` is **always** required in the binary data for reference sampling.
    * Whether the ID is exposed to the model remains controlled by
      `hparams['use_spk_id']`.
    * ``IndexedDataset`` is lazily opened inside the worker to prevent pickling
      errors.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(self, prefix, shuffle=False, items=None, data_dir=None):
        super().__init__(shuffle)
        from utils.commons.hparams import hparams
        self.hparams = hparams
        self.data_dir = hparams['binary_data_dir'] if data_dir is None else data_dir
        self.prefix = prefix
        self.indexed_ds = None  # ⚠️ lazy open in worker

        # ------------------------------------------------------------------
        # Indices & utterance lengths
        # ------------------------------------------------------------------
        if items is not None:
            self.indexed_ds = items
            self.sizes = [1] * len(items)
            self.avail_idxs = list(range(len(self.sizes)))
        else:
            self.sizes = np.load(f"{self.data_dir}/{self.prefix}_lengths.npy")
            if prefix == 'test' and len(hparams['test_ids']) > 0:
                self.avail_idxs = hparams['test_ids']
            else:
                self.avail_idxs = list(range(len(self.sizes)))
            if prefix == 'train' and hparams['min_frames'] > 0:
                self.avail_idxs = [i for i in self.avail_idxs if self.sizes[i] >= hparams['min_frames']]
            self.sizes = [self.sizes[i] for i in self.avail_idxs]

        # Speaker map will be built lazily in each worker
        self.spk2indices = None  # type: dict[int, list[int]] | None
        self._spk_map_ready = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _open_indexed_ds_if_needed(self):
        """Open the mmap dataset after the worker process starts."""
        if self.indexed_ds is None:
            self.indexed_ds = IndexedDataset(f"{self.data_dir}/{self.prefix}")

    def _get_item(self, local_idx):
        """**Public** helper – keep original signature & semantics.
        Accepts *local* index (0 ≤ idx < len(avail_idxs)).
        """
        global_idx = self.avail_idxs[local_idx] if self.avail_idxs is not None else local_idx
        self._open_indexed_ds_if_needed()
        return self.indexed_ds[global_idx]

    def _build_speaker_map(self):
        """
        构建  spk_id → [local_idx]  的映射，但**绝不再去磁盘 mmap 拿整条样本**。

        - 如果存在 {prefix}_spk_ids.npy，则 O(N) 纯内存完成；
        - 否则回退到旧实现（逐条 _get_item）。
        - 每个说话人至多保留 hparams['max_samples_per_spk'] 条（默认 100）。
        """
        if self._spk_map_ready:
            return

        import os
        from collections import defaultdict

        max_per_spk = int(self.hparams.get('max_samples_per_spk', 100))
        spk_ids_path = f"{self.data_dir}/{self.prefix}_spk_ids.npy"
        self.spk2indices = defaultdict(list)

        if os.path.exists(spk_ids_path):
            # ---------- 快速路径 ----------
            # mmap 方式载入，几乎不占内存、速度秒级
            spk_ids = np.load(spk_ids_path, mmap_mode='r')
            # 只取当前 avail_idxs 子集
            local_spk_ids = spk_ids[self.avail_idxs]

            # 打乱，截断时不偏向前段
            for local_idx in np.random.permutation(len(local_spk_ids)):
                sid = int(local_spk_ids[local_idx])
                bucket = self.spk2indices[sid]
                if len(bucket) < max_per_spk:
                    bucket.append(local_idx)
        else:
            # ---------- 兼容旧数据的慢路径 ----------
            for local_idx in np.random.permutation(len(self.avail_idxs)):
                sid = int(self._get_item(local_idx)['spk_id'])
                bucket = self.spk2indices[sid]
                if len(bucket) < max_per_spk:
                    bucket.append(local_idx)

        self._spk_map_ready = True
    # ------------------------------------------------------------------
    # Dataset API
    # ------------------------------------------------------------------
    def __getitem__(self, index):
        self._build_speaker_map()
        item = self._get_item(index)
        hparams = self.hparams

        # 1) Main mel
        max_frames = hparams['max_frames']
        spec = torch.tensor(item['mel'])[:max_frames]
        max_frames = spec.shape[0] // hparams['frames_multiple'] * hparams['frames_multiple']
        spec = spec[:max_frames]

        # 2) Reference mel from same speaker
        spk_id = int(item['spk_id'])
        cand_locals = self.spk2indices[spk_id]
        ref_local = random.choice([l for l in cand_locals if l != index]) if len(cand_locals) > 1 else index
        ref_item = self._get_item(ref_local)
        ref_spec = torch.tensor(ref_item['mel'])[:hparams['max_frames']]
        ref_spec = ref_spec[: (ref_spec.shape[0] // hparams['frames_multiple']) * hparams['frames_multiple']]

        sample = {
            'id': index,
            'item_name': item['item_name'],
            'mel': spec,
            'mel_nonpadding': spec.abs().sum(-1) > 0,
            'ref_mel': ref_spec,
        }

        # Optional speaker embedding
        if hparams.get('use_spk_embed', False):
            embed = item['spk_embed']
            if isinstance(embed, str):
                embed = torch.tensor([float(x) for x in embed.split()])
            else:
                embed = torch.tensor(embed, dtype=torch.float32)
            sample['spk_embed'] = embed

        # Provide spk_id to model only if enabled
        if hparams.get('use_spk_id', False):
            sample['spk_id'] = spk_id

        return sample

    # ------------------------------------------------------------------
    # Collation
    # ------------------------------------------------------------------
    def collater(self, samples):
        if not samples:
            return {}
        hparams = self.hparams

        ids = torch.tensor([s['id'] for s in samples], dtype=torch.long)
        names = [s['item_name'] for s in samples]
        mels = collate_1d_or_2d([s['mel'] for s in samples], 0.0)
        ref_mels = collate_1d_or_2d([s['ref_mel'] for s in samples], 0.0)
        mel_lens = torch.tensor([s['mel'].shape[0] for s in samples], dtype=torch.long)
        ref_lens = torch.tensor([s['ref_mel'].shape[0] for s in samples], dtype=torch.long)

        batch = {
            'id': ids,
            'item_name': names,
            'nsamples': len(samples),
            'mels': mels,
            'mel_lengths': mel_lens,
            'ref_mels': ref_mels,
            'ref_mel_lengths': ref_lens,
        }

        if hparams.get('use_spk_embed', False):
            batch['spk_embed'] = torch.stack([s['spk_embed'] for s in samples])
        if hparams.get('use_spk_id', False):
            batch['spk_ids'] = torch.tensor([s['spk_id'] for s in samples], dtype=torch.long)

        return batch


class FastSpeechDataset(BaseSpeechDataset):
    """Dataset for FastSpeech-like models with ref mels & f0/uv."""

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        item = self._get_item(index)
        hparams = self.hparams

        # Align mel length with f0 length
        T = min(sample['mel'].shape[0], len(item['f0']))
        sample['mel'] = sample['mel'][:T]
        sample['ref_mel'] = sample['ref_mel'][:T]

        if hparams.get('use_pitch_embed', False):
            f0, uv = norm_interp_f0(item['f0'][:T])
            sample['f0'] = torch.tensor(f0, dtype=torch.float32)
            sample['uv'] = torch.tensor(uv, dtype=torch.float32)
        else:
            sample['f0'], sample['uv'] = None, None

        return sample

    def collater(self, samples):
        batch = super().collater(samples)
        hparams = self.hparams
        if hparams.get('use_pitch_embed', False):
            batch['f0'] = collate_1d_or_2d([s['f0'] for s in samples], 0.0)
            batch['uv'] = collate_1d_or_2d([s['uv'] for s in samples])
        return batch


class FastSpeechWordDataset(FastSpeechDataset):
    def __getitem__(self, index):
        sample = super().__getitem__(index)
        item = self._get_item(index)
        max_frames = sample['mel'].shape[0]
        if 'word' in item:
            sample['words'] = item['word']
            sample["ph_words"] = item["ph_gb_word"]
            sample["word_tokens"] = torch.LongTensor(item["word_token"])
        else:
            sample['words'] = item['words']
            sample["ph_words"] = " ".join(item["ph_words"])
            sample["word_tokens"] = torch.LongTensor(item["word_tokens"])
        sample["mel2word"] = torch.LongTensor(item.get("mel2word"))[:max_frames]
        sample["ph2word"] = torch.LongTensor(item['ph2word'][:self.hparams['max_input_tokens']])
        return sample

    def collater(self, samples):
        batch = super().collater(samples)
        ph_words = [s['ph_words'] for s in samples]
        batch['ph_words'] = ph_words
        word_tokens = collate_1d_or_2d([s['word_tokens'] for s in samples], 0)
        batch['word_tokens'] = word_tokens
        mel2word = collate_1d_or_2d([s['mel2word'] for s in samples], 0)
        batch['mel2word'] = mel2word
        ph2word = collate_1d_or_2d([s['ph2word'] for s in samples], 0)
        batch['ph2word'] = ph2word
        batch['words'] = [s['words'] for s in samples]
        batch['word_lengths'] = torch.LongTensor([len(s['word_tokens']) for s in samples])
        if self.hparams['use_word_input']:
            batch['txt_tokens'] = batch['word_tokens']
            batch['txt_lengths'] = torch.LongTensor([s['word_tokens'].numel() for s in samples])
            batch['mel2ph'] = batch['mel2word']
        return batch
