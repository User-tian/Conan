import glob
import importlib
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DistributedSampler
from utils.commons.dataset_utils import BaseDataset, collate_1d, collate_2d
from utils.commons.hparams import hparams
from utils.commons.indexed_datasets import IndexedDataset


class EndlessDistributedSampler(DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.shuffle = shuffle

        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = [i for _ in range(1000) for i in torch.randperm(
                len(self.dataset), generator=g).tolist()]
        else:
            indices = [i for _ in range(1000) for i in list(range(len(self.dataset)))]
        indices = indices[:len(indices) // self.num_replicas * self.num_replicas]
        indices = indices[self.rank::self.num_replicas]
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class VocoderDataset(BaseDataset):
    def __init__(self, prefix, shuffle=False):
        super().__init__(shuffle)
        self.hparams = hparams
        self.prefix = prefix
        self.data_dir = hparams['binary_data_dir']
        self.is_infer = prefix == 'test'
        self.batch_max_frames = 0 if self.is_infer else hparams['max_samples'] // hparams['hop_size']
        self.aux_context_window = hparams['aux_context_window']
        self.hop_size = hparams['hop_size']
        self.use_pitch_embed = hparams['use_pitch_embed']
        if self.is_infer and hparams['test_input_dir'] != '':
            self.indexed_ds, self.sizes = self.load_test_inputs(hparams['test_input_dir'])
            self.avail_idxs = [i for i, _ in enumerate(self.sizes)]
        else:
            self.indexed_ds = None
            self.sizes = np.load(f'{self.data_dir}/{self.prefix}_lengths.npy')
            self.avail_idxs = [idx for idx, s in enumerate(self.sizes) if
                               s - 2 * self.aux_context_window > self.batch_max_frames]
            print(f"| {len(self.sizes) - len(self.avail_idxs)} short items are skipped in {prefix} set.")
            self.sizes = [s for idx, s in enumerate(self.sizes) if
                          s - 2 * self.aux_context_window > self.batch_max_frames]

    def _get_item(self, index):
        if self.indexed_ds is None:
            self.indexed_ds = IndexedDataset(f'{self.data_dir}/{self.prefix}')
        item = self.indexed_ds[index]
        return item

    def __getitem__(self, index):
        index = self.avail_idxs[index]
        item = self._get_item(index)
        sample = {
            "id": index,
            "item_name": item['item_name'],
            "mel": torch.FloatTensor(item['mel']),
            "wav": torch.FloatTensor(item['wav'].astype(np.float32)),
        }
        if 'f0' in item:
            # sample['pitch'] = torch.LongTensor(item['pitch'])
            sample['f0'] = torch.FloatTensor(item['f0'])
        return sample

    def collater(self, batch):
        if len(batch) == 0:
            return {}

        y_batch, c_batch, p_batch, f0_batch = [], [], [], []
        item_name = []
        have_pitch = 'f0' in batch[0]
        for idx in range(len(batch)):
            item_name.append(batch[idx]['item_name'])
            x, c = batch[idx]['wav'], batch[idx]['mel']
            if have_pitch:
                # p = batch[idx]['pitch']
                f0 = batch[idx]['f0']
            self._assert_ready_for_upsampling(x, c, self.hop_size, 0)
            # print(f'x:{x.shape},c:{c.shape},p:{p.shape},f:{f0.shape}')
            # exit()
            # import ipdb
            # ipdb.set_trace()
            if len(c) - 2 * self.aux_context_window > self.batch_max_frames:
                # randomly pickup with the batch_max_steps length of the part
                batch_max_frames = self.batch_max_frames if self.batch_max_frames != 0 else len(
                    c) - 2 * self.aux_context_window - 1
                batch_max_steps = batch_max_frames * self.hop_size
                interval_start = self.aux_context_window
                interval_end = len(c) - batch_max_frames - self.aux_context_window
                start_frame = np.random.randint(interval_start, interval_end)
                start_step = start_frame * self.hop_size
                y = x[start_step: start_step + batch_max_steps]
                c = c[start_frame - self.aux_context_window:
                      start_frame + self.aux_context_window + batch_max_frames]
                if have_pitch:
                    # p = p[start_frame - self.aux_context_window:
                        #   start_frame + self.aux_context_window + batch_max_frames]
                    f0 = f0[start_frame - self.aux_context_window:
                            start_frame + self.aux_context_window + batch_max_frames]
                self._assert_ready_for_upsampling(y, c, self.hop_size, self.aux_context_window)
            else:
                print(f"Removed short sample from batch (length={len(x)}).")
                continue
            y_batch += [y.reshape(-1, 1)]  # [(T, 1), (T, 1), ...]
            c_batch += [c]  # [(T' C), (T' C), ...]
            if have_pitch:
                # p_batch += [p]  # [(T' C), (T' C), ...]
                f0_batch += [f0]  # [(T' C), (T' C), ...]

        # convert each batch to tensor, asuume that each item in batch has the same length
        y_batch = collate_2d(y_batch, 0).transpose(2, 1)  # (B, 1, T)
        c_batch = collate_2d(c_batch, 0).transpose(2, 1)  # (B, C, T')
        if have_pitch:
            # p_batch = collate_1d(p_batch, 0)  # (B, T')
            f0_batch = collate_1d(f0_batch, 0)  # (B, T')
        else:
            p_batch, f0_batch = None, None

        # make input noise signal batch tensor
        z_batch = torch.randn(y_batch.size())  # (B, 1, T)
        return {
            'z': z_batch,
            'mels': c_batch,
            'wavs': y_batch,
            # 'pitches': p_batch,
            'f0': f0_batch,
            'item_name': item_name
        }

    @staticmethod
    def _assert_ready_for_upsampling(x, c, hop_size, context_window):
        """Assert the audio and feature lengths are correctly adjusted for upsamping."""
        assert len(x) == (len(c) - 2 * context_window) * hop_size

    def load_test_inputs(self, test_input_dir, spk_id=0):
        inp_wav_paths = []
        for dir in test_input_dir:
            if len(dir) > 1:
                inp_wav_paths += sorted(glob.glob(f'{dir}/*.wav') + glob.glob(f'{dir}/*.mp3')
                                        + glob.glob(f'{dir}/**/*.wav') + glob.glob(f'{dir}/**/*.mp3'))
            else:
                raise IOError
        # inp_wav_paths = sorted(glob.glob(f'{test_input_dir}/*.wav') + glob.glob(f'{test_input_dir}/*.mp3')
        #                        + glob.glob(f'{test_input_dir}/**/*.wav') + glob.glob(f'{test_input_dir}/**/*.mp3'))
        sizes = []
        items = []

        binarizer_cls = hparams.get("binarizer_cls", 'data_gen.tts.base_binarizer.BaseBinarizer')
        pkg = ".".join(binarizer_cls.split(".")[:-1])
        cls_name = binarizer_cls.split(".")[-1]
        binarizer_cls = getattr(importlib.import_module(pkg), cls_name)
        binarization_args = hparams['binarization_args']

        for wav_fn in inp_wav_paths:
            item_name = wav_fn[len(wav_fn.split('/')[-1]) + 1:].replace("/", "_")
            ph = txt = tg_fn = ''
            encoder = None, None
            item = binarizer_cls.process_item(
                item_name, ph, txt, tg_fn, wav_fn, spk_id, encoder, binarization_args)
            items.append(item)
            sizes.append(item['len'])
        return items, sizes

class HifiGANDataset(VocoderDataset):
    def load_test_inputs(self, test_input_dir, spk_id=0):
        inp_wav_paths = []
        for dir in test_input_dir:
            if len(dir) > 1:
                inp_wav_paths += sorted(glob.glob(f'{dir}/*.wav') + glob.glob(f'{dir}/*.mp3')
                                        + glob.glob(f'{dir}/**/*.wav') + glob.glob(f'{dir}/**/*.mp3'))
            else:
                raise IOError
        # inp_wav_paths = sorted(glob.glob(f'{test_input_dir}/*.wav') + glob.glob(f'{test_input_dir}/*.mp3')
        #                        + glob.glob(f'{test_input_dir}/**/*.wav') + glob.glob(f'{test_input_dir}/**/*.mp3'))
        sizes = []
        items = []

        binarizer_cls = hparams.get("binarizer_cls", 'data_gen.tts.base_binarizer.BaseBinarizer')
        pkg = ".".join(binarizer_cls.split(".")[:-1])
        cls_name = binarizer_cls.split(".")[-1]
        binarizer_cls = getattr(importlib.import_module(pkg), cls_name)
        binarization_args = hparams['binarization_args']

        for wav_fn in inp_wav_paths:
            item_name = wav_fn[len(wav_fn.split('/')[-1]) + 1:].replace("/", "_")
            ph = txt = ''
            tg_fn = wav_fn.replace('.wav', '.TextGrid')
            encoder = None, None
            item = binarizer_cls.process_item(
                item_name, ph, txt, tg_fn, wav_fn, spk_id, encoder, binarization_args)
            items.append(item)
            sizes.append(item['len'])
        return items, sizes
