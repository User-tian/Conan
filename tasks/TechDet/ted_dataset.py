import os
import math
import gc
from collections import defaultdict
import numpy as np
import torch
from tqdm import tqdm

from tasks.tts.dataset_utils import BaseSpeechDataset
from utils.commons.dataset_utils import collate_1d_or_2d, pad_or_cut_xd
from utils.commons.indexed_datasets import IndexedDataset
from utils.audio import librosa_wav2spec, energy_to_coarse, coarse_to_energy, anything_to_coarse, coarse_to_anything
from utils.audio.pitch_utils import norm_interp_f0, denorm_f0, f0_to_coarse
from utils.commons.signal import add_gaussian_noise

techgroup2lbl = {'气泡音':0, '气声':1, 'Breathy':1, 'Pharyngeal':2, 'Vibrato':3, 'Glissando':4, '假声': 5, 'Mixed_Voice_and_Falsetto': 5, '力度':6}

def add_plain_noise(x, type_and_std=None, random_std=True, x_min=None, x_max=None):
    if type_and_std in ['none', None]:
        return x
    noise_type, std = type_and_std.split(':')
    if noise_type == 'gaussian':
        std = float(std) * np.random.rand() if random_std else float(std)
        x = add_gaussian_noise(x, mean=0.0, std=std)
    if x_min is not None and x_max is not None:
        x = x.clamp(min=x_min, max=x_max)
    return x

class TEDataset(BaseSpeechDataset):
    def __init__(self, prefix, shuffle=False, items=None, data_dir=None):
        super(TEDataset, self).__init__(prefix, shuffle, items, data_dir)
        self.soft_filter = {}
        self.noise_ds = None
        noise_snr = self.hparams.get('noise_snr', '6-20')
        if '-' in noise_snr:
            l, r = noise_snr.split('-')
            self.noise_snr = (float(l), float(r))
        else:
            self.noise_snr = float(noise_snr)
        if items is None and self.avail_idxs is not None:
            ds_names_selected = None
            if prefix in ['train', 'valid'] and self.hparams.get('ds_names_in_training', '') != '':
                ds_names_selected = self.hparams.get('ds_names_in_training', '').split(';') + ['']
                print(f'| Iterating training sets to find samples belong to datasets {ds_names_selected[:-1]}')
            elif prefix == 'test' and self.hparams.get('ds_names_in_testing', '') != '':
                ds_names_selected = self.hparams.get('ds_names_in_testing', '').split(';') + ['']
                print(f'| Iterating testing sets to find samples belong to datasets {ds_names_selected[:-1]}')
            if ds_names_selected is not None:
                avail_idxs = []
                # somehow, can't use self.indexed_ds beforehand (need to create a temp), otherwise '_pickle.UnpicklingError'
                temp_ds = IndexedDataset(f'{self.data_dir}/{self.prefix}')
                for idx in tqdm(range(len(self)), total=len(self)):
                    item = temp_ds[self.avail_idxs[idx]]
                    if item.get('ds_name', '') in ds_names_selected:
                        avail_idxs.append(self.avail_idxs[idx])
                print(f'| Chose [{len(avail_idxs)}] samples belonging to the desired datasets from '
                      f'[{len(self.avail_idxs)}] original samples. ({len(avail_idxs) / len(self.avail_idxs) * 100:.2f}%)')
                self.avail_idxs = avail_idxs
        if items is None and prefix == 'train' and self.hparams.get('dataset_downsample_rate', 1.0) < 1.0 \
                and self.avail_idxs is not None:
            ratio = self.hparams.get('dataset_downsample_rate', 1.0)
            orig_len = len(self.avail_idxs)
            tgt_len = round(orig_len * ratio)
            self.avail_idxs = np.random.choice(self.avail_idxs, size=tgt_len, replace=False).tolist()
            print(f'| Downsamping training set with ratio [{ratio * 100:.2f}%], '
                  f'[{tgt_len}] samples of [{orig_len}] samples are selected.')
        if items is None:
            self.sizes = [self.sizes[i] for i in self.avail_idxs]

        self.apply_neg_sampling = self.hparams.get('apply_neg_sampling', False) and self.prefix == 'train'
        if self.apply_neg_sampling:
            print('| Applying negative sampling.')
            self.make_negative_pools()

    def get_label(self, index):
        return self.id2label[index]

    def make_negative_pools(self):
        item_names = []
        self.idx2item_name = {}
        item_name2idx = {}
        item_name2class_name = {}
        self.item_name2tech_id = {}
        self.parallel_dict = {}
        self.id2label = {}
        
        temp_ds = IndexedDataset(f'{self.data_dir}/{self.prefix}')
        print(f'| Iterating training sets to build negative sampling pools')
        for idx in tqdm(range(len(self)), total=len(self)):
            item = temp_ds[self.avail_idxs[idx]]
            frames = item['len']
            item_name = item['item_name']
            tech_len = len(item['breathy_tech'])
            ph_len = len(item['ph'])
            assert tech_len==ph_len, f'{item_name} length of tech({tech_len}) and ph({ph_len}) is not equal.'

            self.idx2item_name[idx] = item_name
            item_name2idx[item_name] = idx
            item_names.append(item_name)                                           
            if len(item_name.split('#')) == 5:
                spk, tech_group, song_name, tech_name, sen_id = item_name.split('#')  # '华为女声#假声#会呼吸的痛#女声_弱假声#0'
            elif len(item_name.split('#')) == 6:
                lan, spk, tech_group, song_name, tech_name, sen_id = item_name.split('#')  #  Chinese#ZH-Alto-1#Breathy#不再见#Breathy_Group#0000
            else:
                print(f'The format of {item_name} is not correct!')
            
            class_name = '#'.join([spk, tech_group, song_name])
            self.id2label[idx] = techgroup2lbl[tech_group]
            item_name2class_name[item_name] = class_name
            is_neg = '无技巧' in item_name or 'Control_Group' in item_name
            if class_name not in self.parallel_dict:
                if is_neg:
                    self.parallel_dict[class_name] = {'pos': [], 'neg': [(idx, frames)]}
                else:
                    self.parallel_dict[class_name] = {'pos': [(idx, frames)], 'neg': []}
            else:
                if is_neg:
                    self.parallel_dict[class_name]['neg'].append((idx, frames))
                else:
                    self.parallel_dict[class_name]['pos'].append((idx, frames))

            self.item_name2tech_id[item_name] = None
            if tech_group == '假声' or tech_group == 'Mixed_Voice_and_Falsetto':
                self.item_name2tech_id[item_name] = 5
            elif tech_group == '力度':
                self.item_name2tech_id[item_name] = 6
            elif tech_group == '气泡音':
                self.item_name2tech_id[item_name] = 0
            elif tech_group == '气声' or tech_group == 'Breathy':
                self.item_name2tech_id[item_name] = 1
            elif tech_group == 'Pharyngeal':
                self.item_name2tech_id[item_name] = 2
            elif tech_group == 'Vibrato':
                self.item_name2tech_id[item_name] = 3
            elif tech_group == 'Glissando':
                self.item_name2tech_id[item_name] = 4
        self.idx2negative_idxs = {}
        for item_name in item_names:
            idx = item_name2idx[item_name]
            class_name = item_name2class_name[item_name]
            if idx in self.parallel_dict[class_name]['pos']:
                self.idx2negative_idxs[idx] = self.parallel_dict[class_name]['neg']
            else:
                self.idx2negative_idxs[idx] = self.parallel_dict[class_name]['pos']

    def add_noise(self, clean_wav):
        if self.noise_ds is None:   # each instance in multiprocessing must create unique ds object
            self.noise_ds = IndexedDataset(f"{self.hparams['noise_data_dir']}/{self.prefix}")
        noise_idx = np.random.randint(len(self.noise_ds))
        noise_item = self.noise_ds[noise_idx]
        noise_wav = noise_item['feat']

        if type(self.noise_snr) == tuple:
            snr = np.random.rand() * (self.noise_snr[1] - self.noise_snr[0]) + self.noise_snr[0]
        else:
            snr = self.noise_snr
        clean_rms = np.sqrt(np.mean(np.square(clean_wav), axis=-1))
        if len(clean_wav) > len(noise_wav):
            ratio = int(np.ceil(len(clean_wav)/len(noise_wav)))
            noise_wav = np.concatenate([noise_wav for _ in range(ratio)])
        if len(clean_wav) < len(noise_wav):
            start = 0
            noise_wav = noise_wav[start: start + len(clean_wav)]
        noise_rms = np.sqrt(np.mean(np.square(noise_wav), axis=-1)) + 1e-5
        adjusted_noise_rms = clean_rms / (10 ** (snr / 20) + 1e-5)
        adjusted_noise_wav = noise_wav * (adjusted_noise_rms / noise_rms)
        mixed = clean_wav + adjusted_noise_wav
        # Avoid clipping noise
        max_int16 = np.iinfo(np.int16).max
        min_int16 = np.iinfo(np.int16).min
        if mixed.max(axis=0) > max_int16 or mixed.min(axis=0) < min_int16:
            if mixed.max(axis=0) >= abs(mixed.min(axis=0)):
                reduction_rate = max_int16 / mixed.max(axis=0)
            else:
                reduction_rate = min_int16 / mixed.min(axis=0)
            mixed = mixed * reduction_rate
        return mixed

    @staticmethod
    def process_mel(wav_fn, hparams):
        wav2spec_dict = librosa_wav2spec(
            wav_fn,
            fft_size=hparams['fft_size'],
            hop_size=hparams['hop_size'],
            win_length=hparams['win_size'],
            num_mels=hparams['audio_num_mel_bins'],
            fmin=hparams['fmin'],
            fmax=hparams['fmax'],
            sample_rate=hparams['audio_sample_rate'],
            loud_norm=hparams['loud_norm'])
        mel = wav2spec_dict['mel']
        del wav2spec_dict['wav'], wav2spec_dict['linear'], wav2spec_dict['mel_basis'], wav2spec_dict['wav_orig']
        return mel

    def __getitem__(self, index):
        hparams = self.hparams
        item = self._get_item(index)

        # find negative sample
        other_item = None
        other_index = None
        if self.apply_neg_sampling and self.prefix == 'train':
            negative_idxs = self.idx2negative_idxs[index]
            idxs_ = np.arange(len(negative_idxs))
            np.random.shuffle(idxs_)
            for idx_ in idxs_:
                if math.ceil((self.sizes[index] + negative_idxs[idx_][1]) / hparams['frames_multiple']) * hparams['frames_multiple'] < hparams['max_frames']:
                    other_item = self._get_item(negative_idxs[idx_][0])
                    other_index = negative_idxs[idx_][0]
                    break

        wav = item['wav']
        if self.apply_neg_sampling and other_item is not None and self.prefix == 'train':
            other_wav = other_item['wav']
            wav = np.concatenate((wav, other_wav))

        noise_added = np.random.rand() < hparams.get('noise_prob', 0.8)
        if self.prefix == 'test' and not hparams.get('noise_in_test', False):
            noise_added = False
        if noise_added:
            wav = self.add_noise(wav)

        mel = TEDataset.process_mel(wav, self.hparams)
        if self.apply_neg_sampling and other_item is not None and self.prefix == 'train':
            assert len(mel) in np.arange(self.sizes[index] + self.sizes[other_index] - 2, self.sizes[index] + self.sizes[other_index] + 3), \
                (len(mel), self.sizes[index], self.sizes[other_index])
        else:
            assert len(mel) == self.sizes[index], (len(mel), self.sizes[index], index)
        max_frames = hparams['max_frames']

        mel2ph = item['mel2ph']
        if self.apply_neg_sampling and other_item is not None and self.prefix == 'train':
            if mel2ph[-1] == 0:
                for i in range(len(mel2ph)-1, 0, -1):
                    if mel2ph[i] > 0:
                        mel2ph[i:] = mel2ph[i]  # eliminate rear 0s
                        break
            last_idx = mel2ph[-1]
            other_mel2ph = other_item['mel2ph']
            other_mel2ph[other_mel2ph > 0] = other_mel2ph[other_mel2ph > 0] + last_idx
            mel2ph = np.concatenate((mel2ph, other_mel2ph))

        mel2ph_len = sum((mel2ph > 0).astype(int))
        if self.apply_neg_sampling and other_item is not None and self.prefix == 'train':
            T = min(item['len'] + other_item['len'], mel2ph_len, len(item['f0']) + len(other_item['f0']))
        else:
            T = min(item['len'], mel2ph_len, len(item['f0']))
        real_len = T
        T = math.ceil(min(T, max_frames) / hparams['frames_multiple']) * hparams['frames_multiple']
        spec = torch.Tensor(mel)[:max_frames]
        spec = pad_or_cut_xd(spec, T, dim=0)
        if 5 < hparams.get('use_mel_bins', hparams['audio_num_mel_bins']) < hparams['audio_num_mel_bins']:
            spec = spec[:, :hparams.get('use_mel_bins', 80)]
        sample = {
            "id": index,
            "item_name": item['item_name'],
            "ph": item.get('ph', ""),
            "mel": spec,
            "mel_nonpadding": spec.abs().sum(-1) > 0,
        }
        if self.apply_neg_sampling and other_item is not None and self.prefix == 'train':
            sample['item_name'] = (item['item_name'], other_item['item_name'])
        if noise_added and self.prefix in ['train', 'valid']:
            noisy_mel = add_plain_noise(sample['mel'], type_and_std=hparams.get('mel_add_noise', 'none'))
            sample['mel'] = torch.clamp(noisy_mel, hparams['mel_vmin'], hparams['mel_vmax'])
        sample['mel2ph'] = mel2ph = pad_or_cut_xd(torch.LongTensor(mel2ph), T, 0)
        sample['mel_nonpadding'] = pad_or_cut_xd(sample['mel_nonpadding'].float(), T, 0)

        ph_bd = torch.zeros_like(mel2ph)
        for idx in range(1, T):
            if mel2ph[idx] == 0:
                break
            if mel2ph[idx] != mel2ph[idx-1]:
                ph_bd[idx] = 1
        sample['ph_bd'] = ph_bd.long()
            
        if hparams['use_pitch_embed']:
            assert 'f0' in item
            if self.apply_neg_sampling and other_item is not None and self.prefix == 'train':
                f0 = np.concatenate((item['f0'], other_item['f0']))
                f0, uv = norm_interp_f0(f0[:T])
            else:
                f0, uv = norm_interp_f0(item["f0"][:T])
            # noise f0
            if noise_added and self.prefix in ['train', 'valid']:
                f0 = torch.FloatTensor(f0)
                f0[uv == 0] = add_plain_noise(f0[uv == 0], type_and_std=hparams.get('f0_add_noise', 'none'))
            uv = pad_or_cut_xd(torch.FloatTensor(uv), T, 0)
            f0 = pad_or_cut_xd(torch.FloatTensor(f0), T, 0)
            pitch_coarse = f0_to_coarse(denorm_f0(f0, uv))
        else:
            f0, uv, pitch, pitch_coarse = None, None, None, None
        sample["f0"], sample["uv"], sample["pitch_coarse"] = f0, uv, pitch_coarse

        if hparams.get('use_breathiness', False):
            if self.apply_neg_sampling and other_item is not None and self.prefix == 'train':
                breathiness = np.concatenate((item['breathiness'], other_item['breathiness']))
                breathiness = torch.FloatTensor(breathiness)
            else:
                breathiness = torch.FloatTensor(item['breathiness'])
            if noise_added and self.prefix in ['train', 'valid']:
                breathiness = add_plain_noise(breathiness, type_and_std=hparams.get('breathiness_add_noise', 'none'), x_min=0, x_max=0.8)
            breathiness_coarse = energy_to_coarse(breathiness)
            breathiness_coarse = pad_or_cut_xd(breathiness_coarse, T, 0)
            sample['breathiness'] = breathiness_coarse

        if hparams.get('use_energy', False):
            if self.apply_neg_sampling and other_item is not None and self.prefix == 'train':
                energy = np.concatenate((item['energy'], other_item['energy']))
                energy = torch.FloatTensor(energy)
            else:
                energy = torch.FloatTensor(item['energy'])
            if noise_added and self.prefix in ['train', 'valid']:
                energy = add_plain_noise(energy, type_and_std=hparams.get('energy_add_noise', 'none'), x_min=0, x_max=0.8)
            energy_coarse = energy_to_coarse(energy)
            energy_coarse = pad_or_cut_xd(energy_coarse, T, 0)
            sample['energy'] = energy_coarse

        if hparams.get('use_zcr', False):
            if self.apply_neg_sampling and other_item is not None and self.prefix == 'train':
                zcr = np.concatenate((item['zcr'], other_item['zcr']))
                zcr = torch.FloatTensor(zcr)
            else:
                zcr = torch.FloatTensor(item['zcr'])
            if noise_added and self.prefix in ['train', 'valid']:
                zcr = add_plain_noise(zcr, type_and_std=hparams.get('zcr_add_noise', 'none'), x_min=0, x_max=1)
            zcr_coarse = anything_to_coarse(zcr, bins=256, x_max=1.0, x_min=0.0, pad=0)
            zcr_coarse = pad_or_cut_xd(zcr_coarse, T, 0)
            sample['zcr'] = zcr_coarse
        
        if self.apply_neg_sampling and self.prefix == 'train':
            sample['tech_id'] = self.item_name2tech_id[item['item_name']]
        
        # make tech matrix
        if self.apply_neg_sampling and other_item is not None and self.prefix == 'train':
            mix_tech = torch.LongTensor(item['mix_tech'] + other_item['mix_tech'])
            falsetto_tech = torch.LongTensor(item['falsetto_tech'] + other_item['falsetto_tech'])
            mf_tech = mix_tech + 2 * falsetto_tech
            breathe_tech = torch.LongTensor(item['breathy_tech'] + other_item['breathy_tech'])
            if 'bubble_tech' in item and 'strong_tech' in item and 'weak_tech' in item:
                bubble_tech = torch.LongTensor(item['bubble_tech'] + other_item['bubble_tech'])
                strong_tech = torch.LongTensor(item['strong_tech'] + other_item['strong_tech'])
                weak_tech = torch.LongTensor(item['weak_tech'] + other_item['weak_tech'])
                sw_tech = weak_tech + 2 * strong_tech
            else:
                bubble_tech = torch.zeros_like(breathe_tech)
                strong_tech = torch.zeros_like(breathe_tech)
                weak_tech = torch.zeros_like(breathe_tech)
                sw_tech = weak_tech + 2 * strong_tech
            if 'pharyngeal_tech' in item and 'vibrato_tech' in item and 'glissando_tech' in item:
                pharyngeal_tech = torch.LongTensor(item['pharyngeal_tech'] + other_item['pharyngeal_tech'])
                vibrato_tech = torch.LongTensor(item['vibrato_tech'] + other_item['vibrato_tech'])
                glissando_tech = torch.LongTensor(item['glissando_tech'] + other_item['glissando_tech'])
            else:
                pharyngeal_tech = torch.zeros_like(breathe_tech)
                vibrato_tech = torch.zeros_like(breathe_tech)
                glissando_tech = torch.zeros_like(breathe_tech)
        else:
            mix_tech = torch.LongTensor(item['mix_tech'])
            falsetto_tech = torch.LongTensor(item['falsetto_tech'])
            mf_tech = mix_tech + 2 * falsetto_tech
            breathe_tech = torch.LongTensor(item['breathy_tech'])
            if 'bubble_tech' in item and 'strong_tech' in item and 'weak_tech' in item:
                bubble_tech = torch.LongTensor(item['bubble_tech'])
                strong_tech = torch.LongTensor(item['strong_tech'])
                weak_tech = torch.LongTensor(item['weak_tech'])
                sw_tech = weak_tech + 2 * strong_tech
            else:
                bubble_tech = torch.zeros_like(breathe_tech)
                strong_tech = torch.zeros_like(breathe_tech)
                weak_tech = torch.zeros_like(breathe_tech)
                sw_tech = weak_tech + 2 * strong_tech
            if 'pharyngeal_tech' in item and 'vibrato_tech' in item and 'glissando_tech' in item:
                pharyngeal_tech = torch.LongTensor(item['pharyngeal_tech'])
                vibrato_tech = torch.LongTensor(item['vibrato_tech'])
                glissando_tech = torch.LongTensor(item['glissando_tech'])
            else:
                pharyngeal_tech = torch.zeros_like(breathe_tech)
                vibrato_tech = torch.zeros_like(breathe_tech)
                glissando_tech = torch.zeros_like(breathe_tech)
        mf_tech = torch.clamp(mf_tech, max=2)
        tech_matrix = torch.stack((bubble_tech, breathe_tech, pharyngeal_tech, vibrato_tech, glissando_tech, mf_tech, sw_tech), dim=1)
        sample['tech'] = tech_matrix    # [T, C]
        
        # delete big redundancy
        if not hparams.get('use_mel', True) and 'mel' in sample:
            sample['mel'] = None
            if 'mel_nonpadding' in sample:
                sample['mel_nonpadding'] = None
        if not hparams.get('use_wav', False) and 'wav' in sample:
            sample['wav'] = None
        tech_len = sample['tech'].shape[0]
        if tech_len != torch.sum(sample["ph_bd"]) + 1:
            return None
        return sample

    def collater(self, samples):
        if len(samples) == 0:
            return {}
        new_samples = []
        for sample in samples:
            if sample is not None:
                new_samples.append(sample)
        samples = new_samples
        hparams = self.hparams
        id = torch.LongTensor([s['id'] for s in samples])
        item_names = [s['item_name'] for s in samples]
        ph = [s['ph'] for s in samples]
        mels = collate_1d_or_2d([s['mel'] for s in samples], 0.0) if 'mel' in samples[0] else None
        mel_lengths = torch.LongTensor([s['mel'].shape[0] for s in samples]) if 'mel' in samples[0] else 0
        mel_nonpadding = collate_1d_or_2d([s['mel_nonpadding'] for s in samples], 0.0) if 'mel_nonpadding' in samples[0] else None

        batch = {
            'id': id,
            'item_name': item_names,
            'nsamples': len(samples),
            'ph': ph,
            'mels': mels,
            'mel_lengths': mel_lengths,
            'mel_nonpadding': mel_nonpadding
        }

        if hparams['use_spk_embed']:
            spk_embed = torch.stack([s['spk_embed'] for s in samples])
            batch['spk_embed'] = spk_embed
        if hparams['use_spk_id']:
            spk_ids = torch.LongTensor([s['spk_id'] for s in samples])
            batch['spk_ids'] = spk_ids

        if hparams['use_pitch_embed']:
            f0 = collate_1d_or_2d([s['f0'] for s in samples], 0.0)
            uv = collate_1d_or_2d([s['uv'] for s in samples])
            pitch_coarse = collate_1d_or_2d([s['pitch_coarse'] for s in samples])
        else:
            f0, uv, pitch, pitch_coarse = None, None, None, None
        batch['mel2ph'] = collate_1d_or_2d([s['mel2ph'] for s in samples], 0.0)
        batch['f0'], batch['uv'], batch['pitch_coarse'] = f0, uv, pitch_coarse
        batch["wav"] = collate_1d_or_2d([s['wav'] for s in samples], 0.0) if 'wav' in samples[0] else None
        batch["ph_bd"] = collate_1d_or_2d([s['ph_bd'] for s in samples], 0.0)
        batch['mel2word'] = collate_1d_or_2d([s['mel2word'] for s in samples], 0) if 'mel2word' in samples[0] else None

        if hparams.get('have_tech_gt', True):
            batch['techs'] = collate_1d_or_2d([s['tech'] for s in samples], 0)
            batch['tech_ids'] = [s['tech_id'] for s in samples] if 'tech_id' in samples[0] else None

        batch['breathiness'] = collate_1d_or_2d([s['breathiness'] for s in samples], 0) if 'breathiness' in samples[0] else None
        batch['energy'] = collate_1d_or_2d([s['energy'] for s in samples], 0) if 'energy' in samples[0] else None
        batch['zcr'] = collate_1d_or_2d([s['zcr'] for s in samples], 0) if 'zcr' in samples[0] else None

        return batch

    @property
    def num_workers(self):
        if self.prefix == 'train':
            return int(os.getenv('NUM_WORKERS', self.hparams['ds_workers']))
        return 1
if __name__ == '__main__':
    from utils.commons.hparams import set_hparams
    from utils.commons.dataset_utils import data_loader, BaseConcatDataset, build_dataloader

    hparams = set_hparams(config='/home2/zhangyu/gwx/github/TechSinger/egs/ted/te.yaml')
    train_dataset = TEDataset(prefix=hparams['train_set_name'], shuffle=True)
    data_loader = build_dataloader(train_dataset, True, hparams['max_tokens'], hparams['max_sentences'])
    i = 0
    for batch in data_loader:
        # print('123')
        # # break
        # break
        i+=1
        print(i)
        pass