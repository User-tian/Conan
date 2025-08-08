import os
import numpy as np
import torch
from inference.tts.base_tts_infer import BaseTTSInfer
from utils.commons.ckpt_utils import load_ckpt
from utils.text.text_encoder import build_token_encoder
from tasks.tts.vocoder_infer.base_vocoder import get_vocoder_cls
from utils.audio import librosa_wav2spec
from utils.commons.hparams import set_hparams
from utils.commons.hparams import hparams
from utils.audio.io import save_wav
import json
from modules.TechSinger.techsinger import RFSinger, RFPostnet
import time
from utils.audio.pitch.utils import norm_interp_f0, denorm_f0

def process_align(ph_durs, mel, item, hop_size ,audio_sample_rate):
    mel2ph = np.zeros([mel.shape[0]], int)
    startTime = 0

    for i_ph in range(len(ph_durs)):
        start_frame = int(startTime * audio_sample_rate / hop_size + 0.5)
        end_frame = int((startTime + ph_durs[i_ph]) * audio_sample_rate / hop_size + 0.5)
        mel2ph[start_frame:end_frame] = i_ph + 1
        startTime = startTime + ph_durs[i_ph]

    return mel2ph

def self_clone(x):
    if x == None:
        return None
    y = x.clone()
    result = torch.cat((x, y), dim=0)
    return result
def process_audio(wav_fn):
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
    wav = wav2spec_dict['wav'].astype(np.float16)
    
    return wav, mel
class techinfer(BaseTTSInfer):
    def build_model(self):
        dict_size = len(self.ph_encoder)
        model = RFSinger(dict_size, self.hparams)
        model.eval()
        load_ckpt(model, hparams['fs2_ckpt_dir'], strict=True)     
        self.model_post=RFPostnet()
        
        load_ckpt(self.model_post, os.path.join('checkpoints', hparams['exp_name']), strict=True)
        self.model_post.eval()
        self.model_post.to(self.device)

        binary_data_dir = hparams['binary_data_dir']
        self.ph_encoder = build_token_encoder(f'{binary_data_dir}/phone_set.json')
        return model

    def build_vocoder(self):
        vocoder = get_vocoder_cls(hparams["vocoder"])()
        return vocoder

    def forward_model(self, inp):
        sample = self.input_to_batch(inp)
        txt_tokens = sample['txt_tokens']  # [B, T_t]
        txt_lengths = sample['txt_lengths']
        notes, note_durs,note_types = sample["notes"], sample["note_durs"],sample['note_types']
        
        spk_id=sample['spk_id']
        mix,falsetto,breathy=sample['mix'],sample['falsetto'],sample['breathy']
        pharyngeal,vibrato,glissando = sample['pharyngeal'],sample['vibrato'],sample['glissando']
        bubble,strong,weak = sample['bubble'],sample['strong'],sample['weak']
        mel2ph = sample['mel2ph']
        # mel2ph=None
        f0, uv = sample["f0"], sample["uv"]
        # f0 = uv = None
        output = {}
        # Run model
        with torch.no_grad():
            umix, ufalsetto, ubreathy = torch.ones_like(mix, dtype=mix.dtype) * 2, torch.ones_like(falsetto, dtype=falsetto.dtype) * 2, torch.ones_like(breathy, dtype=breathy.dtype) * 2
            ububble, ustrong, uweak = torch.ones_like(mix, dtype=mix.dtype) * 2, torch.ones_like(strong, dtype=strong.dtype) * 2, torch.ones_like(weak, dtype=weak.dtype) * 2
            upharyngeal, uvibrato, uglissando = torch.ones_like(bubble, dtype=bubble.dtype) * 2, torch.ones_like(vibrato, dtype=vibrato.dtype) * 2, torch.ones_like(glissando, dtype=glissando.dtype) * 2
            mix = torch.cat((mix, umix), dim=0)
            falsetto = torch.cat((falsetto, ufalsetto), dim=0)
            breathy = torch.cat((breathy, ubreathy), dim=0)
            bubble = torch.cat((bubble, ububble), dim=0)
            strong = torch.cat((strong, ustrong), dim=0)
            weak = torch.cat((weak, uweak), dim=0)
            pharyngeal = torch.cat((pharyngeal, upharyngeal), dim=0)
            vibrato = torch.cat((vibrato, uvibrato), dim=0)
            glissando = torch.cat((glissando, uglissando), dim=0)
            
            txt_tokens = self_clone(txt_tokens)
            mel2ph = self_clone(mel2ph)
            spk_id = self_clone(spk_id)
            f0 = self_clone(f0)
            uv = self_clone(uv)
            notes = self_clone(notes)
            note_durs = self_clone(note_durs)
            note_types = self_clone(note_types)
            
            start_time = time.time()
            output = self.model(txt_tokens, mel2ph=mel2ph, spk_id=spk_id, f0=f0, uv=uv, 
                                note=notes, note_dur=note_durs, note_type=note_types,
                                mix=mix, falsetto=falsetto, breathy=breathy,
                                bubble=bubble, strong=strong, weak=weak,
                                pharyngeal=pharyngeal, vibrato=vibrato, glissando=glissando, 
                                infer=True)
            self.model_post(None, True, output, True, cfg_scale=2.0,  noise=None)

            mel_out =  output['mel_out'][0]
            pred_f0 = output.get('f0_denorm_pred')[0]
            wav_out = self.vocoder.spec2wav(mel_out.cpu(),f0=pred_f0.cpu())
            end_time = time.time()

        return wav_out, mel_out, end_time-start_time
    

    def preprocess_input(self, inp):
        """
        :param inp: {'text': str, 'item_name': (str, optional), 'spk_name': (str, optional)}
        :return:
        """
        ph_gen=' '.join(inp['text_gen'])
        ph_token = self.ph_encoder.encode(ph_gen)
        note=inp['note_gen']
        note_dur=inp['note_dur_gen']
        note_type=inp['note_type_gen']

        item = {'item_name': inp['gen'], 'text': inp['text_gen'], 'ph': inp['text_gen'],
                'ph_token': ph_token, 'spk_id':inp['spk_id'],
                'mel2ph': None, 'note':note, 'note_dur':note_dur,'note_type':note_type,
                'mix_tech': inp['mix_tech'], 'falsetto_tech': inp['falsetto_tech'], 'breathy_tech': inp['breathy_tech'],
                'pharyngeal_tech':inp['pharyngeal_tech'] , 'vibrato_tech':inp['vibrato_tech'],'glissando_tech':inp['glissando_tech'],
                'bubble_tech':inp['bubble_tech'] , 'strong_tech':inp['strong_tech'],'weak_tech':inp['weak_tech']
                }

        f0 = np.load(inp['wav_fn'].replace(".wav", "_f0.npy").replace('/home2/zhangyu/data/GTSinger', '/home2/zhangyu/data/nips_final/nips_submit'))
        f0, uv = norm_interp_f0(f0)
        uv = torch.FloatTensor(uv)
        f0 = torch.FloatTensor(f0)
        item['f0'] = f0
        item['uv'] = uv
        wav, mel= process_audio(inp['wav_fn'])
        # item['mel'] = mel
        item['mel2ph']= mel2ph = process_align(inp["ph_durs"], mel, item,hop_size=hparams['hop_size'], audio_sample_rate=hparams['audio_sample_rate'])

        item['ph_len'] = len(item['ph_token'])
        return item

    def input_to_batch(self, item):
        item_names = [item['item_name']]
        text = [item['text']]
        ph = [item['ph']]
        txt_tokens = torch.LongTensor(item['ph_token'])[None, :].to(self.device)
        txt_lengths = torch.LongTensor([txt_tokens.shape[1]])[None, :].to(self.device)
        
        note = torch.LongTensor(item['note'])[None, :].to(self.device)
        note_dur = torch.FloatTensor(item['note_dur'])[None, :].to(self.device)
        note_type = torch.LongTensor(item['note_type'][:hparams['max_input_tokens']])[None, :].to(self.device)

        mix = torch.LongTensor(item['mix_tech'][:hparams['max_input_tokens']])[None, :].to(self.device)
        falsetto= torch.LongTensor(item['falsetto_tech'][:hparams['max_input_tokens']])[None, :].to(self.device)
        breathy = torch.LongTensor(item['breathy_tech'][:hparams['max_input_tokens']])[None, :].to(self.device)
        pharyngeal = torch.LongTensor(item['pharyngeal_tech'][:hparams['max_input_tokens']])[None, :].to(self.device)
        vibrato = torch.LongTensor(item['vibrato_tech'][:hparams['max_input_tokens']])[None, :].to(self.device)
        glissando = torch.LongTensor(item['glissando_tech'][:hparams['max_input_tokens']])[None, :].to(self.device)
        bubble = torch.LongTensor(item['bubble_tech'][:hparams['max_input_tokens']])[None, :].to(self.device)
        strong = torch.LongTensor(item['strong_tech'][:hparams['max_input_tokens']])[None, :].to(self.device)
        weak = torch.LongTensor(item['weak_tech'][:hparams['max_input_tokens']])[None, :].to(self.device)
        
        mel2ph = torch.LongTensor(item['mel2ph'])[None, :].to(self.device)
        f0 = torch.FloatTensor(item['f0'])[None, :].to(self.device)
        uv = torch.FloatTensor(item['uv'])[None, :].to(self.device)

        spk_id= torch.LongTensor([item['spk_id']]).to(self.device)

        batch = {
            'item_name': item_names,
            'text': text,
            'ph': ph,
            'txt_tokens': txt_tokens,
            'txt_lengths': txt_lengths,
            'spk_id': spk_id,
            # 'mels': mels,
            # 'mel_lengths': mel_lengths,
            'notes': note,
            'note_durs': note_dur,
            'note_types': note_type,
            'mel2ph': mel2ph,
            'f0': f0,
            'uv': uv,
        }

        batch['mix'],batch['falsetto'],batch['breathy']=mix,falsetto,breathy
        batch['pharyngeal'],batch['vibrato'],batch['glissando']=pharyngeal,vibrato,glissando
        batch['bubble'],batch['strong'],batch['weak']=bubble,strong,weak
        return batch

    @classmethod
    def example_run(cls):
        set_hparams()
        exp_name = hparams['exp_name'].split('/')[-1]
        
        item_name_prefix = '在夜里跳舞#别找我麻烦'
        infer_ins = cls(hparams)
        items_list = json.load(open(f"{hparams['processed_data_dir']}/metadata.json"))
        if os.path.exists(f"{hparams['processed_data_dir']}/spker_set.json"):
            spker_set = json.load(open(f"{hparams['processed_data_dir']}/spker_set.json", 'r'))
        else:
            spker_set = {"ZH-Alto-1": 33, "ZH-Tenor-1": 34, "EN-Alto-1": 35, "EN-Alto-2": 36, "EN-Tenor-1": 37}
        # item_name = "Chinese#ZH-Alto-1#Mixed_Voice_and_Falsetto#十年#Mixed_Voice_Group#0000"
        # item_name = 'English#EN-Alto-1#Breathy#all is found#Breathy_Group#0000'
        # item_name = '华为女声#假声#江南#弱假声#0001'
        # item_name = '少女音#葬花吟#17'
        inp = {
                'gen': item_name_prefix
        }
        infer_dir = f'infer_out/{exp_name}/{item_name_prefix}'
        os.makedirs(infer_dir, exist_ok=True)
        all_txt = []
        all_phone = ''
        for item in items_list:
            if inp['gen'] in item['item_name']:
                item_name = item['item_name']
                wav_fn = os.path.join(infer_dir, f'{item_name}.wav')
                inp['text_gen']=item['ph']
                inp['note_gen']=item['ep_pitches']
                inp['note_dur_gen'] =item['ep_notedurs']
                inp['note_type_gen']=item['ep_types']  
                singer = item['singer']
                print(singer)
                inp['spk_id']=spker_set[singer]
                inp['wav_fn']=item['wav_fn']  
                inp['ph_durs']=item['ph_durs']
                inp['mix_tech']=item['mix_tech'] if 'mix_tech' in item else [0] * len(inp['text_gen'])
                inp['falsetto_tech']=item['falsetto_tech']  if 'falsetto_tech' in item else [0] * len(inp['text_gen'])
                inp['breathy_tech']=item['breathy_tech'] if 'breathy_tech' in item else [0] * len(inp['text_gen'])
                inp['bubble_tech']=item['bubble_tech']  if 'bubble_tech' in item else [0] * len(inp['text_gen'])
                inp['strong_tech']=item['strong_tech']  if 'strong_tech' in item else [0] * len(inp['text_gen'])
                inp['weak_tech']=item['weak_tech']  if 'weak_tech' in item else [0] * len(inp['text_gen'])
                inp['pharyngeal_tech']=item['pharyngeal_tech']  if 'pharyngeal_tech' in item else [0] * len(inp['text_gen'])
                inp['vibrato_tech']=item['vibrato_tech']  if 'vibrato_tech' in item else [0] * len(inp['text_gen'])
                inp['glissando_tech']=item['glissando_tech'] if 'glissando_tech' in item else [0] * len(inp['text_gen'])
                all_txt += item['txt']
                # for ph, tech_mix, tech_gli in zip(item['ph'],item['mix_tech'], item['glissando_tech']):
                #     if ph=='<AP>':
                #         all_phone = all_phone + '&lt;AP&gt;(0), '
                #     elif ph=='<SP>':
                #         all_phone = all_phone + '&lt;SP&gt;(0), '
                #     elif tech_mix == '1' and tech_gli=='1':
                #         all_phone = all_phone + ph + '(1, 6), '
                #     elif tech_mix == '1':
                #         all_phone = all_phone + ph + '(1), '
                #     elif tech_gli=='1':
                #         all_phone = all_phone + ph + '(6), '
                #     else:
                #         all_phone = all_phone + ph + '(0), '
                    
                # if os.path.exists(wav_fn):
                #     continue
                # out = infer_ins.infer_once(inp)
                # wav_out, mel_out, time = out
                # save_wav(wav_out, wav_fn, hparams['audio_sample_rate'])
                
                # print(f'enjoy {wav_fn}')
        print(' '.join(all_txt))
        print(all_phone)
        # for tech, ref_tech in tech2id.items():
        
        
if __name__ == '__main__':
    techinfer.example_run()