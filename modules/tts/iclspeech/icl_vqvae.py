import math
import torch
from torch import nn

from modules.commons.layers import Embedding
from modules.tts.commons.align_ops import clip_mel2token_to_multiple, build_word_mask, expand_states, mel2ph_to_mel2word
from modules.tts.fs import FS_DECODERS, FastSpeech
from modules.tts.iclspeech.vqvae.vqvae import VectorQuantizedVAE
from utils.commons.hparams import hparams


class ICLVectorQuantizedVAE(FastSpeech):
    def __init__(self, ph_dict_size, hparams, out_dims=None):
        super().__init__(ph_dict_size, hparams, out_dims)
        # build VAE decoder
        del self.decoder
        del self.mel_out
        self.vqvae = VectorQuantizedVAE(hparams)
        if hparams['use_spk_embed']:
            self.spk_embed_proj = nn.Linear(256, self.hidden_size)

    def forward(self, txt_tokens, mel2ph=None, tgt_mels=None, mel_prompt=None, mel2ph_prompt=None,
                ph_lengths=None, use_gt_mel2ph=True, spk_embed_prompt=None, *args, **kwargs):
        ret = {}
    
        src_nonpadding = (txt_tokens > 0).float()[:, :, None]
        ph_encoder_out = self.encoder(txt_tokens) * src_nonpadding

        # add dur
        if use_gt_mel2ph:
            in_mel2ph = mel2ph
            in_nonpadding = (in_mel2ph > 0).float()[:, :, None]
            nonpadding_prompt = (mel2ph_prompt > 0).float()[:, :, None]
            ph_z_e_x, ph_z_q_x, ph_z_q_x_st, global_z_q_x_st = self.vqvae.forward_first_stage(tgt_mels, mel_prompt, in_nonpadding.transpose(1,2), nonpadding_prompt.transpose(1,2), in_mel2ph, src_nonpadding.transpose(1,2), ph_lengths)
            if spk_embed_prompt != None:
                global_z_q_x_st = self.spk_embed_proj(spk_embed_prompt)[:,:,None]
                
            if hparams.get("no_prosody_code_to_dur_predictor", False) is True:
                dur_inp = (ph_encoder_out + global_z_q_x_st.transpose(1,2)) * src_nonpadding
            else:
                dur_inp = (ph_encoder_out + ph_z_q_x_st.transpose(1,2)) * src_nonpadding

            out_mel2ph = self.forward_dur(dur_inp, mel2ph, txt_tokens, ret)
            out_nonpadding = in_nonpadding
            txt_cond = expand_states(ph_encoder_out, mel2ph)
            txt_cond = txt_cond * out_nonpadding
            x_tilde = self.vqvae.forward_second_stage(txt_cond, ph_z_q_x_st, global_z_q_x_st, out_nonpadding.transpose(1,2), out_mel2ph)
        else:
            in_mel2ph = mel2ph
            in_nonpadding = (in_mel2ph > 0).float()[:, :, None]
            nonpadding_prompt = (mel2ph_prompt > 0).float()[:, :, None]
            ph_z_e_x, ph_z_q_x, ph_z_q_x_st, global_z_q_x_st = self.vqvae.forward_first_stage(tgt_mels, mel_prompt, in_nonpadding.transpose(1,2), nonpadding_prompt.transpose(1,2), in_mel2ph, src_nonpadding.transpose(1,2), ph_lengths)
            if spk_embed_prompt != None:
                global_z_q_x_st = self.spk_embed_proj(spk_embed_prompt)[:,:,None]

            if hparams.get("no_prosody_code_to_dur_predictor", False) is True:
                dur_inp = (ph_encoder_out + global_z_q_x_st.transpose(1,2)) * src_nonpadding
            else:
                dur_inp = (ph_encoder_out + ph_z_q_x_st.transpose(1,2)) * src_nonpadding

            out_mel2ph = self.forward_dur(dur_inp, None, txt_tokens, ret)
            out_nonpadding = (out_mel2ph > 0).float()[:, :, None]
            txt_cond = expand_states(ph_encoder_out, out_mel2ph)
            txt_cond = txt_cond * out_nonpadding
            x_tilde = self.vqvae.forward_second_stage(txt_cond, ph_z_q_x_st, global_z_q_x_st, out_nonpadding.transpose(1,2), out_mel2ph)
        
        ret['x_tilde'], ret['z_e_x'], ret['z_q_x'] = x_tilde, [ph_z_e_x], [ph_z_q_x]
        return ret
    
    def forward_dur(self, dur_input, mel2ph, txt_tokens, ret):
        """

        :param dur_input: [B, T_txt, H]
        :param mel2ph: [B, T_mel]
        :param txt_tokens: [B, T_txt]
        :param ret:
        :return:
        """
        src_padding = txt_tokens == 0
        if self.hparams['predictor_grad'] != 1:
            dur_input = dur_input.detach() + self.hparams['predictor_grad'] * (dur_input - dur_input.detach())
        dur = self.dur_predictor(dur_input, src_padding)
        ret['dur'] = dur
        if mel2ph is None:
            mel2ph = self.length_regulator(dur, src_padding).detach()
        ret['mel2ph'] = mel2ph = clip_mel2token_to_multiple(mel2ph, self.hparams['frames_multiple'])
        return mel2ph

    def get_text_cond(self, txt_tokens):
        src_nonpadding = (txt_tokens > 0).float()[:, :, None]
        ph_encoder_out = self.encoder(txt_tokens) * src_nonpadding
        return ph_encoder_out