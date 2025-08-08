import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from modules.commons.layers import Embedding
from modules.tts.commons.align_ops import clip_mel2token_to_multiple, expand_states
from modules.tts.fs import FS_ENCODERS, FastSpeech
from modules.tts.iclspeech.vqvae.vqvae import VectorQuantizedVAE
from modules.tts.iclspeech.icl_vqvae_lm import VQLanguageModel
from utils.commons.hparams import hparams

class VQLanguageModel_MultiLingual(VQLanguageModel):
    def forward(self, ph_tokens_gen, ph_tokens_prompt, prev_vq_code, spk_embed, use_prosody_prompt, incremental_state=None, ret=None):
        # run encoder
        x = self.vqcode_emb(prev_vq_code)
        if use_prosody_prompt:
            ph_embed_gen = self.ph_encoder(ph_tokens_gen)
            ph_embed_prompt = self.ph_encoder(ph_tokens_prompt)
            ph_embed = torch.cat((ph_embed_prompt,ph_embed_gen), dim=1)
        else:
            ph_embed_gen = self.ph_encoder(ph_tokens_gen)
            ph_embed = ph_embed_gen

        if self.spk_mode == 'direct':
            # Currently we only support one-sentence prompt based zero-shot generation
            # The spk_embed is obtained from the one-sentence mel prompt
            # Thus, we do not use attention mechanics here
            ph_embed = ph_embed + self.spk_embed_proj(spk_embed)

        # run decoder
        if incremental_state is not None:
            positions = self.embed_positions(
                prev_vq_code,
                incremental_state=incremental_state
            )
            ph_embed = ph_embed[:, x.shape[1] - 1:x.shape[1]]
            x = x[:, -1:]
            positions = positions[:, -1:]
            self_attn_padding_mask = None
        else:
            positions = self.embed_positions(
                prev_vq_code,
                incremental_state=incremental_state
            )
            self_attn_padding_mask = None

        x += positions
        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        ph_embed = ph_embed.transpose(0, 1)
        x = x + ph_embed

        for layer in self.layers:
            if incremental_state is None:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            x, attn_logits = layer(
                x,
                incremental_state=incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
            )

        x = self.layer_norm(x)
        # T x B x C -> B x T x C
        x = x.transpose(0, 1)
        x = self.project_out_dim(x)
        return x

    def infer(self, ph_tokens_gen, ph_tokens_prompt, ph_vqcode, spk_embed, prompt_length, use_prosody_prompt, ret):
        # mode = one-sentence prompt, zero-shot generation
        incremental_state = None
        # Add prompt
        if use_prosody_prompt:
            vq_decoded = torch.zeros([1, ph_tokens_prompt.shape[1]+ph_tokens_gen.shape[1]], dtype=torch.long)
            vq_decoded[:, :prompt_length] = ph_vqcode[:, :prompt_length]
            start_step = prompt_length
        else:
            vq_decoded = torch.zeros([1, ph_tokens_gen.shape[1]], dtype=torch.long)
            start_step = 0
        # Start Decode
        vq_decoded = F.pad(vq_decoded, [1, 0], value=hparams['vqvae_ph_codebook_dim'] + 1)
        for step in range(start_step, vq_decoded.shape[1] - 1):
            print(f'{step}/{vq_decoded.shape[1] - 1}')
            vq_pred = self(ph_tokens_gen, ph_tokens_prompt, vq_decoded[:, :-1], spk_embed, use_prosody_prompt,
                           incremental_state=incremental_state, ret=ret)
            vq_pred = torch.argmax(F.softmax(vq_pred, dim=-1), -1)
            vq_decoded[:, step + 1] = vq_pred[:, step]
        return vq_decoded[:, 1:]

class ICLVectorQuantizedVAELM(FastSpeech):
    def __init__(self, ph_dict_size, hparams, out_dims=None):
        super().__init__(ph_dict_size, hparams, out_dims)
        # build VAE decoder
        del self.decoder
        del self.mel_out
        self.vqvae = VectorQuantizedVAE(hparams)
        self.vq_lm = VQLanguageModel_MultiLingual(ph_dict_size)
        if hparams['use_spk_embed']:
            self.spk_embed_proj = nn.Linear(256, self.hidden_size)

    def forward(self,  txt_tokens_gen, txt_tokens_prompt, mel2ph=None, mel2ph_prompt=None, infer=False, tgt_mels=None,
                mel_prompt=None, spk_embed_prompt=None, use_prosody_prompt=True, *args, **kwargs):
        # Only support inference
        ret = {}
        prompt_src_nonpadding = (txt_tokens_prompt > 0).float()[:, :, None]

        # Infer with pred VQCode
        in_nonpadding = (mel2ph_prompt > 0).float()[:, :, None]
        # Get GT VQCode for the first sentence
        ph_vqcode = self.vqvae.encode_ph_vqcode(mel_prompt, in_nonpadding.transpose(1,2), mel2ph_prompt, txt_tokens_prompt.shape[1], prompt_src_nonpadding.transpose(1,2))
        spk_embed = self.vqvae.encode_spk_embed(mel_prompt)
        if spk_embed_prompt != None:
            spk_embed = self.spk_embed_proj(spk_embed_prompt)[:,:,None]
        # Infer VQCode for the second sentence
        ph_vqcode = (ph_vqcode.detach() + 1)
        vq_codes_pred = self.vq_lm.infer(txt_tokens_gen, txt_tokens_prompt, ph_vqcode, spk_embed.transpose(1,2), txt_tokens_prompt.shape[1], use_prosody_prompt, ret)
        z_q_x_bar = self.vqvae.vqcode_to_latent((vq_codes_pred - 1).clamp_min(0))
        if use_prosody_prompt:
            z_q_x_bar = z_q_x_bar[:,:,txt_tokens_prompt.shape[1]:]

        # Infer mel with pred VQCode
        ph_encoder_out = self.encoder(txt_tokens_gen)
        dur_inp = (ph_encoder_out + z_q_x_bar.transpose(1,2))
        out_mel2ph = self.forward_dur(dur_inp, None, txt_tokens_gen, ret)
        ret['out_mel2ph'] = out_mel2ph
        out_nonpadding = (out_mel2ph > 0).float()[:, :, None]
        txt_cond = expand_states(ph_encoder_out, out_mel2ph)
        txt_cond = txt_cond * out_nonpadding
        x_tilde = self.vqvae.forward_second_stage(txt_cond, z_q_x_bar, spk_embed, out_nonpadding.transpose(1,2), out_mel2ph)
        ret['x_tilde'] = x_tilde

        ret['vq_codes_pred'], ret['vq_codes'] = vq_codes_pred, ph_vqcode
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