import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from modules.commons.layers import Embedding
from modules.tts.commons.align_ops import clip_mel2token_to_multiple, expand_states
from modules.tts.fs import FS_ENCODERS, FastSpeech
from modules.tts.iclspeech.vqvae.vqvae import VectorQuantizedVAE
from modules.tts.iclspeech.attention.simple_attention import SimpleAttention
from modules.tts.iclspeech.reltransformer_controlnet import RelTransformerEncoder_ControlNet
from modules.commons.rel_transformer import RelTransformerEncoder
from modules.commons.layers import LayerNorm, Linear
# from modules.tts.iclspeech.flash_transformer import TransformerDecoderLayer, SinusoidalPositionalEmbedding
from modules.commons.transformer import TransformerDecoderLayer, SinusoidalPositionalEmbedding
from utils.commons.hparams import hparams


class VQLanguageModel(nn.Module):
    def __init__(self, dict_size, hidden_size=320):
        super().__init__()
        self.hidden_size = hidden_size
        self.ph_encoder = RelTransformerEncoder(
        dict_size, hidden_size, hidden_size,
        hidden_size*4, hparams['num_heads'], hparams['enc_layers'],
        hparams['enc_ffn_kernel_size'], hparams['dropout'], prenet=hparams['enc_prenet'], pre_ln=hparams['enc_pre_ln'])
        self.vqcode_emb = Embedding(hparams['vqvae_ph_codebook_dim'] + 2, hidden_size, 0)
        self.embed_positions = SinusoidalPositionalEmbedding(hidden_size, 0, init_size=1024)
        dec_num_layers = 6
        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerDecoderLayer(hidden_size, 0., kernel_size=5) for _ in
            range(dec_num_layers)
        ])
        self.layer_norm = LayerNorm(hidden_size)
        self.project_out_dim = Linear(hidden_size, hparams['vqvae_ph_codebook_dim'] + 1, bias=True)

        # Speaker embed related
        self.spk_embed_proj = Linear(320, hidden_size, bias=True)
        self.spk_mode = 'direct' # 'direct' or 'attn'

    def forward(self, ph_tokens, prev_vq_code, spk_embed, incremental_state=None, ret=None):
        # run encoder
        x = self.vqcode_emb(prev_vq_code)
        src_nonpadding = (ph_tokens > 0).float()[:, :, None]
        ph_embed = self.ph_encoder(ph_tokens) * src_nonpadding

        if self.spk_mode == 'direct':
            # Currently we only support one-sentence prompt based zero-shot generation
            # The spk_embed is obtained from the one-sentence mel prompt
            # Thus, we do not use attention mechanics here
            ph_embed = ph_embed + self.spk_embed_proj(spk_embed)
            ph_embed = ph_embed * src_nonpadding

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
            self_attn_padding_mask = ph_tokens.eq(0).data

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

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if (
                not hasattr(self, '_future_mask')
                or self._future_mask is None
                or self._future_mask.device != tensor.device
                or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(self.fill_with_neg_inf2(tensor.new(dim, dim)), 1)
        return self._future_mask[:dim, :dim]

    def infer(self, ph_tokens, ph_vqcode, spk_embed, sent_lengths, ret):
        # mode = one-sentence prompt, zero-shot generation
        incremental_state = {}
        # Add prompt
        vq_decoded = torch.zeros_like(ph_tokens)
        vq_decoded[:, :sent_lengths[0]] = ph_vqcode[:, :sent_lengths[0]]
        # Start Decode
        vq_decoded = F.pad(vq_decoded, [1, 0], value=hparams['vqvae_ph_codebook_dim'] + 1)
        decoder_output_hiddens = []
        for step in range(sent_lengths[0], vq_decoded.shape[1] - 1):
            print(f'{step}/{vq_decoded.shape[1] - 1}')
            vq_pred = self(ph_tokens, vq_decoded[:, :step + 1], spk_embed,
                           incremental_state=incremental_state, ret=ret)
            decoder_output_hiddens.append(vq_pred[:, -1])
            if hparams['infer_argmax']:
                vq_pred = torch.argmax(F.softmax(vq_pred[:, -1], dim=-1), 1)
            else:
                vq_pred = torch.multinomial(F.softmax(vq_pred[:, -1], dim=-1), 1).squeeze(1)
            vq_decoded[:, step + 1] = vq_pred
        return vq_decoded[:, 1:]

    def fill_with_neg_inf2(self, t):
        """FP16-compatible function that fills a tensor with -inf."""
        return t.float().fill_(-1e8).type_as(t)
    

class ICLVectorQuantizedVAELM(FastSpeech):
    def __init__(self, ph_dict_size, hparams, out_dims=None):
        super().__init__(ph_dict_size, hparams, out_dims)
        # build VAE decoder
        del self.decoder
        del self.mel_out
        self.vqvae = VectorQuantizedVAE(hparams)
        self.vq_lm = VQLanguageModel(ph_dict_size)

    def forward(self, txt_tokens, sent_txt_lengths, mel2ph=None, infer=False, tgt_mels=None,
                mel_prompt=None, global_step=None, ph_lengths=None, use_gt_mel2ph=True, *args, **kwargs):
        ret = {}
        src_nonpadding = (txt_tokens > 0).float()[:, :, None]

        # Forward LM
        if not infer:
            in_mel2ph = mel2ph
            in_nonpadding = (in_mel2ph > 0).float()[:, :, None]
            # Get GT VQCode
            ph_vqcode = self.vqvae.encode_ph_vqcode(tgt_mels, in_nonpadding.transpose(1,2), in_mel2ph, ph_lengths)
            spk_embed = self.vqvae.encode_spk_embed(mel_prompt)
            # Forward VQ LM
            ph_vqcode = (ph_vqcode.detach() + 1) * src_nonpadding.squeeze(-1).long()
            prev_ph_vqcode = F.pad(ph_vqcode[:, :-1], [1, 0], value=hparams['vqvae_ph_codebook_dim'] + 1)
            vq_codes_pred = self.vq_lm(txt_tokens, prev_ph_vqcode, spk_embed.transpose(1,2), ret=ret)

        else:
            # Infer with pred VQCode
            in_mel2ph = mel2ph
            in_nonpadding = (in_mel2ph > 0).float()[:, :, None]
            # Get GT VQCode for the first sentence
            ph_vqcode = self.vqvae.encode_ph_vqcode(tgt_mels, in_nonpadding.transpose(1,2), in_mel2ph, ph_lengths)
            spk_embed = self.vqvae.encode_spk_embed(mel_prompt)
            ph_vqcode = (ph_vqcode.detach() + 1) * src_nonpadding.squeeze(-1).long()
            # Infer VQCode for the second sentence
            vq_codes_pred = self.vq_lm.infer(txt_tokens, ph_vqcode, spk_embed.transpose(1,2), sent_txt_lengths, ret)
            z_q_x_bar = self.vqvae.vqcode_to_latent((vq_codes_pred - 1).clamp_min(0))

            # Infer with GT VQCode
            # in_mel2ph = mel2ph
            # in_nonpadding = (in_mel2ph > 0).float()[:, :, None]
            # ph_vqcode, _, global_z_q_x_bar = self.vqvae.encode_to_vqcode(tgt_mels, in_nonpadding.transpose(1,2), in_mel2ph, ph_lengths)
            # vq_codes_pred = self.vq_lm.infer(txt_tokens, ph_vqcode, global_z_q_x_bar.transpose(1,2), sent_lengths)
            # z_q_x_bar = self.vqvae.vqcode_to_latent(ph_vqcode)
            
            # Infer mel with pred VQCode
            ph_encoder_out = self.encoder(txt_tokens) * src_nonpadding
            if use_gt_mel2ph:
                out_mel2ph = mel2ph
            else:
                dur_inp = (ph_encoder_out + z_q_x_bar.transpose(1,2) + spk_embed.transpose(1,2)) * src_nonpadding
                out_mel2ph = self.forward_dur(dur_inp, None, txt_tokens, ret)
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