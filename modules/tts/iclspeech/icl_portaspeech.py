import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear

from modules.commons.layers import Embedding
from modules.commons.transformer import MultiheadAttention, FFTBlocks
from modules.tts.commons.align_ops import clip_mel2token_to_multiple, build_word_mask, expand_states, mel2ph_to_mel2word
from modules.tts.fs import FS_DECODERS, FastSpeech
from modules.tts.iclspeech.fvae import FVAE
from modules.tts.iclspeech.positional_encoding import RelPositionalEncoding
from modules.tts.iclspeech.attention.simple_attention import SimpleAttention, split_heads
from modules.commons.transformer import SinusoidalPositionalEmbedding
from utils.commons.meters import Timer


class ICLPortaSpeech(FastSpeech):
    def __init__(self, ph_dict_size, word_dict_size, hparams, out_dims=None):
        super().__init__(ph_dict_size, hparams, out_dims)
        # build VAE decoder
        self.decoder = FS_DECODERS[hparams['decoder_type']](hparams)
        self.mel_out = Linear(self.hidden_size, self.out_dims, bias=True)
        if hparams['use_pitch_embed']:
            self.pitch_embed = Embedding(300, self.hidden_size, 0)
        if self.hparams['add_word_pos']:
            self.word_pos_proj = Linear(self.hidden_size, self.hidden_size)
        
        # Build VQ latents attention
        self.ph_attn = SimpleAttention(num_heads=4)
        self.spk_attn = SimpleAttention(num_heads=1)
        self.pos_embed_alpha = nn.Parameter(torch.Tensor([1]))
        self.embed_positions = SinusoidalPositionalEmbedding(
                self.hidden_size, 0, init_size=5000,
        )
        self.query_proj_in = nn.Linear(hparams['hidden_size'], self.hidden_size)
        self.key_proj_in = nn.Linear(hparams['hidden_size'], self.hidden_size)
        self.value_ph_proj_in = nn.Linear(hparams['hidden_size'], self.hidden_size)
        self.value_spk_proj_in = nn.Linear(hparams['hidden_size'], self.hidden_size)
        self.value_spk_proj_out = nn.Linear(hparams['hidden_size'], self.hidden_size)
        self.use_vq_attn = self.hparams['use_vq_attn']

    def build_embedding(self, dictionary, embed_dim):
        num_embeddings = len(dictionary)
        emb = Embedding(num_embeddings, embed_dim, self.padding_idx)
        return emb

    def forward(self, txt_tokens, ph_lengths=None, mel2ph=None, vq_text_cond=None, latents=None, attn_mask=None, spk_attn_mask=None,
                spk_embed=None, spk_id=None, pitch=None, vq_attn_mode=None, tgt_mels=None,
                global_step=None, *args, **kwargs):
        ret = {}
        x, tgt_nonpadding = self.run_text_encoder(
            txt_tokens, ph_lengths, mel2ph, vq_text_cond, latents, attn_mask, spk_attn_mask, vq_attn_mode, ret)
        ret['nonpadding'] = tgt_nonpadding
        ret['decoder_inp'] = x
        ret['mel_out_fvae'] = ret['mel_out'] = self.run_decoder(x, tgt_nonpadding, latents, ret, tgt_mels, global_step)
        return ret

    def run_text_encoder(self, txt_tokens, ph_lengths, mel2ph, vq_text_cond, latents, attn_mask, spk_attn_mask, vq_attn_mode, ret):
        src_nonpadding = (txt_tokens > 0).float()[:, :, None]
        ph_encoder_out = self.encoder(txt_tokens) * src_nonpadding
        
        if self.use_vq_attn:
            if vq_attn_mode == 'paragraph':
                #### VQ latents attention (phoneme-level)
                # Get positional embedding (considering the pad token)
                B, T, C = ph_encoder_out.shape
                ph_pos = (src_nonpadding.view(1, B*T, 1).cumsum(dim=1) * src_nonpadding.view(1, B*T, 1)).long().contiguous()
                ph_pos = self.pos_embed_alpha * self.embed_positions(ph_encoder_out.reshape(1, B*T, C), positions=ph_pos)
                
                # Add positional embedding to q, k, v
                query = ph_encoder_out.reshape(1, B*T, C).contiguous() + ph_pos
                key = vq_text_cond.reshape(1, B*T, C).contiguous() + ph_pos
                value_ph = self.value_ph_proj_in(latents[0].transpose(1, 2).reshape(1, B*T, C)).contiguous() + ph_pos
                
                # Attention
                x_residual = query
                x_ph, weight_ph = self.ph_attn(query, key, value_ph, attn_mask=attn_mask)
                x = (x_residual + x_ph).view(B, T, C) * src_nonpadding
                x = x.contiguous()
                ret['vq_attn_ph'] = weight_ph

            elif vq_attn_mode == 'intra-sentence':
                #### VQ latents attention (phoneme-level)
                # Get positional embedding (considering the pad token)
                B, T, C = ph_encoder_out.shape
                ph_pos = (src_nonpadding.cumsum(dim=1) * src_nonpadding).long()
                ph_pos = self.pos_embed_alpha * self.embed_positions(ph_encoder_out, positions=ph_pos)
                
                # Add positional embedding to q, k, v
                query = ph_encoder_out + ph_pos
                key = vq_text_cond + ph_pos
                value_ph = self.value_ph_proj_in(latents[0].transpose(1, 2)).contiguous() + ph_pos
                
                # Attention
                x_residual = query
                x_ph, weight_ph = self.ph_attn(query, key, value_ph, attn_mask=attn_mask)
                x = (x_residual + x_ph) * src_nonpadding
                ret['vq_attn_ph'] = weight_ph

            else:
                raise Exception("Please specify the vq_attn_mode correctly.") 
        else:
            x = ph_encoder_out
        
        # Speaker embed attention
        query_spk = ph_encoder_out.sum(dim=1) / src_nonpadding.sum(dim=1)
        key_spk = vq_text_cond.sum(dim=1) / src_nonpadding.sum(dim=1)
        value_spk = self.value_spk_proj_in(latents[1].transpose(1, 2)) # [B, 1, C]
        query_spk, key_spk, value_spk = query_spk[None, :, :], key_spk[None, :, :], value_spk.transpose(0, 1)
        x_spk, weight_spk = self.spk_attn(query_spk.clone().detach(), key_spk, value_spk, attn_mask=spk_attn_mask[None, :, :])
        ret['vq_attn_spk'] = weight_spk
        x = (x + x_spk.transpose(0, 1)) * src_nonpadding

        # Forward duration
        mel2ph = self.forward_dur(x, mel2ph, ret)
        mel2ph = clip_mel2token_to_multiple(mel2ph, self.hparams['frames_multiple'])
        x = expand_states(x, mel2ph)

        # Add positional encoding to z_q_x_st
        tgt_nonpadding = (mel2ph > 0).float()[:, :, None]
        pos_emb_mel = (tgt_nonpadding.cumsum(dim=1) * tgt_nonpadding).long()
        pos_emb_mel = self.pos_embed_alpha * self.embed_positions(x, positions=pos_emb_mel)
        x = (x + pos_emb_mel) * tgt_nonpadding
        return x, tgt_nonpadding

    def run_decoder(self, x, tgt_nonpadding, latents, ret, tgt_mels=None, global_step=0):
        x = self.decoder(x)
        x = self.mel_out(x)
        ret['kl'] = 0
        return x * tgt_nonpadding

    def forward_dur(self, dur_input, mel2ph, ret, **kwargs):
        """

        :param dur_input: [B, T_txt, H]
        :param mel2ph: [B, T_mel]
        :param txt_tokens: [B, T_txt]
        :param ret:
        :return:
        """
        src_padding = dur_input.data.abs().sum(-1) == 0
        dur_input = dur_input.detach() + self.hparams['predictor_grad'] * (dur_input - dur_input.detach())
        dur = self.dur_predictor(dur_input, src_padding)
        ret['dur'] = dur
        if mel2ph is None:
            mel2ph = self.length_regulator(dur, src_padding).detach()
        ret['mel2ph'] = mel2ph = clip_mel2token_to_multiple(mel2ph, self.hparams['frames_multiple'])
        return mel2ph

    def get_pos_embed(self, word2word, x2word):
        x_pos = build_word_mask(word2word, x2word).float()  # [B, T_word, T_ph]
        x_pos = (x_pos.cumsum(-1) / x_pos.sum(-1).clamp(min=1)[..., None] * x_pos).sum(1)
        x_pos = self.sin_pos(x_pos.float())  # [B, T_ph, H]
        return x_pos

    def store_inverse_all(self):
        def remove_weight_norm(m):
            try:
                if hasattr(m, 'store_inverse'):
                    m.store_inverse()
                nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(remove_weight_norm)

    def temporal_avg_pool(self, x, mask=None):
        len_ = (~mask).sum(dim=-1).unsqueeze(-1)
        x = x.masked_fill(mask, 0)
        x = x.sum(dim=-1).unsqueeze(-1)
        out = torch.div(x, len_)
        return out