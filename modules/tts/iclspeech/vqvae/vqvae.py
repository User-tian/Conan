import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence

from modules.tts.iclspeech.leftpad_conv import ConvBlocks as LeftPadConvBlocks
from modules.commons.conv import ConvBlocks, ConditionalConvBlocks
from modules.tts.iclspeech.spk_encoder.stylespeech_encoder import MelStyleEncoder
from modules.tts.iclspeech.vqvae.vq_functions import vq, vq_st, vq_st_test_global, vq_st_test_ph
from modules.tts.commons.align_ops import clip_mel2token_to_multiple, expand_states
from utils.nn.seq_utils import group_hidden_by_segs
from modules.commons.transformer import SinusoidalPositionalEmbedding


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            print("Skipping initialization of ", classname)


class VQEmbedding(nn.Module):
    def __init__(self, K, D):
        super().__init__()
        self.embedding = nn.Embedding(K, D)
        self.embedding.weight.data.uniform_(-1./K, 1./K)

    def forward(self, z_e_x):
        # z_e_x: (B, C, T)
        # output: (B, T, C)
        z_e_x_ = z_e_x.permute(0, 2, 1).contiguous()
        indices = vq(z_e_x_, self.embedding.weight)
        return indices

    def straight_through(self, z_e_x):
        # z_e_x: (B, C, T)
        # output: (B, C, T), (B, C, T)
        z_e_x_ = z_e_x.permute(0, 2, 1).contiguous()
        z_q_x_, indices = vq_st(z_e_x_, self.embedding.weight.detach())
        z_q_x = z_q_x_.permute(0, 2, 1).contiguous()

        z_q_x_bar_flatten = torch.index_select(self.embedding.weight,
            dim=0, index=indices)
        z_q_x_bar_ = z_q_x_bar_flatten.view_as(z_e_x_)
        z_q_x_bar = z_q_x_bar_.permute(0, 2, 1).contiguous()

        return z_q_x, z_q_x_bar


class VectorQuantizedVAE(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        input_dim = hparams['vqvae_input_dim']
        hidden_size = c_cond = hparams['hidden_size']
        self.frames_multiple = hparams['frames_multiple']
        self.vqvae_ph_channel = hparams['vqvae_ph_channel']

        self.ph_conv_in = nn.Conv1d(20, hidden_size, 1)
        self.global_conv_in = nn.Conv1d(input_dim, hidden_size, 1)
        self.ph_encoder = LeftPadConvBlocks(
                            hidden_size, hidden_size, None, kernel_size=5,
                            layers_in_block=2, is_BTC=False, num_layers=5)
        if hparams.get('use_ph_postnet', False):
            self.ph_postnet = LeftPadConvBlocks(
                            hidden_size, hparams['vqvae_ph_channel'], None, kernel_size=5,
                            layers_in_block=2, is_BTC=False, num_layers=5)
        self.global_encoder = ConvBlocks(
                            hidden_size, hidden_size, None, kernel_size=31,
                            layers_in_block=2, is_BTC=False, num_layers=5)
        
        self.ph_latents_proj_in = nn.Conv1d(hidden_size, hparams['vqvae_ph_channel'], 1)
        self.ph_codebook = VQEmbedding(hparams['vqvae_ph_codebook_dim'], hparams['vqvae_ph_channel'])
        self.ph_latents_proj_out = nn.Conv1d(hparams['vqvae_ph_channel'], hidden_size, 1)
        self.decoder = ConditionalConvBlocks(
                            hidden_size, c_cond, hidden_size, [1] * 5, kernel_size=5,
                            layers_in_block=2, is_BTC=False)
        self.conv_out = nn.Conv1d(hidden_size, input_dim, 1)
        self.apply(weights_init)
        self.pos_embed_alpha = nn.Parameter(torch.Tensor([1]))
        self.embed_positions = SinusoidalPositionalEmbedding(
                hidden_size, 0, init_size=5000,
        )

    def encode_ph_vqcode(self, x, in_nonpadding, in_mel2ph, max_ph_length, ph_nonpadding):
        # forward encoder
        x_ph = self.ph_conv_in(x[:,:20,:]) * in_nonpadding
        ph_z_e_x = self.ph_encoder(x_ph, nonpadding=in_nonpadding) * in_nonpadding # (B, C, T)
        # Forward ph postnet
        if self.hparams.get('use_ph_postnet', False):
            ph_z_e_x = group_hidden_by_segs(ph_z_e_x, in_mel2ph, max_ph_length, is_BHT=True)[0]
            ph_z_e_x = self.ph_postnet(ph_z_e_x, nonpadding=ph_nonpadding) * ph_nonpadding
            ph_vqcode = self.ph_codebook(ph_z_e_x)
        else:
            # group by hidden to phoneme-level
            ph_z_e_x = self.ph_latents_proj_in(ph_z_e_x)
            ph_z_e_x = group_hidden_by_segs(ph_z_e_x, in_mel2ph, max_ph_length, is_BHT=True)[0]
            ph_vqcode = self.ph_codebook(ph_z_e_x)
        return ph_vqcode

    def encode_spk_embed(self, x):
        in_nonpadding = (x.abs().sum(dim=-2) > 0).float()[:, None, :]
        # forward encoder
        x_global = self.global_conv_in(x) * in_nonpadding
        global_z_e_x = self.global_encoder(x_global, nonpadding=in_nonpadding) * in_nonpadding
        # group by hidden to phoneme-level
        global_z_e_x = self.temporal_avg_pool(x=global_z_e_x, mask=(in_nonpadding==0)) # (B, C, T) -> (B, C, 1)
        spk_embed = global_z_e_x
        return spk_embed

    def vqcode_to_latent(self, ph_vqcode):
        # VQ process
        z_q_x_bar_flatten = torch.index_select(self.ph_codebook.embedding.weight,
            dim=0, index=ph_vqcode.view(-1))
        ph_z_q_x_bar_ = z_q_x_bar_flatten.view(ph_vqcode.size(0), ph_vqcode.size(1), self.vqvae_ph_channel)
        ph_z_q_x_bar = ph_z_q_x_bar_.permute(0, 2, 1).contiguous()
        ph_z_q_x_bar = self.ph_latents_proj_out(ph_z_q_x_bar)
        return ph_z_q_x_bar

    def decode(self, latents, mel2ph):
        raise NotImplementedError

    def temporal_avg_pool(self, x, mask=None):
        len_ = (~mask).sum(dim=-1).unsqueeze(-1)
        x = x.masked_fill(mask, 0)
        x = x.sum(dim=-1).unsqueeze(-1)
        out = torch.div(x, len_)
        return out

    def forward_first_stage(self, x, x_prompt, in_nonpadding, in_nonpadding_prompt, in_mel2ph, ph_nonpadding, ph_lengths):
        # forward encoder
        x_ph = self.ph_conv_in(x[:,:20,:]) * in_nonpadding
        ph_z_e_x = self.ph_encoder(x_ph, nonpadding=in_nonpadding) * in_nonpadding # (B, C, T)
        x_global = self.global_conv_in(x_prompt) * in_nonpadding_prompt
        global_z_e_x = self.global_encoder(x_global, nonpadding=in_nonpadding_prompt) * in_nonpadding_prompt

        # Forward ph postnet
        if self.hparams.get('use_ph_postnet', False):
            ph_z_e_x = group_hidden_by_segs(ph_z_e_x, in_mel2ph, ph_lengths.max(), is_BHT=True)[0]
            ph_z_e_x = self.ph_postnet(ph_z_e_x, nonpadding=ph_nonpadding) * ph_nonpadding
            global_z_e_x = self.temporal_avg_pool(x=global_z_e_x, mask=(in_nonpadding_prompt==0)) # (B, C, T) -> (B, C, 1)
        else:
            # group by hidden to phoneme-level
            ph_z_e_x = self.ph_latents_proj_in(ph_z_e_x)
            ph_z_e_x = group_hidden_by_segs(ph_z_e_x, in_mel2ph, ph_lengths.max(), is_BHT=True)[0]
            global_z_e_x = self.temporal_avg_pool(x=global_z_e_x, mask=(in_nonpadding_prompt==0)) # (B, C, T) -> (B, C, 1)

        # VQ process
        ph_z_q_x_st, ph_z_q_x = self.ph_codebook.straight_through(ph_z_e_x)
        ph_z_q_x_st = self.ph_latents_proj_out(ph_z_q_x_st)
        global_z_q_x_st = global_z_e_x
        return ph_z_e_x, ph_z_q_x, ph_z_q_x_st, global_z_q_x_st

    def forward_second_stage(self, txt_cond, ph_z_q_x_st, global_z_q_x_st, out_nonpadding, out_mel2ph):
        # expand hidden to frame-level
        ph_z_q_x_st = expand_states(ph_z_q_x_st.transpose(1, 2), out_mel2ph).transpose(1, 2)

        # combine ph-level and global-level latents
        z_q_x_st = ph_z_q_x_st + global_z_q_x_st

        # Add positional encoding to z_q_x_st
        txt_cond = txt_cond.transpose(1,2)
        nonpadding_BTC = out_nonpadding.transpose(1, 2)
        pos_emb = (nonpadding_BTC.cumsum(dim=1) * nonpadding_BTC).long()
        pos_emb = self.pos_embed_alpha * self.embed_positions(z_q_x_st.transpose(1, 2), positions=pos_emb)
        pos_emb = pos_emb.transpose(1, 2).contiguous()
        z_q_x_st = z_q_x_st + pos_emb
        txt_cond = txt_cond + pos_emb

        # forward decoder
        x_tilde = self.decoder(z_q_x_st, cond=txt_cond, nonpadding=out_nonpadding) * out_nonpadding
        x_tilde = self.conv_out(x_tilde) * out_nonpadding
        return x_tilde