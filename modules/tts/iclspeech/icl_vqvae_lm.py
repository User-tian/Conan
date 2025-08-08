import math
import random
import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm
import torch.utils.data as tud
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

def beam_search(
    model, 
    ph_tokens, 
    prev_vq_code, 
    spk_embed, 
    predictions = 20,
    beam_width = 3,
    batch_size = 1, 
    progress_bar = 0
):
    """
    Implements Beam Search to extend the sequences given in X. The method can compute 
    several outputs in parallel with the first dimension of X.

    Parameters
    ----------    
    X: LongTensor of shape (examples, length)
        The sequences to start the decoding process.

    predictions: int
        The number of tokens to append to X.

    beam_width: int
        The number of candidates to keep in the search.

    batch_size: int
        The batch size of the inner loop of the method, which relies on the beam width. 

    progress_bar: bool
        Shows a tqdm progress bar, useful for tracking progress with large tensors.

    Returns
    -------
    X: LongTensor of shape (examples, length + predictions)
        The sequences extended with the decoding process.

    probabilities: FloatTensor of length examples
        The estimated log-probabilities for the output sequences. They are computed by iteratively adding the 
        probability of the next token at every step.
    """
    with torch.no_grad():
        # The next command can be a memory bottleneck, but can be controlled with the batch 
        # size of the predict method.
        next_probabilities = model.forward(ph_tokens, prev_vq_code, spk_embed)[:, -predictions, :]
        vocabulary_size = next_probabilities.shape[-1]
        probabilities, idx = next_probabilities.squeeze().log_softmax(-1)\
            .topk(k = beam_width, axis = -1)
        prev_vq_code = prev_vq_code.repeat((beam_width, 1, 1)).transpose(0, 1)\
            .flatten(end_dim = -2)
        ph_tokens = ph_tokens.repeat((beam_width, 1, 1)).transpose(0, 1)\
            .flatten(end_dim = -2)
        prev_vq_code[:, -predictions] = idx
        # This has to be minus one because we already produced a round
        # of predictions before the for loop.
        predictions_iterator = range(predictions - 1)
        if progress_bar > 0:
            predictions_iterator = tqdm(predictions_iterator)
        for i in predictions_iterator:
            next_probabilities = model.forward(ph_tokens, prev_vq_code, spk_embed)[:, -predictions+i+1, :].squeeze().log_softmax(-1)
            # next_probabilities = next_probabilities.reshape(
                # (-1, beam_width, next_probabilities.shape[-1])
            # )
            probabilities = probabilities.unsqueeze(-1) + next_probabilities
            probabilities, idx = probabilities.topk(
                k = beam_width, 
                axis = -1
            )
            # next_chars = torch.remainder(idx, vocabulary_size).flatten()\
                # .unsqueeze(-1)
            # best_candidates = (idx / vocabulary_size).long()
            # best_candidates += torch.arange(
                # X.shape[0] // beam_width, 
                # device = X.device
            # ).unsqueeze(-1) * beam_width
            # X = X[best_candidates].flatten(end_dim = -2)
            prev_vq_code = prev_vq_code.repeat((beam_width, 1, 1)).transpose(0, 1)
            print(prev_vq_code[:, :, -predictions+i+1].shape)
            print(idx.shape)
            prev_vq_code[:, :, -predictions+i+1] = idx
            prev_vq_code = prev_vq_code.flatten(end_dim = -2)
            ph_tokens = ph_tokens.repeat((beam_width, 1, 1)).transpose(0, 1)\
                .flatten(end_dim = -2)
            probabilities = probabilities.flatten()
        print(prev_vq_code.shape)
        import sys
        sys.exit(0)
        torch.argmax(F.softmax(vq_pred, dim=-1), -1)
        return prev_vq_code

class VQLanguageModel(nn.Module):
    def __init__(self, dict_size):
        super().__init__()
        self.hidden_size = hidden_size = hparams['lm_hidden_size']
        self.ph_encoder = RelTransformerEncoder(
        dict_size, hidden_size, hidden_size,
        hidden_size*4, hparams['num_heads'], hparams['enc_layers'],
        hparams['enc_ffn_kernel_size'], hparams['dropout'], prenet=hparams['enc_prenet'], pre_ln=hparams['enc_pre_ln'])
        self.vqcode_emb = Embedding(hparams['vqvae_ph_codebook_dim'] + 2, hidden_size, 0)
        self.embed_positions = SinusoidalPositionalEmbedding(hidden_size, 0, init_size=1024)
        dec_num_layers = 8
        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerDecoderLayer(hidden_size, 0., kernel_size=5, num_heads=8) for _ in
            range(dec_num_layers)
        ])
        self.layer_norm = LayerNorm(hidden_size)
        self.project_out_dim = Linear(hidden_size, hparams['vqvae_ph_codebook_dim'] + 1, bias=True)

        # Speaker embed related
        self.spk_embed_proj = Linear(hparams['hidden_size'], hidden_size, bias=True)
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

    def infer(self, ph_tokens, ph_vqcode, spk_embed, prompt_length, ret, mode='argmax'):
        # mode = one-sentence prompt, zero-shot generation
        incremental_state = None
        # Add prompt
        vq_decoded = torch.zeros_like(ph_tokens)
        vq_decoded[:, :prompt_length] = ph_vqcode[:, :prompt_length]
        # Start Decode
        vq_decoded = F.pad(vq_decoded, [1, 0], value=hparams['vqvae_ph_codebook_dim'] + 1)
        if mode == 'argmax':
            for step in range(prompt_length, vq_decoded.shape[1] - 1):
                print(f'{step}/{vq_decoded.shape[1] - 1}')
                vq_pred = self(ph_tokens, vq_decoded[:, :-1], spk_embed,
                            incremental_state=incremental_state, ret=ret)
                vq_pred = torch.argmax(F.softmax(vq_pred, dim=-1), -1)
                vq_decoded[:, step + 1] = vq_pred[:, step]
        elif mode == 'topk':
            K = 10
            for step in range(prompt_length, vq_decoded.shape[1] - 1):
                print(f'{step}/{vq_decoded.shape[1] - 1}')
                vq_pred = self(ph_tokens, vq_decoded[:, :-1], spk_embed,
                            incremental_state=incremental_state, ret=ret)
                _, idx = F.softmax(vq_pred, dim=-1).topk(k = K, axis = -1)
                rand_idx = random.randint(0,K-1)
                vq_decoded[:, step + 1] = idx[:, step, rand_idx]
        else:
            # Buggy
            predictions = beam_search(self, ph_tokens, vq_decoded[:, :-1], spk_embed, predictions=vq_decoded.shape[1]-prompt_length)
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
        if hparams['use_spk_embed']:
            self.spk_embed_proj = nn.Linear(256, self.hidden_size)

    def forward(self, txt_tokens, txt_tokens_gen, txt_tokens_prompt, mel2ph=None, mel2ph_prompt=None, infer=False, tgt_mels=None,
                mel_prompt=None, spk_embed_prompt=None, global_step=None, use_gt_mel2ph=True, *args, **kwargs):
        ret = {}
        src_nonpadding = (txt_tokens > 0).float()[:, :, None]
        prompt_src_nonpadding = (txt_tokens_prompt > 0).float()[:, :, None]

        # Forward LM
        if not infer:
            in_mel2ph = mel2ph
            in_nonpadding = (in_mel2ph > 0).float()[:, :, None]
            # Get GT VQCode
            ph_vqcode = self.vqvae.encode_ph_vqcode(tgt_mels, in_nonpadding.transpose(1,2), in_mel2ph, txt_tokens.shape[1], src_nonpadding.transpose(1,2))
            spk_embed = self.vqvae.encode_spk_embed(mel_prompt)
            if spk_embed_prompt != None:
                spk_embed = self.spk_embed_proj(spk_embed_prompt)[:,:,None]
            # Forward VQ LM
            ph_vqcode = (ph_vqcode.detach() + 1) * src_nonpadding.squeeze(-1).long()
            prev_ph_vqcode = F.pad(ph_vqcode[:, :-1], [1, 0], value=hparams['vqvae_ph_codebook_dim'] + 1)
            vq_codes_pred = self.vq_lm(txt_tokens, prev_ph_vqcode, spk_embed.transpose(1,2), ret=ret)

        else:
            # # Infer with pred VQCode
            in_nonpadding = (mel2ph_prompt > 0).float()[:, :, None]
            # Get GT VQCode for the first sentence
            ph_vqcode = self.vqvae.encode_ph_vqcode(mel_prompt, in_nonpadding.transpose(1,2), mel2ph_prompt, txt_tokens_prompt.shape[1], prompt_src_nonpadding.transpose(1,2))
            spk_embed = self.vqvae.encode_spk_embed(mel_prompt)
            if spk_embed_prompt != None:
                spk_embed = self.spk_embed_proj(spk_embed_prompt)[:,:,None]
            # Infer VQCode for the second sentence
            ph_vqcode = (ph_vqcode.detach() + 1)
            vq_codes_pred = self.vq_lm.infer(txt_tokens, ph_vqcode, spk_embed.transpose(1,2), txt_tokens_prompt.shape[1], ret)
            z_q_x_bar = self.vqvae.vqcode_to_latent((vq_codes_pred - 1).clamp_min(0))
            
            # Infer with GT VQCode
            # in_mel2ph = mel2ph
            # in_nonpadding = (in_mel2ph > 0).float()[:, :, None]
            # ph_vqcode = self.vqvae.encode_ph_vqcode(tgt_mels, in_nonpadding.transpose(1,2), in_mel2ph, txt_tokens.shape[1], src_nonpadding.transpose(1,2))
            # spk_embed = self.vqvae.encode_spk_embed(mel_prompt)
            # if spk_embed_prompt != None:
            #     spk_embed = self.spk_embed_proj(spk_embed_prompt)[:,:,None]
            # # ph_vqcode = (ph_vqcode.detach() + 1)
            # vq_codes_pred = self.vq_lm.infer(txt_tokens, ph_vqcode, spk_embed.transpose(1,2), txt_tokens_prompt.shape[1], ret)
            # # vq_codes_pred = (vq_codes_pred - 1).clamp_min(0)
            # z_q_x_bar = self.vqvae.vqcode_to_latent(ph_vqcode)
            
            # Infer mel with pred VQCode
            ph_encoder_out = self.encoder(txt_tokens)
            if use_gt_mel2ph:
                out_mel2ph = mel2ph
            else:
                if hparams.get("no_prosody_code_to_dur_predictor", False) is True:
                    dur_inp = (ph_encoder_out + spk_embed.transpose(1,2)) * src_nonpadding
                else:
                    dur_inp = (ph_encoder_out + z_q_x_bar.transpose(1,2)) * src_nonpadding
                out_mel2ph = self.forward_dur(dur_inp, None, txt_tokens, ret)
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


    def infer(self, txt_tokens, txt_tokens_gen, txt_tokens_prompt, mel2ph=None, mel2ph_prompt=None, infer=False, tgt_mels=None,
                mel_prompt=None, spk_embed_prompt=None, global_step=None, use_gt_mel2ph=True, *args, **kwargs):
        ret = {}
        src_nonpadding = (txt_tokens > 0).float()[:, :, None]
        prompt_src_nonpadding = (txt_tokens_prompt > 0).float()[:, :, None]

        # # Infer with pred VQCode
        in_nonpadding = (mel2ph_prompt > 0).float()[:, :, None]
        # Get GT VQCode for the first sentence
        ph_vqcode = self.vqvae.encode_ph_vqcode(mel_prompt, in_nonpadding.transpose(1,2), mel2ph_prompt, txt_tokens_prompt.shape[1], prompt_src_nonpadding.transpose(1,2))
        spk_embed = self.vqvae.encode_spk_embed(mel_prompt)
        if spk_embed_prompt != None:
            spk_embed = self.spk_embed_proj(spk_embed_prompt)[:,:,None]
        # Infer VQCode for the second sentence
        ph_vqcode = (ph_vqcode.detach() + 1)
        vq_codes_pred = self.vq_lm.infer(txt_tokens, ph_vqcode, spk_embed.transpose(1,2), txt_tokens_prompt.shape[1], ret)
        z_q_x_bar = self.vqvae.vqcode_to_latent((vq_codes_pred - 1).clamp_min(0))
        
        # Infer with GT VQCode
        # in_mel2ph = mel2ph
        # in_nonpadding = (in_mel2ph > 0).float()[:, :, None]
        # ph_vqcode = self.vqvae.encode_ph_vqcode(tgt_mels, in_nonpadding.transpose(1,2), in_mel2ph, txt_tokens.shape[1], src_nonpadding.transpose(1,2))
        # spk_embed = self.vqvae.encode_spk_embed(mel_prompt)
        # if spk_embed_prompt != None:
        #     spk_embed = self.spk_embed_proj(spk_embed_prompt)[:,:,None]
        # # ph_vqcode = (ph_vqcode.detach() + 1)
        # vq_codes_pred = self.vq_lm.infer(txt_tokens, ph_vqcode, spk_embed.transpose(1,2), txt_tokens_prompt.shape[1], ret)
        # # vq_codes_pred = (vq_codes_pred - 1).clamp_min(0)
        # z_q_x_bar = self.vqvae.vqcode_to_latent(ph_vqcode)
        
        # Infer mel with pred VQCode
        z_q_x_bar = z_q_x_bar[:, :, txt_tokens_prompt.shape[1]:]
        ph_encoder_out = self.encoder(txt_tokens_gen)
        if use_gt_mel2ph:
            out_mel2ph = mel2ph
        else:
            if hparams.get("no_prosody_code_to_dur_predictor", False) is True:
                dur_inp = (ph_encoder_out + spk_embed.transpose(1,2))
            else:
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