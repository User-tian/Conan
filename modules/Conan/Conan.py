# midi singer
import torch.nn as nn
from modules.tts.fs import FastSpeech
import math
from modules.tts.commons.align_ops import clip_mel2token_to_multiple, expand_states
from utils.audio.pitch.utils import denorm_f0, f0_to_coarse
import torch
import torch.nn.functional as F
from modules.Conan.diff.gaussian_multinomial_diffusion import (
    GaussianMultinomialDiffusion,
    GaussianMultinomialDiffusionx0,
)
from modules.Conan.diff.net import DiffNet, F0DiffNet, OriDiffNet, CausalConv1d

from utils.commons.hparams import hparams
from modules.Conan.diff.diff_f0 import GaussianDiffusionF0, GaussianDiffusionx0
from modules.Conan.flow.flow_f0 import ReflowF0
from modules.commons.nar_tts_modules import PitchPredictor
from modules.commons.layers import Embedding
from modules.Conan.flow.flow import FlowMel
from modules.commons.conv import ConvBlocks
from modules.commons.conv import TextConvEncoder, ConvBlocks, CausalConvBlocks, CausalFM
from modules.Conan.prosody_util import ProsodyAligner, LocalStyleAdaptor
from modules.commons.transformer import SinusoidalPositionalEmbedding

Flow_DECODERS = {
    "wavenet": lambda hp: DiffNet(hp["audio_num_mel_bins"]),
    "orig": lambda hp: OriDiffNet(hp["audio_num_mel_bins"]),
    "conv": lambda hp: CausalFM(
        hp["hidden_size"],
        hp["hidden_size"],
        hp["dec_dilations"],
        hp["dec_kernel_size"],
        layers_in_block=hp["layers_in_block"],
        norm_type=hp["enc_dec_norm"],
        dropout=hp["dropout"],
        post_net_kernel=hp.get("dec_post_net_kernel", 3),
    ),
}

DEFAULT_MAX_SOURCE_POSITIONS = 2000
DEFAULT_MAX_TARGET_POSITIONS = 2000


class Conan(FastSpeech):
    def __init__(self, dict_size, hparams, out_dims=None):
        super().__init__(dict_size, hparams, out_dims)

        hidden_size = hparams["hidden_size"]
        kernel_size = hparams["kernel_size"]
        self.content_embedding = nn.Embedding(102, hidden_size)
        # self.content_proj = nn.Sequential(
        #     nn.Conv1d(hidden_size, hidden_size, kernel_size=kernel_size, padding=kernel_size//2),
        #     # nn.Linear(80, hidden_size, bias=True)
        #     nn.LeakyReLU()
        # )
        self.content_proj = nn.Sequential(
            CausalConv1d(hidden_size, hidden_size, kernel_size=kernel_size, dilation=1),
            nn.LeakyReLU(),
        )
        self.global_conv_in = nn.Conv1d(80, hidden_size, 1)
        self.global_encoder = ConvBlocks(
            hidden_size,
            hidden_size,
            None,
            kernel_size=31,
            layers_in_block=2,
            is_BTC=False,
            num_layers=5,
        )

        if hparams["style"]:
            self.padding_idx = 0
            self.prosody_extractor = LocalStyleAdaptor(
                self.hidden_size, hparams["nVQ"], self.padding_idx
            )
            self.l1 = nn.Linear(self.hidden_size * 2, self.hidden_size)
            self.align = ProsodyAligner(num_layers=2)

            # build attention layer
            self.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
            self.embed_positions = SinusoidalPositionalEmbedding(
                self.hidden_size,
                self.padding_idx,
                init_size=self.max_source_positions + self.padding_idx + 1,
            )

        # self.time_ratio = hparams['sample_rate'] / hparams['hop_size'] / 50.0
        if hparams["f0_gen"] == "flow":
            self.uv_predictor = PitchPredictor(
                self.hidden_size,
                n_chans=128,
                n_layers=5,
                dropout_rate=0.1,
                odim=2,
                kernel_size=hparams["predictor_kernel"],
            )
            self.pitch_flownet = F0DiffNet(in_dims=1)
            self.f0_gen = ReflowF0(
                out_dims=1,
                denoise_fn=self.pitch_flownet,
                timesteps=hparams["f0_timesteps"],
                f0_K_step=hparams["f0_K_step"],
            )
        else:
            self.uv_predictor = PitchPredictor(
                self.hidden_size,
                n_chans=128,
                n_layers=5,
                dropout_rate=0.1,
                odim=2,
                kernel_size=hparams["predictor_kernel"],
            )

    def forward(
        self,
        content,
        spk_embed=None,
        target=None,
        ref=None,
        f0=None,
        uv=None,
        infer=False,
        global_steps=0,
        **kwargs,
    ):  # add **kwargs
        ret = {}  # initialize result dictionary
        # if hparams['hop_size']==320:
        ret["content"] = content  # store input content
        # elif hparams['hop_size']==160:
        #     ret['content'] = content.repeat_interleave(2, dim=-1)
        # assert False,f'content:{content.size()}'
        # --- assume tgt_nonpadding (target non-padding region) based on content's padding token (e.g. -1) ---
        # this is usually used for mask loss or output. shape might be [B, T_tok, 1]
        # print(f'content:{content.size()},f0:{f0.size()},target:{target.size()}')
        tgt_nonpadding = (content != -1).float()[:, :, None]
        # note: if decoder needs frame-level mask, this might need adjustment, or handle length-expanded mask inside decoder

        # compute content embedding
        content_embed = self.content_embedding(content)
        # process content embedding through CausalConv1d
        content_embed = self.content_proj(content_embed.transpose(1, 2)).transpose(1, 2)
        ret["content_embed_proj"] = content_embed  # store result

        # handle speaker embedding spk_embed
        if spk_embed is not None:
            # if spk_embed is directly provided
            # ret["style_embed"] = style_embed = self.forward_style_embed(spk_embed, None)
            ret["style_embed"] = style_embed = spk_embed
        else:
            # if spk_embed is not provided, extract from target mel spectrogram
            if ref is None:
                raise ValueError(
                    "When spk_embed is None, need target tensor to extract speaker embedding."
                )
            # encode_spk_embed may contain non-causal operations (like global encoder), but final output is static vector after avg_pool
            ret["style_embed"] = style_embed = self.encode_spk_embed(
                ref.transpose(1, 2)
            ).transpose(1, 2)

        # pitch input = content embedding + style embedding
        pitch_inp = content_embed + style_embed

        if hparams["style"]:
            # add prosody VQ
            prosody = self.get_prosody(pitch_inp, ref, ret, infer, global_steps)

            ret["pitch_embed"] = pitch_inp = pitch_inp + prosody
        else:
            ret["pitch_embed"] = pitch_inp

        # --- call forward_pitch to compute pitch embedding, and pass kwargs (including initial_noise) down ---
        # f0 and uv might be passed from outside (during training), or generated inside forward_pitch (during inference)
        if infer:
            f0, uv = None, None
        # ret['f0_denorm_pred'] will be filled inside forward_pitch
        pitch_embed_out = self.forward_pitch(
            pitch_inp, f0, uv, ret, **kwargs
        )  # pass kwargs
        # decoder input = pitch input + pitch embedding
        ret["decoder_inp"] = decoder_inp = pitch_inp + pitch_embed_out

        # --- call forward_decoder to generate Mel output ---
        # assume forward_decoder handles length regulation (from token to frame) and mask application internally
        # if forward_decoder needs frame-level mask, need to ensure it can correctly obtain or infer
        ret["mel_out"] = self.forward_decoder(
            decoder_inp, tgt_nonpadding, ret, infer=infer, **kwargs
        )  # also pass kwargs in case decoder needs them
        ret["tgt_nonpadding"] = tgt_nonpadding
        # === promote initial_noise_used stored by ReflowF0 in ret (if exists) to top-level ret ===
        # so test functions can directly get it from the dictionary returned by model()
        if "initial_noise_used" in ret:
            # note: this is internal dictionary assignment, actually ret is already the same dictionary,
            # just need to ensure 'initial_noise_used' key exists in final returned ret.
            # can add ret['initial_noise_used'] = ret['initial_noise_used'] to be explicit, but not necessary.
            pass  # just ensure key exists

        return ret  # return dictionary containing all results

    def encode_spk_embed(self, x):
        in_nonpadding = (x.abs().sum(dim=-2) > 0).float()[:, None, :]
        # forward encoder
        x_global = self.global_conv_in(x) * in_nonpadding
        global_z_e_x = (
            self.global_encoder(x_global, nonpadding=in_nonpadding) * in_nonpadding
        )
        # group by hidden to phoneme-level
        global_z_e_x = self.temporal_avg_pool(
            x=global_z_e_x, mask=(in_nonpadding == 0)
        )  # (B, C, T) -> (B, C, 1)
        spk_embed = global_z_e_x
        return spk_embed    

    def temporal_avg_pool(self, x, mask=None):
        len_ = (~mask).sum(dim=-1).unsqueeze(-1)
        x = x.masked_fill(mask, 0)
        x = x.sum(dim=-1).unsqueeze(-1)
        out = torch.div(x, len_)
        return out

    def get_prosody(self, encoder_out, ref_mels, ret, infer=False, global_steps=0):
        # get VQ prosody

        if "ref_upsample" not in ret or ret["ref_upsample"] is None:
            B, T, _ = ref_mels.shape  # get batch and frame length
            device = ref_mels.device
            base_ids = (
                torch.arange(T, device=device) // 4 + 1
            )  # [T] → 1,1,1,1,2,2,2,2,…
            ret["ref_upsample"] = base_ids.unsqueeze(0).expand(B, -1)  # [B, T]

        if global_steps > hparams["vq_start"] or infer:
            prosody_embedding, loss, ppl = self.prosody_extractor(
                ref_mels, ret["ref_upsample"], no_vq=False
            )
            ret["vq_loss"] = loss
            ret["ppl"] = ppl
        else:
            prosody_embedding = self.prosody_extractor(
                ref_mels, ret["ref_upsample"], no_vq=True
            )

        # add positional embedding
        positions = self.embed_positions(prosody_embedding[:, :, 0])
        prosody_embedding = self.l1(torch.cat([prosody_embedding, positions], dim=-1))

        # style-to-content attention
        src_key_padding_mask = encoder_out[:, :, 0].eq(self.padding_idx).data
        prosody_key_padding_mask = prosody_embedding[:, :, 0].eq(self.padding_idx).data
        # assert False,f'encoder_out:{encoder_out.size()},prosody_embedding:{prosody_embedding.size()},src_key_padding_mask:{src_key_padding_mask.size()},prosody_key_padding_mask:{prosody_key_padding_mask.size()}'
        if global_steps < hparams["forcing"]:
            output, guided_loss, attn_emo = self.align(
                encoder_out.transpose(0, 1),
                prosody_embedding.transpose(0, 1),
                src_key_padding_mask,
                prosody_key_padding_mask,
                forcing=True,
            )
        else:
            output, guided_loss, attn_emo = self.align(
                encoder_out.transpose(0, 1),
                prosody_embedding.transpose(0, 1),
                src_key_padding_mask,
                prosody_key_padding_mask,
                forcing=False,
            )

        ret["gloss"] = guided_loss
        ret["attn"] = attn_emo
        return output.transpose(0, 1)

    def forward_pitch(self, decoder_inp, f0, uv, ret, **kwargs):  # add **kwargs
        pitch_pred_inp = decoder_inp
        # apply predictor_grad to control gradient backprop
        if self.hparams["predictor_grad"] != 1:
            pitch_pred_inp = pitch_pred_inp.detach() + self.hparams[
                "predictor_grad"
            ] * (pitch_pred_inp - pitch_pred_inp.detach())

        # --- select F0 generation method based on config and pass kwargs ---
        if hparams["f0_gen"] == "diff":
            f0_out, uv_out = self.add_diff_pitch(pitch_pred_inp, f0, uv, ret, **kwargs)
        elif hparams["f0_gen"] == "gmdiff":
            f0_out, uv_out = self.add_gmdiff_pitch(
                pitch_pred_inp, f0, uv, ret, **kwargs
            )
        elif hparams["f0_gen"] == "flow":
            f0_out, uv_out = self.add_flow_pitch(
                pitch_pred_inp, f0, uv, ret, **kwargs
            )  # pass kwargs
        elif hparams["f0_gen"] == "orig":
            f0_out, uv_out = self.add_orig_pitch(pitch_pred_inp, f0, uv, ret, **kwargs)
        else:
            raise ValueError(f"Unknown f0_gen type: {hparams['f0_gen']}")

        # --- use f0_out, uv_out returned from add_x_pitch ---
        # f0_out might be log F0 or other forms, denorm_f0 should handle it
        f0_denorm = denorm_f0(f0_out, uv_out)  # use returned f0 and uv for denormalization
        pitch = f0_to_coarse(f0_denorm)  # convert to pitch categories
        ret["f0_denorm_pred"] = f0_denorm  # store final predicted denormalized F0 in ret
        pitch_embed = self.pitch_embed(pitch)  # compute pitch embedding
        return pitch_embed  # return pitch embedding

    # def forward_dur(self, dur_input, mel2ph, txt_tokens, ret):
    #     """

    #     :param dur_input: [B, T_txt, H]
    #     :param mel2ph: [B, T_mel]
    #     :param txt_tokens: [B, T_txt]
    #     :param ret:
    #     :return:
    #     """
    #     src_padding = txt_tokens == 0
    #     if self.hparams['predictor_grad'] != 1:
    #         dur_input = dur_input.detach() + self.hparams['predictor_grad'] * (dur_input - dur_input.detach())
    #     dur = self.dur_predictor(dur_input, src_padding)
    #     ret['dur'] = dur
    #     if mel2ph is None:
    #         dur = (dur.exp() - 1).clamp(min=0)
    #         mel2ph = self.length_regulator(dur, src_padding).detach()
    #     ret['mel2ph'] = mel2ph = clip_mel2token_to_multiple(mel2ph, self.hparams['frames_multiple'])
    #     return mel2ph

    def add_orig_pitch(self, decoder_inp, f0, uv, ret, encoder_out=None, **kwargs):
        # pitch_padding = mel2ph == 0
        if f0 is None:
            infer = True
        else:
            infer = False
        ret["uv_pred"] = uv_pred = self.uv_predictor(decoder_inp)

        if infer:
            uv = uv_pred[:, :, 0] > 0
            # print(f'uv:{uv},content:{ret["content"]}')
            if "content" in ret:
                content_padding_mask = ret["content"] == self.hparams['silent_token']  # assume -1 is padding index
                if content_padding_mask.shape == uv.shape:
                    uv[content_padding_mask] = 1  # force padding regions to be unvoiced
            uv = uv
            f0 = uv_pred[:, :, 1]
            ret["fdiff"] = 0.0
        else:
            # nonpadding = (mel2ph > 0).float() * (uv == 0).float()
            nonpadding = (uv == 0).float()
            f0_pred = uv_pred[:, :, 1]
            ret["fdiff"] = (
                (F.mse_loss(f0_pred, f0, reduction="none") * nonpadding).sum()
                / nonpadding.sum()
                * hparams["lambda_f0"]
            )
        return f0, uv

    def add_diff_pitch(
        self, decoder_inp, f0, uv, mel2ph, ret, encoder_out=None, **kwargs
    ):
        # pitch_padding = mel2ph == 0
        if f0 is None:
            infer = True
        else:
            infer = False
        ret["uv_pred"] = uv_pred = self.uv_predictor(decoder_inp)

        def minmax_norm(x, uv=None):
            x_min = 6
            x_max = 10
            if torch.any(x > x_max):
                raise ValueError("check minmax_norm!!")
            normed_x = (x - x_min) / (x_max - x_min) * 2 - 1
            if uv is not None:
                normed_x[uv > 0] = 0
            return normed_x

        def minmax_denorm(x, uv=None):
            x_min = 6
            x_max = 10
            denormed_x = (x + 1) / 2 * (x_max - x_min) + x_min
            if uv is not None:
                denormed_x[uv > 0] = 0
            return denormed_x

        if infer:
            uv = uv_pred[:, :, 0] > 0
            midi_notes = kwargs.get("midi_notes").transpose(-1, -2)
            uv[midi_notes[:, 0, :] == 0] = 1
            uv = uv
            lower_bound = midi_notes - 3
            upper_bound = midi_notes + 3
            upper_norm_f0 = minmax_norm((2 ** ((upper_bound - 69) / 12) * 440).log2())
            lower_norm_f0 = minmax_norm((2 ** ((lower_bound - 69) / 12) * 440).log2())
            upper_norm_f0[upper_norm_f0 < -1] = -1
            upper_norm_f0[upper_norm_f0 > 1] = 1
            lower_norm_f0[lower_norm_f0 < -1] = -1
            lower_norm_f0[lower_norm_f0 > 1] = 1
            f0 = self.f0_gen(
                decoder_inp.transpose(-1, -2),
                None,
                None,
                ret,
                infer,
                dyn_clip=[lower_norm_f0, upper_norm_f0],
            )  #
            # f0 = self.f0_gen(decoder_inp.transpose(-1, -2), None, None, ret, infer)
            f0 = f0[:, :, 0]
            f0 = minmax_denorm(f0)
            ret["fdiff"] = 0.0
        else:
            # nonpadding = (mel2ph > 0).float() * (uv == 0).float()
            nonpadding = (mel2ph > 0).float()
            norm_f0 = minmax_norm(f0)
            ret["fdiff"] = self.f0_gen(
                decoder_inp.transpose(-1, -2),
                norm_f0,
                nonpadding.unsqueeze(dim=1),
                ret,
                infer,
            )
        return f0, uv

    def add_flow_pitch(self, decoder_inp, f0, uv, ret, encoder_out=None, **kwargs):
        # pitch_padding = mel2ph == 0
        if f0 is None:
            infer = True
        else:
            infer = False
        ret["uv_pred"] = uv_pred = self.uv_predictor(decoder_inp)

        # define F0 normalization and denormalization functions (minmax_norm, minmax_denorm)
        def minmax_norm(x, uv=None):
            x_min = 6
            x_max = 10
            # if torch.any(x > x_max): # check if there are values exceeding the range (might only check during training)
            #     # print(f"Warning: F0 value > {x_max} found during normalization.")
            #     pass # or can choose to clip
            normed_x = (x - x_min) / (x_max - x_min) * 2 - 1
            if uv is not None:
                normed_x[uv > 0] = 0  # set unvoiced regions to 0
            return normed_x

        def minmax_denorm(x, uv=None):
            x_min = 6
            x_max = 10
            denormed_x = (x + 1) / 2 * (x_max - x_min) + x_min
            if uv is not None:
                denormed_x[uv > 0] = 0  # set unvoiced regions to 0
            return denormed_x

        # --- extract initial_noise from kwargs ---
        initial_noise = kwargs.get("initial_noise", None)

        # determine whether it's inference or training mode
        if f0 is None:  # if no f0 is provided, consider it inference mode
            infer = True
            if uv is None:  # if uv is also not provided, need to get it from predictor
                # ensure uv_predictor is initialized
                if not hasattr(self, "uv_predictor"):
                    raise AttributeError("uv_predictor is not defined in the model.")
                #  uv_pred = self.uv_predictor(decoder_inp) # predict UV, shape [B, T_tok, 2]
                uv = uv_pred[:, :, 0] > 0  # take first dimension as UV flag (True means unvoiced)
                # (optional) apply content padding mask to UV
                if "content" in ret:
                    content_padding_mask = (
                        ret["content"] == self.hparams['silent_token']
                    )  # assume -1 is padding index
                    if content_padding_mask.shape == uv.shape:
                        uv[content_padding_mask] = 1  # force padding regions to be unvoiced
                    else:
                        print(
                            f"Warning: content mask shape {content_padding_mask.shape} doesn't match uv shape {uv.shape}, cannot apply."
                        )
                else:
                    print(
                        "Warning: missing 'content' in ret, cannot apply content padding to UV."
                    )

            # --- call self.f0_gen (ReflowF0 instance) for F0 prediction, and pass initial_noise ---
            # input cond needs to be [B, C, T] shape
            # decoder_inp is [B, T, C], needs transpose
            # f0_gen output should be normalized F0, shape [B, T]
            f0_pred_norm = self.f0_gen(
                decoder_inp.transpose(1, 2),
                None,
                None,
                ret,
                infer=True,
                initial_noise=initial_noise,
            )
            # use predicted (or provided) uv for denormalization
            f0_out = minmax_denorm(f0_pred_norm, uv)
            ret["pflow"] = 0.0  # no flow loss during inference
            uv_out = uv  # return uv used for denormalization
        else:  # if f0 is provided, consider it training mode
            infer = False
            # compute nonpadding (voiced region mask)
            nonpadding = (uv == 0).float()
            # use provided f0, uv for normalization
            norm_f0 = minmax_norm(f0, uv)
            # call f0_gen to compute flow loss, usually don't pass initial_noise during training
            # f0_gen training input norm_f0 needs to be [B, 1, 1, T] or [B, 1, D, T]
            # add_flow_pitch receives f0 as [B, T], norm_f0 is also [B, T]
            # need to adjust shape before calling f0_gen
            if norm_f0.ndim == 2:
                norm_f0_unsqueezed = norm_f0.unsqueeze(1).unsqueeze(1)  # -> [B, 1, 1, T]
            else:  # if already [B, T, 1] or other shapes, need corresponding adjustment
                raise ValueError(f"Unexpected norm_f0 shape during training: {norm_f0.shape}")
            # nonpadding needs to be [B, 1, T]
            ret["pflow"] = self.f0_gen(
                decoder_inp.transpose(1, 2),
                norm_f0_unsqueezed,
                nonpadding.unsqueeze(1),
                ret,
                infer=False,
            )
            f0_out = f0  # return original f0 during training
            uv_out = uv  # return original uv during training

        return f0_out, uv_out  # return computed or original f0 and uv

    def add_gmdiff_pitch(
        self, decoder_inp, f0, uv, mel2ph, ret, encoder_out=None, **kwargs
    ):
        # pitch_padding = mel2ph == 0
        if f0 is None:
            infer = True
        else:
            infer = False

        def minmax_norm(x, uv=None):
            x_min = 6
            x_max = 10
            if torch.any(x > x_max):
                raise ValueError("check minmax_norm!!")
            normed_x = (x - x_min) / (x_max - x_min) * 2 - 1
            if uv is not None:
                normed_x[uv > 0] = 0
            return normed_x

        def minmax_denorm(x, uv=None):
            x_min = 6
            x_max = 10
            denormed_x = (x + 1) / 2 * (x_max - x_min) + x_min
            if uv is not None:
                denormed_x[uv > 0] = 0
            return denormed_x

        if infer:
            # uv = uv
            midi_notes = kwargs.get("midi_notes").transpose(-1, -2)
            lower_bound = midi_notes - 3  # 1 for good gtdur F0RMSE
            upper_bound = midi_notes + 3  # 1 for good gtdur F0RMSE
            upper_norm_f0 = minmax_norm((2 ** ((upper_bound - 69) / 12) * 440).log2())
            lower_norm_f0 = minmax_norm((2 ** ((lower_bound - 69) / 12) * 440).log2())
            upper_norm_f0[upper_norm_f0 < -1] = -1
            upper_norm_f0[upper_norm_f0 > 1] = 1
            lower_norm_f0[lower_norm_f0 < -1] = -1
            lower_norm_f0[lower_norm_f0 > 1] = 1
            pitch_pred = self.f0_gen(
                decoder_inp.transpose(-1, -2),
                None,
                None,
                None,
                ret,
                infer,
                dyn_clip=[lower_norm_f0, upper_norm_f0],
            )  # [lower_norm_f0, upper_norm_f0]
            f0 = pitch_pred[:, :, 0]
            uv = pitch_pred[:, :, 1]
            uv[midi_notes[:, 0, :] == 0] = 1
            f0 = minmax_denorm(f0)
            ret["gdiff"] = 0.0
            ret["mdiff"] = 0.0
        else:
            nonpadding = (mel2ph > 0).float()
            norm_f0 = minmax_norm(f0)
            ret["mdiff"], ret["gdiff"], ret["nll"] = self.f0_gen(
                decoder_inp.transpose(-1, -2),
                norm_f0.unsqueeze(dim=1),
                uv,
                nonpadding,
                ret,
                infer,
            )
        return f0, uv

    def forward_decoder(self, decoder_inp, tgt_nonpadding, ret, infer, **kwargs):
        x = decoder_inp  # [B, T, H]
        x = self.decoder(x)
        x = self.mel_out(x)
        # return x* tgt_nonpadding
        return x


class ConanPostnet(nn.Module):
    def __init__(self):
        super().__init__()
        cond_hs = 80 + hparams["hidden_size"]

        self.ln_proj = nn.Linear(cond_hs, hparams["hidden_size"])
        self.postflow = FlowMel(
            out_dims=80,
            denoise_fn=Flow_DECODERS[hparams["flow_decoder_type"]](hparams),
            timesteps=hparams["timesteps"],
            K_step=hparams["K_step"],
            loss_type=hparams["flow_loss_type"],
            spec_min=hparams["spec_min"],
            spec_max=hparams["spec_max"],
        )

    def forward(self, tgt_mels, infer, ret, cfg=False, cfg_scale=1.0, noise=None):
        g = self.get_condition(ret)
        x_recon = ret["mel_out"]
        ucond = None
        if cfg and infer:
            B = g.shape[0]
            B_f = B // 2
            if tgt_mels != None:
                tgt_mels = tgt_mels[:B_f]
            x_recon = x_recon[:B_f]
            ucond = g[B_f:]
            g = g[:B_f]
        self.postflow(g, tgt_mels, x_recon, ret, infer, ucond, noise, cfg_scale)

    def get_condition(self, ret):
        x_recon = ret["mel_out"]
        decoder_inp = ret["decoder_inp"]
        g = x_recon.detach()
        B, T, _ = g.shape
        g = torch.cat([g, decoder_inp], dim=-1)
        g = self.ln_proj(g)
        return g


# class TechSinger(RFSinger):
    # def __init__(self, dict_size, hparams, out_dims=None):
    #     super().__init__(dict_size, hparams, out_dims)

    #     cond_hs = 80 + hparams["hidden_size"]
    #     self.ln_proj = nn.Linear(cond_hs, hparams["hidden_size"])
    #     self.postflow = FlowMel(
    #         out_dims=80,
    #         denoise_fn=Flow_DECODERS[hparams["flow_decoder_type"]](hparams),
    #         timesteps=hparams["timesteps"],
    #         K_step=hparams["K_step"],
    #         loss_type=hparams["flow_loss_type"],
    #         spec_min=hparams["spec_min"],
    #         spec_max=hparams["spec_max"],
    #     )

    # def forward(
    #     self,
    #     txt_tokens,
    #     mel2ph=None,
    #     spk_id=None,
    #     f0=None,
    #     uv=None,
    #     note=None,
    #     note_dur=None,
    #     note_type=None,
    #     mix=None,
    #     falsetto=None,
    #     breathy=None,
    #     bubble=None,
    #     strong=None,
    #     weak=None,
    #     pharyngeal=None,
    #     vibrato=None,
    #     glissando=None,
    #     target=None,
    #     cfg=False,
    #     cfg_scale=1.0,
    #     infer=False,
    # ):
    #     ret = {}
    #     encoder_out = self.encoder(txt_tokens)  # [B, T, C]
    #     note_out = self.note_encoder(note, note_dur, note_type)
    #     encoder_out = encoder_out + note_out
    #     src_nonpadding = (txt_tokens > 0).float()[:, :, None]
    #     ret["spk_embed"] = style_embed = self.forward_style_embed(None, spk_id)
    #     tech = self.tech_encoder(
    #         mix, falsetto, breathy, bubble, strong, weak, pharyngeal, vibrato, glissando
    #     )
    #     # add dur
    #     dur_inp = (encoder_out + style_embed) * src_nonpadding
    #     mel2ph = self.forward_dur(dur_inp, mel2ph, txt_tokens, ret)
    #     tgt_nonpadding = (mel2ph > 0).float()[:, :, None]
    #     decoder_inp = expand_states(encoder_out, mel2ph)
    #     in_nonpadding = (mel2ph > 0).float()[:, :, None]

    #     ret["tech"] = tech = expand_states(tech, mel2ph) * tgt_nonpadding
    #     ret["mel2ph"] = mel2ph

    #     # add pitch embed
    #     midi_notes = None
    #     pitch_inp = (decoder_inp + style_embed + tech) * tgt_nonpadding
    #     if infer:
    #         f0, uv = None, None
    #         midi_notes = expand_states(note[:, :, None], mel2ph)
    #     decoder_inp = decoder_inp + self.forward_pitch(
    #         pitch_inp, f0, uv, mel2ph, ret, encoder_out, midi_notes=midi_notes
    #     )
    #     # decoder input
    #     ret["decoder_inp"] = decoder_inp = (
    #         decoder_inp + style_embed + tech
    #     ) * tgt_nonpadding
    #     ret["coarse_mel_out"] = self.forward_decoder(
    #         decoder_inp, tgt_nonpadding, ret, infer=infer
    #     )
    #     ret["tgt_nonpadding"] = tgt_nonpadding

    #     self.forward_post(target, infer, ret, cfg=cfg, cfg_scale=cfg_scale)
    #     return ret

    # def forward_post(self, tgt_mels, infer, ret, cfg=False, cfg_scale=1.0, noise=None):
    #     g = self.get_condition(ret)
    #     x_recon = ret["coarse_mel_out"]
    #     ucond = None
    #     if cfg and infer:
    #         B = g.shape[0]
    #         B_f = B // 2
    #         if tgt_mels != None:
    #             tgt_mels = tgt_mels[:B_f]
    #         x_recon = x_recon[:B_f]
    #         ucond = g[B_f:]
    #         g = g[:B_f]
    #     self.postflow(g, tgt_mels, x_recon, ret, infer, ucond, noise, cfg_scale)

    # def get_condition(self, ret):
    #     x_recon = ret["coarse_mel_out"]
    #     decoder_inp = ret["decoder_inp"]
    #     g = x_recon.detach()
    #     B, T, _ = g.shape
    #     g = torch.cat([g, decoder_inp], dim=-1)
    #     g = self.ln_proj(g)
    #     return g
