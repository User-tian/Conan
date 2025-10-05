import os
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import utils
from modules.fastspeech.multi_window_disc import Discriminator
from modules.vocoder.hifigan.hifigan_causal import (
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
    HifiGanGenerator,
    feature_loss,
    generator_loss,
    discriminator_loss,
    cond_discriminator_loss,
    mel_loss,
)
from modules.vocoder.hifigan.mel_utils import mel_spectrogram
from modules.parallel_wavegan.losses import MultiResolutionSTFTLoss
from tasks.vocoder.dataset_utils import HifiGANDataset
from tasks.vocoder.vocoder_base import VocoderBaseTask
from utils import audio
from utils.commons.hparams import hparams
from utils.nn.model_utils import print_arch
from utils.commons.tensor_utils import tensors_to_scalars
import math


def parselmouth_pitch(
    wav_data,
    hop_size,
    audio_sample_rate,
    f0_min,
    f0_max,
    voicing_threshold=0.6,
    *args,
    **kwargs,
):
    import parselmouth

    time_step = hop_size / audio_sample_rate * 1000
    n_mel_frames = int(len(wav_data) // hop_size)
    f0_pm = (
        parselmouth.Sound(wav_data, audio_sample_rate)
        .to_pitch_ac(
            time_step=time_step / 1000,
            voicing_threshold=voicing_threshold,
            pitch_floor=f0_min,
            pitch_ceiling=f0_max,
        )
        .selected_array["frequency"]
    )
    pad_size = (n_mel_frames - len(f0_pm) + 1) // 2
    f0 = np.pad(
        f0_pm, [[pad_size, n_mel_frames - len(f0_pm) - pad_size]], mode="constant"
    )
    return f0


class HifiGanTask(VocoderBaseTask):
    def build_model(self):
        self.model_gen = HifiGanGenerator(hparams)
        self.model_disc = nn.ModuleDict()
        self.model_disc["mpd"] = MultiPeriodDiscriminator(
            use_cond=hparams["use_cond_disc"]
        )
        self.model_disc["msd"] = MultiScaleDiscriminator(
            use_cond=hparams["use_cond_disc"]
        )
        self.stft_loss = MultiResolutionSTFTLoss(
            fft_sizes=hparams["stft_loss_param"]["fft_sizes"],
            hop_sizes=hparams["stft_loss_param"]["hop_sizes"],
            win_lengths=hparams["stft_loss_param"]["win_lengths"],
        )
        if hparams["use_spec_disc"]:
            self.model_disc["specd"] = Discriminator(
                time_lengths=[8, 16, 32],
                freq_length=80,
                hidden_size=128,
                kernel=(3, 3),
                cond_size=0,
                reduction="stack",
            )
        print_arch(self.model_gen)
        if hparams["load_ckpt"] != "":
            self.load_ckpt(
                hparams["load_ckpt"], "model_gen", "model_gen", force=True, strict=True
            )
            self.load_ckpt(
                hparams["load_ckpt"],
                "model_disc",
                "model_disc",
                force=True,
                strict=True,
            )
        return self.model_gen

    def build_optimizer(self, model):
        optimizer_gen = torch.optim.AdamW(
            self.model_gen.parameters(),
            betas=[hparams["adam_b1"], hparams["adam_b2"]],
            **hparams["generator_optimizer_params"],
        )
        optimizer_disc = torch.optim.AdamW(
            self.model_disc.parameters(),
            betas=[hparams["adam_b1"], hparams["adam_b2"]],
            **hparams["discriminator_optimizer_params"],
        )
        return [optimizer_gen, optimizer_disc]

    def build_scheduler(self, optimizer):
        return {
            "gen": torch.optim.lr_scheduler.StepLR(
                optimizer=optimizer[0], **hparams["generator_scheduler_params"]
            ),
            "disc": torch.optim.lr_scheduler.StepLR(
                optimizer=optimizer[1], **hparams["discriminator_scheduler_params"]
            ),
        }

    def _training_step(self, sample, batch_idx, optimizer_idx):
        mel = sample["mels"]
        y = sample["wavs"]
        f0 = sample["f0"] if hparams.get("use_pitch_embed", False) else None

        # print(f'f0_shape: {f0.shape}, mel_shape: {mel.shape}, wav_shape: {y.shape}')

        loss_output = {}
        if optimizer_idx == 0:
            #######################
            #      Generator      #
            #######################
            y_ = self.model_gen(mel, f0)
            # import ipdb
            # ipdb.set_trace()
            y_mel = mel_spectrogram(
                y.squeeze(1), hparams, for_loss=hparams["use_different_mel_loss"]
            ).transpose(1, 2)
            y_hat_mel = mel_spectrogram(
                y_.squeeze(1), hparams, for_loss=hparams["use_different_mel_loss"]
            ).transpose(1, 2)
            #
            # loss_output['mel'] = F.l1_loss(y_hat_mel, y_mel) * hparams['lambda_mel']
            loss_output["mel"] = mel_loss(y, y_, hparams)
            _, y_p_hat_g, fmap_f_r, fmap_f_g = self.model_disc["mpd"](y, y_, mel)
            _, y_s_hat_g, fmap_s_r, fmap_s_g = self.model_disc["msd"](y, y_, mel)
            loss_output["a_p"] = generator_loss(y_p_hat_g) * hparams["lambda_adv"]
            loss_output["a_s"] = generator_loss(y_s_hat_g) * hparams["lambda_adv"]
            if hparams["use_fm_loss"]:
                loss_output["fm_f"] = feature_loss(fmap_f_r, fmap_f_g)
                loss_output["fm_s"] = feature_loss(fmap_s_r, fmap_s_g)
            if hparams["use_spec_disc"]:
                p_ = self.model_disc["specd"](y_hat_mel)["y"]
                loss_output["a_mel"] = (
                    self.mse_loss_fn(p_, p_.new_ones(p_.size()))
                    * hparams["lambda_mel_adv"]
                )
            if hparams["use_ms_stft"]:
                loss_output["sc"], loss_output["mag"] = self.stft_loss(
                    y.squeeze(1), y_.squeeze(1)
                )
            self.y_ = y_.detach()
            self.y_mel = y_mel.detach()
            self.y_hat_mel = y_hat_mel.detach()
        else:
            #######################
            #    Discriminator    #
            #######################
            y_ = self.y_
            # MPD
            y_p_hat_r, y_p_hat_g, _, _ = self.model_disc["mpd"](y, y_.detach(), mel)
            loss_output["r_p"], loss_output["f_p"] = discriminator_loss(
                y_p_hat_r, y_p_hat_g
            )
            # MSD
            y_s_hat_r, y_s_hat_g, _, _ = self.model_disc["msd"](y, y_.detach(), mel)
            loss_output["r_s"], loss_output["f_s"] = discriminator_loss(
                y_s_hat_r, y_s_hat_g
            )
            # spec disc
            if hparams["use_spec_disc"]:
                p = self.model_disc["specd"](self.y_mel)["y"]
                p_ = self.model_disc["specd"](self.y_hat_mel)["y"]
                loss_output["r_mel"] = self.mse_loss_fn(p, p.new_ones(p.size()))
                loss_output["f_mel"] = self.mse_loss_fn(p_, p_.new_zeros(p_.size()))
            if hparams["use_cond_disc"]:
                mel_shift = torch.roll(mel, -1, 0)
                yp_f1, yp_f2, _, _ = self.model_disc["mpd"](
                    y.detach(), y_.detach(), mel_shift
                )
                loss_output["f_p_cd1"] = cond_discriminator_loss(yp_f1)
                loss_output["f_p_cd2"] = cond_discriminator_loss(yp_f2)
                ys_f1, ys_f2, _, _ = self.model_disc["msd"](
                    y.detach(), y_.detach(), mel_shift
                )
                loss_output["f_s_cd1"] = cond_discriminator_loss(ys_f1)
                loss_output["f_s_cd2"] = cond_discriminator_loss(ys_f2)
        total_loss = sum(loss_output.values())
        return total_loss, loss_output

    def on_before_optimization(self, opt_idx):
        if opt_idx == 0:
            nn.utils.clip_grad_norm_(
                self.model_gen.parameters(), hparams["generator_grad_norm"]
            )
        else:
            nn.utils.clip_grad_norm_(
                self.model_disc.parameters(), hparams["discriminator_grad_norm"]
            )

    def on_after_optimization(self, epoch, batch_idx, optimizer, optimizer_idx):
        if optimizer_idx == 0:
            self.scheduler["gen"].step(
                self.global_step // hparams["accumulate_grad_batches"]
            )
        else:
            self.scheduler["disc"].step(
                self.global_step // hparams["accumulate_grad_batches"]
            )

    def validation_step(self, sample, batch_idx):
        outputs = {}
        total_loss, loss_output = self._training_step(sample, batch_idx, 0)
        outputs["losses"] = tensors_to_scalars(loss_output)
        outputs["total_loss"] = tensors_to_scalars(total_loss)

        if self.global_step % 50000 == 0 and batch_idx < 10:
            mels = sample["mels"]
            y = sample["wavs"]
            f0 = sample["f0"] if hparams.get("use_pitch_embed", False) else None
            y_ = self.model_gen(mels, f0)
            for idx, (wav_pred, wav_gt, item_name) in enumerate(
                zip(y_, y, sample["item_name"])
            ):
                wav_pred = wav_pred / wav_pred.abs().max()
                # assert False, f'wav_gtshape: {wav_gt.shape}, wav_predshape: {wav_pred.shape}'
                if self.global_step == 1000000:
                    wav_gt = wav_gt / wav_gt.abs().max()
                    self.logger.add_audio(
                        f"wav_{batch_idx}_{idx}_gt",
                        wav_gt.transpose(0, 1),
                        self.global_step,
                        hparams["audio_sample_rate"],
                    )
                self.logger.add_audio(
                    f"wav_{batch_idx}_{idx}_pred",
                    wav_pred.transpose(0, 1),
                    self.global_step,
                    hparams["audio_sample_rate"],
                )
        return outputs

    # def test_step(self, sample, batch_idx):
    #     mels = sample['mels']
    #     y = sample['wavs']
    #     f0 = sample['f0'] if hparams.get('use_pitch_embed', False) else None
    #     loss_output = {}
    #     y_ = self.model_gen(mels, f0)
    #     gen_dir = os.path.join(hparams['work_dir'], f'generated_{self.trainer.global_step}_{hparams["gen_dir_name"]}')
    #     os.makedirs(gen_dir, exist_ok=True)
    #     if hparams['save_f0']:
    #         f0_dir = f"{gen_dir}/f0"
    #         os.makedirs(f0_dir, exist_ok=True)

    #     #for idx, (wav_pred, wav_gt, item_name) in enumerate(zip(y_, y, sample["item_name"])):
    #     for idx, (wav_pred, wav_gt, item_name, mel) in enumerate(zip(y_, y, sample["item_name"], sample['mels'])):
    #         wav_gt = wav_gt.clamp(-1, 1)
    #         wav_pred = wav_pred.clamp(-1, 1)
    #         audio.save_wav(
    #             wav_gt.view(-1).cpu().float().numpy(), f'{gen_dir}/{item_name}_gt.wav',
    #             hparams['audio_sample_rate'])
    #         audio.save_wav(
    #             wav_pred.view(-1).cpu().float().numpy(), f'{gen_dir}/{item_name}_pred.wav',
    #             hparams['audio_sample_rate'])
    #     return loss_output

    @torch.no_grad()  # Test steps don't need gradients
    def test_step(self, sample, batch_idx):
        mels = sample["mels"]  # (B, mel_bins, T_mel)
        wavs_gt = sample["wavs"]  # (B, 1, T_wav)
        f0s = (
            sample.get("f0") if hparams.get("use_pitch_embed", False) else None
        )  # (B, T_mel) or None
        item_names = sample["item_name"]

        # Get necessary hparams
        hop_size = hparams["hop_size"]
        sample_rate = hparams["audio_sample_rate"]
        segment_duration_ms = 80  # Expected segment duration in milliseconds

        # Calculate mel frames and audio samples corresponding to each 80ms segment
        segment_samples = int(segment_duration_ms / 1000 * sample_rate)
        segment_mel_frames = math.ceil(
            segment_samples / hop_size
        )  # Round up to cover 80ms
        segment_samples_aligned = (
            segment_mel_frames * hop_size
        )  # Precise number of samples corresponding to these mel frames

        # Create output directory
        gen_dir = os.path.join(
            hparams["work_dir"],
            f'generated_{self.trainer.global_step}_{hparams["gen_dir_name"]}_incremental_check',
        )
        os.makedirs(gen_dir, exist_ok=True)
        if hparams.get("save_f0", False):
            f0_dir = f"{gen_dir}/f0"
            os.makedirs(f0_dir, exist_ok=True)

        loss_output = {}  # test_step usually doesn't compute loss

        # Iterate through each sample in the batch
        for idx in range(mels.size(0)):
            mel_single = mels[idx : idx + 1]  # (1, mel_bins, T_mel)
            wav_gt_single = wavs_gt[idx : idx + 1]  # (1, 1, T_wav)
            f0_single = (
                f0s[idx : idx + 1] if f0s is not None else None
            )  # (1, T_mel) or None
            item_name = item_names[idx]
            total_mel_frames = mel_single.shape[-1]

            generated_wav_segments = []
            prev_generated_wav_chunk_full = None  # Used to store complete audio generated from previous step

            # Iterate with segment_mel_frames as step size
            # Use enumerate to get iteration count i (starting from 0)
            for i, start_mel_frame in enumerate(
                range(0, total_mel_frames, segment_mel_frames)
            ):
                # Determine mel frame range required for current inference (from 0 to current segment end)
                end_mel_frame = min(
                    start_mel_frame + segment_mel_frames, total_mel_frames
                )
                if end_mel_frame <= start_mel_frame:  # If no new frames, skip
                    continue
                current_mel_input = mel_single[
                    :, :, :end_mel_frame
                ]  # Input is mel from the beginning

                # Prepare F0 input (if used)
                current_f0_input = None
                if f0_single is not None:
                    current_f0_input = f0_single[:, :end_mel_frame]

                # --- Generate audio ---
                # Use model to generate audio corresponding to current_mel_input
                generated_wav_chunk_full = self.model_gen(
                    current_mel_input, current_f0_input
                )  # (1, 1, end_mel_frame * hop_size)

                # --- Consistency check (starting from second iteration) ---
                if prev_generated_wav_chunk_full is not None:
                    len_to_compare = prev_generated_wav_chunk_full.shape[-1]
                    # Ensure current generated audio is long enough for comparison
                    if generated_wav_chunk_full.shape[-1] >= len_to_compare:
                        current_initial_part = generated_wav_chunk_full[
                            :, :, :len_to_compare
                        ]
                        # Use torch.allclose for comparison, allowing small floating point errors
                        # atol (absolute tolerance), rtol (relative tolerance) may need adjustment
                        are_close = torch.allclose(
                            current_initial_part,
                            prev_generated_wav_chunk_full,
                            atol=1e-5,
                            rtol=1e-5,
                        )

                        if not are_close:
                            # If inconsistent, print warning information and difference norm
                            diff_norm = torch.norm(
                                current_initial_part - prev_generated_wav_chunk_full
                            )
                            print(
                                f"WARNING: Consistency Check Failed for item {item_name} at step {i+1}!"
                            )
                            print(
                                f"         Input mel frames: 0-{end_mel_frame} vs 0-{start_mel_frame}"
                            )
                            print(
                                f"         Comparing audio samples: 0-{len_to_compare}"
                            )
                            print(f"         Difference norm: {diff_norm.item():.6f}")
                        # else:
                        # (Optional) If needed, you can print matching information
                        # print(f"INFO: Consistency Check Passed for item {item_name} at step {i+1}.")

                    else:
                        # If current generated audio is shorter than previous (theoretically shouldn't happen unless model or input has issues)
                        print(
                            f"WARNING: Current generated audio ({generated_wav_chunk_full.shape[-1]} samples) "
                            f"is shorter than previous ({len_to_compare} samples) for item {item_name} at step {i+1}. Cannot compare."
                        )

                # --- Store current complete generation result for next comparison ---
                # Use .clone() to avoid subsequent operations affecting stored values
                prev_generated_wav_chunk_full = generated_wav_chunk_full.clone()

                # --- Extract audio segment corresponding to current step's *new* part ---
                # Calculate start and end indices of audio samples to extract
                start_audio_sample = start_mel_frame * hop_size
                # end_audio_sample should be end_mel_frame * hop_size
                # but ensure it doesn't exceed actual generated audio length (total length of generated_wav_chunk_full)
                end_audio_sample = min(
                    end_mel_frame * hop_size, generated_wav_chunk_full.shape[-1]
                )

                # Extract the required new segment from generated complete audio chunk (corresponding to mel[start_mel_frame:end_mel_frame])
                # Note: indices are relative to generated_wav_chunk_full
                required_segment = generated_wav_chunk_full[
                    :, :, start_audio_sample:end_audio_sample
                ]
                generated_wav_segments.append(required_segment)

            # --- Concatenate all extracted audio segments ---
            if generated_wav_segments:
                wav_pred_incremental = torch.cat(
                    generated_wav_segments, dim=-1
                )  # Concatenate last dimension (time)
            else:
                wav_pred_incremental = torch.zeros_like(
                    wav_gt_single[:, :, :0]
                )  # Handle empty mel case

            # --- Save audio ---
            wav_gt_save = wav_gt_single.clamp(-1, 1).squeeze().cpu().float().numpy()
            wav_pred_save = (
                wav_pred_incremental.clamp(-1, 1).squeeze().cpu().float().numpy()
            )

            audio.save_wav(
                wav_gt_save, os.path.join(gen_dir, f"{item_name}_gt.wav"), sample_rate
            )
            audio.save_wav(
                wav_pred_save,
                os.path.join(gen_dir, f"{item_name}_pred.wav"),
                sample_rate,
            )

            # --- Save F0 (if needed) ---
            if hparams.get("save_f0", False) and f0_single is not None:
                f0_to_save = f0_single.squeeze().cpu().numpy()
                np.save(os.path.join(f0_dir, f"{item_name}_f0.npy"), f0_to_save)

        return loss_output  # Return empty loss_output


import os
import time
import yaml
import torch
import numpy as np

from modules.vocoder.hifigan.hifigan_causal import HifiGanGenerator
from utils.commons.hparams import hparams
import utils.audio as audio  # 用于保存 WAV 文件


def deep_merge_dicts(base: dict, override: dict) -> dict:
    """
    Recursively merge override into base:
    - If a key in override is also a dict, recursively merge;
    - Otherwise directly override base with override.
    Returns a new dictionary without modifying input parameters.
    """
    merged = {}
    keys = set(base.keys()) | set(override.keys())
    for k in keys:
        if k in base and k in override:
            if isinstance(base[k], dict) and isinstance(override[k], dict):
                merged[k] = deep_merge_dicts(base[k], override[k])
            else:
                merged[k] = override[k]
        elif k in override:
            merged[k] = override[k]
        else:
            merged[k] = base[k]
    return merged


def load_recursive_yaml(yaml_path: str, project_root: str, visited=None) -> dict:
    """
    Recursively load a YAML file and all its base_config:
    - yaml_path: file path to load (both absolute and relative paths work).
    - project_root: project root directory, used to resolve base_config paths starting with "egs/".
    - visited: used to record already loaded paths, avoiding circular dependencies.
    Returns merged dict (load base first, then current file, with current file as override).
    """
    if visited is None:
        visited = set()

    yaml_path = os.path.abspath(yaml_path)
    if yaml_path in visited:
        return {}
    visited.add(yaml_path)

    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(f"Cannot find YAML file: {yaml_path}")

    # —— 1. Read current layer YAML content —— #
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # —— 2. If base_config exists, recursively load them first —— #
    base_merged = {}
    if "base_config" in cfg:
        base_entry = cfg["base_config"]
        if isinstance(base_entry, str):
            base_list = [base_entry]
        elif isinstance(base_entry, list):
            base_list = base_entry
        else:
            raise ValueError(
                f"base_config must be string or list of strings, but got {type(base_entry)} in {yaml_path}"
            )

        for base_rel in base_list:
            # If base_rel is absolute path, use directly;
            # If starts with "egs/", treat as path relative to project_root;
            # Otherwise treat as relative path to current YAML file's directory.
            if os.path.isabs(base_rel):
                base_path = base_rel
            elif base_rel.startswith("egs/"):
                # Concatenate based on project root
                base_path = os.path.join(project_root, base_rel)
            else:
                base_dir = os.path.dirname(yaml_path)
                base_path = os.path.join(base_dir, base_rel)

            base_path = os.path.abspath(base_path)
            base_cfg_dict = load_recursive_yaml(base_path, project_root, visited)
            base_merged = deep_merge_dicts(base_merged, base_cfg_dict)

    # —— 3. Merge: base_merged + current layer cfg, prioritizing current layer cfg —— #
    merged = deep_merge_dicts(base_merged, cfg)
    return merged


def load_hparams_with_base(main_yaml_path: str):
    """
    Load all hyperparameters from specified main YAML (and all its base_config) into utils.hparams.hparams.
    """
    # First calculate project_root: assume main YAML is in "…/project_root/egs_usr/…"
    # Then project_root is the parent directory of the main YAML's directory
    main_yaml_abspath = os.path.abspath(main_yaml_path)
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(main_yaml_abspath), "..")
    )

    merged_cfg = load_recursive_yaml(main_yaml_abspath, project_root)
    # Clear existing hparams and write new values
    hparams.clear()
    hparams.update(merged_cfg)
    return hparams


if __name__ == "__main__":
    # ----------------------------------------------------------------------
    # 1. Main YAML path (please modify according to actual situation)
    main_yaml = "/home2/zhangyu/hifigan_casual/egs_usr/hifinsf_16k320_shuffle.yaml"
    if not os.path.isfile(main_yaml):
        raise FileNotFoundError(f"Cannot find main YAML file: {main_yaml}")

    # 2. Load and merge all base_config
    load_hparams_with_base(main_yaml)
    print("------ Complete hparams configuration loaded ------")

    # Force mel_bins / n_mel_channels to be 80
    hparams["n_mel_channels"] = 80
    hparams["mel_bins"] = 80

    print(f"Sample rate audio_sample_rate = {hparams.get('audio_sample_rate')}")
    print(f"Mel channels n_mel_channels = {hparams.get('n_mel_channels')}")

    # ----------------------------------------------------------------------
    # 3. Select device (prioritize CUDA, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ----------------------------------------------------------------------
    # 4. Build and load HiFi-GAN Generator
    generator = HifiGanGenerator(hparams).to(device)

    # If you have pretrained checkpoint, you can uncomment the code below and modify the path:
    # ckpt_path = "/home2/zhangyu/hifigan_casual/checkpoints/generator.pth"
    # if os.path.isfile(ckpt_path):
    #     checkpoint = torch.load(ckpt_path, map_location=device)
    #     generator.load_state_dict(checkpoint["generator"], strict=True)
    #     print("Loaded pretrained weights:", ckpt_path)
    # else:
    #     print("Specified Generator checkpoint not found, will use randomly initialized model.")

    generator.eval()

    # ----------------------------------------------------------------------
    # 5. Generate a random mel spectrogram frame for inference testing
    n_mel_channels = hparams["n_mel_channels"]
    mel_frame = torch.randn(1, n_mel_channels, 4, device=device)

    # if hparams.get("use_pitch_embed", False):
    #     # Randomly generate an f0 value with shape (1, 1)
    #     f0_frame = torch.randn(1, 1, device=device) * 5 + 100
    # else:
    f0_frame = None

    # Warm up inference once
    with torch.no_grad():
        _ = generator(mel_frame, f0_frame)

    # ----------------------------------------------------------------------
    # 6. Multiple inferences and measure average latency (unit: milliseconds)
    num_runs = 50
    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            wav_out = generator(mel_frame, f0_frame)
    if device.type == "cuda":
        torch.cuda.synchronize()
    end_time = time.time()

    avg_latency_ms = (end_time - start_time) / num_runs * 1000
    print(f"Single frame Mel inference average latency: {avg_latency_ms:.2f} ms")

    # ----------------------------------------------------------------------
    # 7. (Optional) Save the last generated wav_out as WAV file for easy listening
    wav_np = wav_out.squeeze().cpu().numpy()
    save_path = os.path.join(os.getcwd(), "test_output.wav")
    audio.save_wav(wav_np, save_path, hparams["audio_sample_rate"])
    print(f"Generated audio saved to: {save_path}")

    print("Inference test completed.")
