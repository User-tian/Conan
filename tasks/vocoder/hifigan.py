import os
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import utils
from modules.fastspeech.multi_window_disc import Discriminator
from modules.hifigan.hifigan import (
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
    HifiGanGenerator,
    feature_loss,
    generator_loss,
    discriminator_loss,
    cond_discriminator_loss,
    mel_loss,
)
from modules.hifigan.mel_utils import mel_spectrogram
from modules.parallel_wavegan.losses import MultiResolutionSTFTLoss
from tasks.vocoder.dataset_utils import VocoderSingDataset
from tasks.vocoder.vocoder_base import VocoderBaseTask
from utils import audio
from utils.hparams import hparams
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
        utils.print_arch(self.model_gen)
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
        outputs["losses"] = utils.tensors_to_scalars(loss_output)
        outputs["total_loss"] = utils.tensors_to_scalars(total_loss)

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

    @torch.no_grad()  # 测试步骤不需要梯度
    def test_step(self, sample, batch_idx):
        mels = sample["mels"]  # (B, mel_bins, T_mel)
        wavs_gt = sample["wavs"]  # (B, 1, T_wav)
        f0s = (
            sample.get("f0") if hparams.get("use_pitch_embed", False) else None
        )  # (B, T_mel) or None
        item_names = sample["item_name"]

        # 获取必要的 hparams
        hop_size = hparams["hop_size"]
        sample_rate = hparams["audio_sample_rate"]
        segment_duration_ms = 80  # 期望的段持续时间（毫秒）

        # 计算每个 80ms 段对应的 Mel 帧数和音频样本数
        segment_samples = int(segment_duration_ms / 1000 * sample_rate)
        segment_mel_frames = math.ceil(
            segment_samples / hop_size
        )  # 向上取整以覆盖 80ms
        segment_samples_aligned = (
            segment_mel_frames * hop_size
        )  # 对应这些 Mel 帧的精确样本数

        # 创建输出目录
        gen_dir = os.path.join(
            hparams["work_dir"],
            f'generated_{self.trainer.global_step}_{hparams["gen_dir_name"]}_incremental_check',
        )
        os.makedirs(gen_dir, exist_ok=True)
        if hparams.get("save_f0", False):
            f0_dir = f"{gen_dir}/f0"
            os.makedirs(f0_dir, exist_ok=True)

        loss_output = {}  # test_step 通常不计算损失

        # 遍历批次中的每个样本
        for idx in range(mels.size(0)):
            mel_single = mels[idx : idx + 1]  # (1, mel_bins, T_mel)
            wav_gt_single = wavs_gt[idx : idx + 1]  # (1, 1, T_wav)
            f0_single = (
                f0s[idx : idx + 1] if f0s is not None else None
            )  # (1, T_mel) or None
            item_name = item_names[idx]
            total_mel_frames = mel_single.shape[-1]

            generated_wav_segments = []
            prev_generated_wav_chunk_full = None  # 用于存储上一步生成的完整音频

            # 以 segment_mel_frames 为步长进行迭代
            # 使用 enumerate 获取迭代次数 i (从 0 开始)
            for i, start_mel_frame in enumerate(
                range(0, total_mel_frames, segment_mel_frames)
            ):
                # 确定当前推理所需的 Mel 帧范围 (从 0 到当前段结束)
                end_mel_frame = min(
                    start_mel_frame + segment_mel_frames, total_mel_frames
                )
                if end_mel_frame <= start_mel_frame:  # 如果没有新的帧了，跳过
                    continue
                current_mel_input = mel_single[
                    :, :, :end_mel_frame
                ]  # 输入是从头开始的 Mel

                # 准备 F0 输入 (如果使用)
                current_f0_input = None
                if f0_single is not None:
                    current_f0_input = f0_single[:, :end_mel_frame]

                # --- 生成音频 ---
                # 使用模型生成对应于 current_mel_input 的音频
                generated_wav_chunk_full = self.model_gen(
                    current_mel_input, current_f0_input
                )  # (1, 1, end_mel_frame * hop_size)

                # --- 一致性检查 (从第二次迭代开始) ---
                if prev_generated_wav_chunk_full is not None:
                    len_to_compare = prev_generated_wav_chunk_full.shape[-1]
                    # 确保当前生成的音频足够长以进行比较
                    if generated_wav_chunk_full.shape[-1] >= len_to_compare:
                        current_initial_part = generated_wav_chunk_full[
                            :, :, :len_to_compare
                        ]
                        # 使用 torch.allclose 进行比较，允许小的浮点误差
                        # atol (absolute tolerance), rtol (relative tolerance) 可能需要调整
                        are_close = torch.allclose(
                            current_initial_part,
                            prev_generated_wav_chunk_full,
                            atol=1e-5,
                            rtol=1e-5,
                        )

                        if not are_close:
                            # 如果不一致，打印警告信息和差异范数
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
                        # (可选) 如果需要，可以打印匹配信息
                        # print(f"INFO: Consistency Check Passed for item {item_name} at step {i+1}.")

                    else:
                        # 如果当前生成的音频比上一步的还短（理论上不应发生，除非模型或输入有问题）
                        print(
                            f"WARNING: Current generated audio ({generated_wav_chunk_full.shape[-1]} samples) "
                            f"is shorter than previous ({len_to_compare} samples) for item {item_name} at step {i+1}. Cannot compare."
                        )

                # --- 存储当前完整生成结果，用于下一次比较 ---
                # 使用 .clone() 避免后续操作影响存储的值
                prev_generated_wav_chunk_full = generated_wav_chunk_full.clone()

                # --- 提取当前步骤对应的 *新* 音频段 ---
                # 计算需要提取的音频样本的起始和结束索引
                start_audio_sample = start_mel_frame * hop_size
                # end_audio_sample 应该是 end_mel_frame * hop_size
                # 但要确保不超过实际生成的音频长度 (generated_wav_chunk_full 的总长)
                end_audio_sample = min(
                    end_mel_frame * hop_size, generated_wav_chunk_full.shape[-1]
                )

                # 从生成的完整音频块中提取我们需要的 新段 (对应于 mel[start_mel_frame:end_mel_frame])
                # 注意：索引是相对于 generated_wav_chunk_full 的
                required_segment = generated_wav_chunk_full[
                    :, :, start_audio_sample:end_audio_sample
                ]
                generated_wav_segments.append(required_segment)

            # --- 拼接所有提取的音频段 ---
            if generated_wav_segments:
                wav_pred_incremental = torch.cat(
                    generated_wav_segments, dim=-1
                )  # 拼接最后一个维度 (时间)
            else:
                wav_pred_incremental = torch.zeros_like(
                    wav_gt_single[:, :, :0]
                )  # 处理空 mel 情况

            # --- 保存音频 ---
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

            # --- 保存 F0 (如果需要) ---
            if hparams.get("save_f0", False) and f0_single is not None:
                f0_to_save = f0_single.squeeze().cpu().numpy()
                np.save(os.path.join(f0_dir, f"{item_name}_f0.npy"), f0_to_save)

        return loss_output  # 返回空的 loss_output


import os
import time
import yaml
import torch
import numpy as np

from modules.hifigan.hifigan import HifiGanGenerator
from utils.hparams import hparams
import utils.audio as audio  # 用于保存 WAV 文件


def deep_merge_dicts(base: dict, override: dict) -> dict:
    """
    递归地将 override 合并到 base 中：
    - 如果某个 key 在 override 中也是 dict，则递归 merge；
    - 否则直接用 override 覆盖 base。
    返回一个新的字典，不会修改传入参数。
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
    递归地加载一个 YAML 文件及其所有 base_config：
    - yaml_path：待加载的文件路径（绝对路径或相对路径均可）。
    - project_root：项目根目录，用于解析以 "egs/" 开头的 base_config 路径。
    - visited：用于记录已加载过的路径，避免循环依赖。
    返回合并后的 dict（先加载 base，再加载当前文件，以当前文件为 override）。
    """
    if visited is None:
        visited = set()

    yaml_path = os.path.abspath(yaml_path)
    if yaml_path in visited:
        return {}
    visited.add(yaml_path)

    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(f"找不到 YAML 文件：{yaml_path}")

    # —— 1. 读取当前层 YAML 内容 —— #
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # —— 2. 如果存在 base_config，则先递归加载它们 —— #
    base_merged = {}
    if "base_config" in cfg:
        base_entry = cfg["base_config"]
        if isinstance(base_entry, str):
            base_list = [base_entry]
        elif isinstance(base_entry, list):
            base_list = base_entry
        else:
            raise ValueError(
                f"base_config 必须是字符串或字符串列表，但在 {yaml_path} 中得到 {type(base_entry)}"
            )

        for base_rel in base_list:
            # 如果 base_rel 是绝对路径，直接使用；
            # 如果以 "egs/" 开头，就当成相对于 project_root（项目根）的路径；
            # 否则当作相对于当前 YAML 文件所在目录的相对路径。
            if os.path.isabs(base_rel):
                base_path = base_rel
            elif base_rel.startswith("egs/"):
                # 以项目根为基准拼接
                base_path = os.path.join(project_root, base_rel)
            else:
                base_dir = os.path.dirname(yaml_path)
                base_path = os.path.join(base_dir, base_rel)

            base_path = os.path.abspath(base_path)
            base_cfg_dict = load_recursive_yaml(base_path, project_root, visited)
            base_merged = deep_merge_dicts(base_merged, base_cfg_dict)

    # —— 3. 合并：base_merged + 本层 cfg，以本层 cfg 为优先 —— #
    merged = deep_merge_dicts(base_merged, cfg)
    return merged


def load_hparams_with_base(main_yaml_path: str):
    """
    从指定的主 YAML（及其所有 base_config）加载所有超参数到 utils.hparams.hparams 中。
    """
    # 先计算 project_root：假设主 YAML 在 "…/project_root/egs_usr/…"
    # 那么 project_root 就是主 YAML 所在目录的上一级
    main_yaml_abspath = os.path.abspath(main_yaml_path)
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(main_yaml_abspath), "..")
    )

    merged_cfg = load_recursive_yaml(main_yaml_abspath, project_root)
    # 清空现有 hparams，写入新值
    hparams.clear()
    hparams.update(merged_cfg)
    return hparams


if __name__ == "__main__":
    # ----------------------------------------------------------------------
    # 1. 主 YAML 路径（请根据实际情况修改）
    main_yaml = "/home2/zhangyu/hifigan_casual/egs_usr/hifinsf_16k320_shuffle.yaml"
    if not os.path.isfile(main_yaml):
        raise FileNotFoundError(f"找不到主 YAML 文件：{main_yaml}")

    # 2. 加载并合并所有 base_config
    load_hparams_with_base(main_yaml)
    print("------ 已加载完整的 hparams 配置 ------")

    # 强制将 mel_bins / n_mel_channels 设为 80
    hparams["n_mel_channels"] = 80
    hparams["mel_bins"] = 80

    print(f"采样率 audio_sample_rate = {hparams.get('audio_sample_rate')}")
    print(f"Mel 通道数 n_mel_channels = {hparams.get('n_mel_channels')}")

    # ----------------------------------------------------------------------
    # 3. 选择设备（优先 CUDA，否则 CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备：", device)

    # ----------------------------------------------------------------------
    # 4. 构建并加载 HiFi-GAN Generator
    generator = HifiGanGenerator(hparams).to(device)

    # 如果你有预训练的 checkpoint，可以把下面代码取消注释并修改路径：
    # ckpt_path = "/home2/zhangyu/hifigan_casual/checkpoints/generator.pth"
    # if os.path.isfile(ckpt_path):
    #     checkpoint = torch.load(ckpt_path, map_location=device)
    #     generator.load_state_dict(checkpoint["generator"], strict=True)
    #     print("已加载预训练权重：", ckpt_path)
    # else:
    #     print("未找到指定的 Generator checkpoint，将使用随机初始化模型。")

    generator.eval()

    # ----------------------------------------------------------------------
    # 5. 生成一帧随机的 Mel 谱 输入供推理测试
    n_mel_channels = hparams["n_mel_channels"]
    mel_frame = torch.randn(1, n_mel_channels, 4, device=device)

    # if hparams.get("use_pitch_embed", False):
    #     # 随机生成一个 f0 值，形状 (1, 1)
    #     f0_frame = torch.randn(1, 1, device=device) * 5 + 100
    # else:
    f0_frame = None

    # 预热推理一次
    with torch.no_grad():
        _ = generator(mel_frame, f0_frame)

    # ----------------------------------------------------------------------
    # 6. 多次推理并测量平均延迟（单位：毫秒）
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
    print(f"单帧 Mel 推理平均延迟：{avg_latency_ms:.2f} ms")

    # ----------------------------------------------------------------------
    # 7. （可选）将最后一次生成的 wav_out 保存为 WAV 文件，方便试听
    wav_np = wav_out.squeeze().cpu().numpy()
    save_path = os.path.join(os.getcwd(), "test_output.wav")
    audio.save_wav(wav_np, save_path, hparams["audio_sample_rate"])
    print(f"已将生成的音频保存到：{save_path}")

    print("推理测试结束。")
