import math
import random
import numpy as np
import torch
import torch.nn as nn
from types import MethodType

# 假设模型和工具函数可以正确导入
from modules.TechSinger.techsinger import RFSinger # <--- 你的 RFSinger 类 (确保已修改 forward)
# from modules.TechSinger.diff.net import F0DiffNet # (如果需要)
from modules.TechSinger.flow.flow_f0 import ReflowF0 # <--- 确保已修改 forward 以处理 initial_noise

from utils.commons.hparams import set_hparams, hparams
from utils.commons.ckpt_utils import load_ckpt
from utils.audio.pitch.utils import denorm_f0, f0_to_coarse

# ---------------- 配置 ----------------
CONFIG_PATH = "/home/zy/VC/egs/stage1.yaml" # <--- 请确保路径正确
CKPT_PATH   = "" # <--- 请确保路径正确, 或设为 ""

# ---------------- 时间尺度 (将在 main 函数中从 hparams 加载后更新) ----------------
FRAME_MS = 20
TOKEN_MS = 80
FRAMES_PER_TOKEN = 4

# ---------------- 工具函数 ----------------
def frames_from_ms(ms: int) -> int:
    if FRAME_MS <= 0: return 0
    return int(round(ms / FRAME_MS))

def ms_to_tokens(ms: int) -> int:
    num_frames = frames_from_ms(ms)
    if FRAMES_PER_TOKEN <= 0: return 0
    return math.ceil(num_frames / FRAMES_PER_TOKEN)

def calculate_diffs(data1: torch.Tensor | None, data2: torch.Tensor | None, compare_len: int) -> tuple[float, float]:
    mean_diff, max_diff = -1.0, -1.0
    if data1 is None or data2 is None or data1.numel() == 0 or data2.numel() == 0:
        return mean_diff, max_diff
    d1, d2 = data1[0], data2[0]
    T = min(compare_len, d1.size(0), d2.size(0))
    if T <= 0: return 0.0, 0.0
    slice1, slice2 = d1[:T], d2[:T]
    abs_diff = (slice1 - slice2).abs()
    mean_diff = abs_diff.mean().item()
    max_diff = abs_diff.max().item()
    return mean_diff, max_diff

# ------------- 增量式前缀一致性测试函数 (使用预生成并切片的噪声) -------------
@torch.no_grad()
def test_incremental_prefix_consistency(
    model: RFSinger,
    base_content: torch.Tensor, # 提供足够长的基础 content
    base_target: torch.Tensor,  # 提供足够长的基础 target
    spk_embed: torch.Tensor | None,
    module_key: str,          # 要测试的模块输出键名
    base_ms: int = 80,        # 起始毫秒数
    increment_ms: int = 80,   # 每次增加的毫秒数
    num_steps: int = 3        # 测试的步数
):
    """
    测试指定模块输出的增量式前缀一致性 (使用预生成并切片的噪声)。
    逐步增加输入长度，比较当前运行输出的前缀与上一步运行的完整输出。
    仅当测试 F0 或 Mel 时才使用和传递噪声。
    """
    print(f"\n--- 开始增量测试模块: {module_key} (步长 {increment_ms}ms, 共 {num_steps} 步, 使用切片噪声) ---")
    device = base_content.device
    previous_output = None
    previous_ms = 0
    results = {}
    model = model.to(device) # 确保模型在正确设备

    # --- 预先生成最大长度的噪声 (仅当测试 F0 或 Mel 时需要) ---
    full_noise = None
    is_stochastic_module = module_key in ['f0_denorm_pred', 'mel_out']
    if is_stochastic_module:
        max_ms = base_ms + (num_steps - 1) * increment_ms
        max_tokens = ms_to_tokens(max_ms)
        # 确定噪声形状，需要知道 F0 维度 (通常为 1)
        f0_dim = getattr(model.f0_gen, 'mel_bins', 1) if hasattr(model, 'f0_gen') else 1 # 尝试获取 F0 维度
        batch_size = base_content.size(0)
        # 噪声形状：[B, 1, F0_dim, T_tok]
        noise_shape_max = (batch_size, 1, f0_dim, max_tokens)
        try:
            full_noise = torch.randn(noise_shape_max, device=device)
            print(f"  已预生成完整噪声，形状: {full_noise.shape}")
        except Exception as e:
            print(f"错误：生成完整噪声时失败: {e}")
            # 如果无法生成噪声，后续将无法传递，测试仍可进行但无共享噪声
            full_noise = None # 确保 full_noise 为 None

    # 总的 token 和 frame 长度
    T_tok_total = base_content.size(1)
    T_frame_total = base_target.size(1)

    for i in range(1, num_steps + 1):
        current_ms = base_ms + (i - 1) * increment_ms
        print(f"\n  第 {i} 步: 计算 {current_ms}ms 输出...")

        # 计算当前步所需 token 和 frame 长度
        current_frames = frames_from_ms(current_ms)
        current_tokens = math.ceil(current_frames / FRAMES_PER_TOKEN) if FRAMES_PER_TOKEN > 0 else 0

        if current_tokens <= 0 or current_frames <= 0:
            print("错误：计算出的 token 或 frame 长度无效。")
            continue
        if current_tokens > T_tok_total or current_frames > T_frame_total:
            print(f"警告: 需要的长度 ({current_tokens} tokens, {current_frames} frames) 超出基础数据范围，测试提前结束。")
            break

        # 准备当前步的输入
        current_content = base_content[:, :current_tokens].to(device)
        current_target = base_target[:, :].to(device)

        # --- 准备当前步要传递的噪声 (从 full_noise 切片) ---
        noise_for_this_step = None
        noise_status = "不适用"
        if is_stochastic_module and full_noise is not None:
             try:
                 # 截取与当前 token 数量匹配的前缀
                 if full_noise.shape[-1] >= current_tokens:
                     noise_for_this_step = full_noise[..., :current_tokens].clone()
                     noise_status = f"使用预生成噪声前缀 (T={current_tokens})"
                 else:
                     noise_status = "预生成噪声长度不足"
             except Exception as e:
                 print(f"错误：截取噪声时失败: {e}")
                 noise_status = "截取噪声出错"
        elif is_stochastic_module and full_noise is None:
             noise_status = "预生成噪声失败"


        # --- 执行模型推理 ---
        out_current = model.forward(current_content,
                                    spk_embed=spk_embed.to(device) if spk_embed is not None else None,
                                    target=current_target,
                                    ref=current_target,
                                    infer=True,
                                    initial_noise=noise_for_this_step) # 传递切片后的噪声
        current_output = out_current.get(module_key, None)

        if current_output is None or current_output.numel() == 0:
            print(f"警告: 未能在第 {i} 步获取到有效的模块输出 '{module_key}'。")
            previous_output = None
            continue

        # --- 如果不是第一步，进行比较 ---
        if i > 1 and previous_output is not None:
            compare_len = previous_output[0].size(0) # 比较长度是上一步输出的完整长度
            mean_diff, max_diff = calculate_diffs(current_output.to(device), previous_output.to(device), compare_len)

            step_label = f"{previous_ms}ms -> {current_ms}ms"
            results[step_label] = {'mean': mean_diff, 'max': max_diff, 'len': compare_len, 'noise_status': noise_status}

            print(f"  比较 {current_ms}ms 输出前缀 vs {previous_ms}ms 完整输出 (比较长度 T={compare_len})")
            print(f"    噪声状态: {noise_status}")
            print(f"    Mean Abs Diff: {'%.6f' % mean_diff if mean_diff != -1.0 else 'N/A'}")
            print(f"    Max Abs Diff:  {'%.6f' % max_diff if max_diff != -1.0 else 'N/A'}")

        # 更新上一步的输出和毫秒数
        previous_output = current_output.clone()
        previous_ms = current_ms

    return results

def replace_time_norm(module: nn.Module):
    """
    遍历整网，把 LayerNorm / InstanceNorm1d 就地替换成 GroupNorm(1, C)。
    权重、偏置直接复制，等价于“切掉时间归一化”。
    """
    for name, child in list(module.named_children()):
        if isinstance(child, (nn.LayerNorm, nn.InstanceNorm1d)):
            C = child.normalized_shape[0] if isinstance(child, nn.LayerNorm) else child.num_features
            gn = nn.GroupNorm(1, C, affine=True)
            # 复制原有权重和偏置
            setattr(module, name, gn)
        else:
            replace_time_norm(child)

# ------------- 主流程 (调用新的增量测试函数) -------------
def main() -> None:
    # 1. 加载配置 & 模型
    try:
        set_hparams(CONFIG_PATH)
        global FRAME_MS, TOKEN_MS, FRAMES_PER_TOKEN
        FRAME_MS = hparams.get('hop_size', 320) / hparams.get('sample_rate', 16000) * 1000
        TOKEN_MS = hparams.get('token_ms', 80)
        FRAMES_PER_TOKEN = int(TOKEN_MS / FRAME_MS) if FRAME_MS > 0 else 4
        print(f"HParams loaded. FRAME_MS={FRAME_MS:.2f}, TOKEN_MS={TOKEN_MS}, FRAMES_PER_TOKEN={FRAMES_PER_TOKEN}")
    except Exception as e:
        print(f"加载配置失败: {e}")
        return

    # ===> 强制使用 CPU <===
    device = torch.device("cpu")
    print(f"\n强制使用设备: {device}")
    # ===>             <===

    model = RFSinger(dict_size=0, hparams=hparams).to(device)
    model.eval()
    if CKPT_PATH:
        print(f"正在加载模型权重: {CKPT_PATH} ...")
            # 确保你的 load_ckpt 支持 map_location 或能正确处理 CPU 加载
        load_ckpt(model, CKPT_PATH, strict=True) # 假设 load_ckpt 能处理设备
            # 或者如果需要: load_ckpt(model, CKPT_PATH, strict=True, map_location=device)


    # 调用
    # replace_time_norm(model)
    # print("✅ 已把所有时间归一化层替换为 GroupNorm")


    # 2. 构造足够长的基础输入数据 (确保在 CPU 上)
    base_ms = 80
    increment_ms = 80
    num_steps = 10
    max_ms = base_ms + (num_steps - 1) * increment_ms

    T_tok_max = ms_to_tokens(max_ms) + 5 # 加一点余量
    T_frame_max = frames_from_ms(max_ms) + int(5 * FRAMES_PER_TOKEN)

    if T_frame_max <= 0 or T_tok_max <= 0:
        print("错误: 计算出的最大 Token 或 Frame 长度 <= 0。")
        return

    content = torch.randint(1, 101, (1, T_tok_max), device=device)
    mel_dim = hparams.get('audio_num_mel_bins', 80)
    target  = torch.randn(1, T_frame_max, mel_dim, device=device)
    spk_embed = None

    print(f"基础 Content shape: {content.shape}, 基础 Target shape: {target.shape}")

    # 3. 设置随机种子
    # random.seed(58)
    # np.random.seed(58)
    # torch.manual_seed(58)

    # --- 执行增量式前缀一致性测试 (使用切片噪声) ---
    print(f"\n--- 开始增量式前缀一致性测试 (CPU, 切片噪声) ---")

    # 测试 Content Encoder
    # ce_results = test_incremental_prefix_consistency(
    #     model, content, target, spk_embed,
    #     module_key='content_embed_proj',
    #     base_ms=base_ms, increment_ms=increment_ms, num_steps=num_steps
    #     # Content Encoder 不涉及随机噪声，use_shared_noise 无效
    # )

    # 测试 F0 输出
    f0_results = test_incremental_prefix_consistency(
        model, content, target, spk_embed,
        module_key='f0_denorm_pred',
        base_ms=base_ms, increment_ms=increment_ms, num_steps=num_steps
        # 内部会尝试使用切片噪声
    )

    # 测试 Mel 输出
    mel_results = test_incremental_prefix_consistency(
        model, content, target, spk_embed,
        module_key='mel_out',
        base_ms=base_ms, increment_ms=increment_ms, num_steps=num_steps
        # 内部会尝试使用切片噪声
    )

    # --- 结果汇总 ---
    print("\n--- 增量式前缀一致性测试结果汇总 (CPU, 切片噪声) ---")

    # print("\nContent Encoder ('content_embed_proj'):")
    # if ce_results:
    #     for step, res in ce_results.items():
    #         print(f"  {step}: Mean={res['mean']:.6f}, Max={res['max']:.6f} (Len={res['len']})")
    # else: print("  无结果。")

    print("\nF0 Predictor Output ('f0_denorm_pred'):")
    if f0_results:
        for step, res in f0_results.items():
            print(f"  {step}: Mean={res['mean']:.6f}, Max={res['max']:.6f} (Len={res['len']}), Noise: {res['noise_status']}")
    else: print("  无结果。")

    print("\nMel Decoder Output ('mel_out'):")
    if mel_results:
        for step, res in mel_results.items():
            print(f"  {step}: Mean={res['mean']:.6f}, Max={res['max']:.6f} (Len={res['len']}), Noise: {res['noise_status']}")
    else: print("  无结果。")

    print("\n--- 测试结束 ---")


if __name__ == "__main__":
    main()