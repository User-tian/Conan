import os
import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav

def get_spk_embedding(encoder, wav_path):
    """
    输入：
      - encoder: 已初始化的 VoiceEncoder 实例
      - wav_path: 音频文件路径（支持 wav/mp3/flac 等常见格式）
    返回：
      - 已经 L2 归一化的说话人向量（numpy.ndarray）
    """
    wav = preprocess_wav(wav_path)
    emb = encoder.embed_utterance(wav)
    return emb

def cosine_similarity(vec1, vec2):
    """
    计算两个向量的余弦相似度
    """
    v1 = np.asarray(vec1)
    v2 = np.asarray(vec2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))

def find_files_by_tag(folder, tag):
    """
    返回 folder 下所有文件名中包含 tag 的文件的完整路径列表
    """
    file_list = []
    for fname in os.listdir(folder):
        if tag in fname:
            full_path = os.path.join(folder, fname)
            if os.path.isfile(full_path):
                file_list.append(full_path)
    return file_list

def sort_by_basename_key(file_list, tag_to_remove):
    """
    对 file_list 中的路径进行排序，排序 key 为：
    basename（去掉扩展名）中去除 tag_to_remove 后的部分。
    例如：
      - file "abc_[P].wav"，去掉 "[P]" 后 key="abc_.wav" → 再去掉扩展名 → key="abc_"
      - file "abc_[G].wav"，去掉 "[G]" 后 key="abc_.wav" → 再去掉扩展名 → key="abc_"
    这样才能保证 [P]/[G] 配对排序一致。
    """
    def key_func(full_path):
        base = os.path.basename(full_path)
        name, _ext = os.path.splitext(base)
        # 将 "[P]" 或 "[G]" 删除
        key_name = name.replace(tag_to_remove, "")
        return key_name
    return sorted(file_list, key=key_func)

if __name__ == "__main__":
    folder = "/home/zy/VC/checkpoints/stage1_fast/generated_160000_/wavs"

    # 1. 找到两组文件
    p_files = find_files_by_tag(folder, "[P]")
    g_files = find_files_by_tag(folder, "[G]")

    if len(p_files) == 0:
        print(f"警告：在目录 `{folder}` 中未找到任何包含 \"[P]\" 的文件。")
        exit(1)
    if len(g_files) == 0:
        print(f"警告：在目录 `{folder}` 中未找到任何包含 \"[G]\" 的文件。")
        exit(1)

    # 2. 分别排序，使得去掉 "[P]"/"[G]" 后的 basename 相同的两组文件能一一对应
    p_files_sorted = sort_by_basename_key(p_files, "[P]")
    g_files_sorted = sort_by_basename_key(g_files, "[G]")

    # 检查两组排序后长度是否一致
    if len(p_files_sorted) != len(g_files_sorted):
        print(f"警告：排序后 [P] 文件数量 ({len(p_files_sorted)}) 与 [G] 文件数量 ({len(g_files_sorted)}) 不一致。")
        print("这可能是因为某些文件名去掉标记后没有一一对应。")
        # 虽然数量不一致，但我们仍然按最小长度配对
    pair_count = min(len(p_files_sorted), len(g_files_sorted))
    print(f"共找到 {len(p_files_sorted)} 个 [P] 文件，{len(g_files_sorted)} 个 [G] 文件。将按序配对前 {pair_count} 对进行计算。")

    # 3. 初始化 VoiceEncoder（默认使用 CPU）
    encoder = VoiceEncoder()

    sim_list = []
    for idx in range(pair_count):
        p_path = p_files_sorted[idx]
        g_path = g_files_sorted[idx]
        try:
            p_emb = get_spk_embedding(encoder, p_path)
        except Exception as e:
            print(f"提取 `{p_path}` 嵌入时出错：{e}")
            continue
        try:
            g_emb = get_spk_embedding(encoder, g_path)
        except Exception as e:
            print(f"提取 `{g_path}` 嵌入时出错：{e}")
            continue

        sim = cosine_similarity(p_emb, g_emb)
        sim_list.append(sim)
        print(f"Pair {idx+1}: [P]({os.path.basename(p_path)})  <->  [G]({os.path.basename(g_path)})  相似度 = {sim:.4f}")

    if len(sim_list) == 0:
        print("没有成功计算到任何相似度。")
        exit(1)

    mean_sim = float(np.mean(sim_list))
    print("──────────────────────────────────────────")
    print(f"共计算 {len(sim_list)} 对，平均相似度 = {mean_sim:.4f}")
