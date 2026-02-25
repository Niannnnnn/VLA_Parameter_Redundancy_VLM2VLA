import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import re
import pandas as pd
from prismatic import load
import csv

def extract_model_info(model, model_name):
    """提取模型的结构和参数信息"""
    try:
        # 获取模型结构
        structure = {
            'name': model_name,
            'num_parameters': sum(p.numel() for p in model.parameters()),  # 总参数数量
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),  # 可训练参数数量
        }
        return structure
    except Exception as e:
        print(f"提取模型信息时出错：{e}")
        return None

def show_model_params(model, save_path):
    # print(f"模型参数名称与尺寸如下：")
    state_dict = model.state_dict()
    # for name, param in state_dict.items():
    #     print(f"{name:60s} {tuple(param.shape)}")

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(str(model))

    
    print(f"✅ 参数信息已保存到 {save_path}")

def weight_mapping(model_orig, model_finetuned, save_path):
    plm_state = model_orig.state_dict()
    olm_state = model_finetuned.state_dict()

    # 定义匹配前缀映射关系
    prefix_map = [
        ("llm_backbone.llm.model.layers.", "module.language_model.model.layers."),
    ]

    # 定义我们关心的权重关键字
    target_suffixes = [
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "mlp.up_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.down_proj.weight",
        # 如果还想保留 layernorm，可加上：
        # "input_layernorm.weight",
        # "post_attention_layernorm.weight",
    ]

    mapping_list = []

    for plm_name in plm_state.keys():
        # 只考虑 weight
        if not plm_name.endswith("weight"):
            continue

        # 只匹配我们关心的层（q,k,v,o,up,gate,down）
        if not any(suffix in plm_name for suffix in target_suffixes):
            continue

        for plm_prefix, olm_prefix in prefix_map:
            if plm_name.startswith(plm_prefix):
                olm_name = plm_name.replace(plm_prefix, olm_prefix)
                if olm_name in olm_state:
                    mapping_list.append((plm_name, olm_name))
                else:
                    mapping_list.append((plm_name, olm_name + "   # NOT FOUND"))
                break

    # 保存结果
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        for plm_name, olm_name in mapping_list:
            f.write(f"{plm_name}   ->   {olm_name}\n")

    print(f"✅ 精确权重映射完成，共 {len(mapping_list)} 个匹配项。已保存至: {save_path}")
    return mapping_list

def match_parameters(model_orig, model_finetuned, output_dir):
    print("正在匹配参数...")
    # 获取两个模型的状态字典
    orig_state_dict = model_orig.state_dict()
    # 处理 DataParallel 模型的状态字典
    if hasattr(model_finetuned, 'module'):
        finetuned_state_dict = model_finetuned.module.state_dict()
    else:
        finetuned_state_dict = model_finetuned.state_dict()

    print(f"原始模型参数数量: {len(orig_state_dict)}, 微调模型参数数量: {len(finetuned_state_dict)}")
    show_model_params(model_orig, os.path.join(output_dir, "model_orig.txt"))
    show_model_params(model_finetuned, os.path.join(output_dir, "model_finetuned.txt"))

    mapping_list = weight_mapping(model_orig, model_finetuned, os.path.join(output_dir, "prismatic_openvla_mapping.txt"))

    return mapping_list

def compute_vlm_ffn_magnitude(mapping_list, model_orig, output_dir, compute_device="cuda:0"):
    print("🔍 正在计算 FFN 权重强度 (W_VLM)...")
    device = torch.device(compute_device)
    plm_state = model_orig.state_dict()
    ffn_magnitudes = {}

    for (plm_name, olm_name) in mapping_list:
        if "# NOT FOUND" in plm_name: continue
        if not any(x in plm_name for x in ["up_proj", "gate_proj", "down_proj"]): continue

        W_orig = plm_state[plm_name].to(device).float()
        if any(x in plm_name for x in ["up_proj", "gate_proj"]):
            mag = torch.norm(W_orig, p=2, dim=1)
        else:
            mag = torch.norm(W_orig, p=2, dim=0)
        ffn_magnitudes[plm_name] = mag.detach().cpu()

    os.makedirs(output_dir, exist_ok=True)
    # 完整版
    save_path = os.path.join(output_dir, "ffn_vlm_magnitude_l2.txt")
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("Layer_Name | Min_L2 | Max_L2 | Mean_L2 | Channel_Values\n")
        f.write("-" * 80 + "\n")
        for layer_name, mag_tensor in ffn_magnitudes.items():
            m_min, m_max, m_mean = mag_tensor.min(), mag_tensor.max(), mag_tensor.mean()
            val_str = ",".join([f"{x:.6f}" for x in mag_tensor.tolist()])
            f.write(f"{layer_name} | {m_min:.6f} | {m_max:.6f} | {m_mean:.6f} | {val_str}\n")
    # 简略版
    save_path_short = os.path.join(output_dir, "ffn_vlm_magnitude_l2_short.txt")
    with open(save_path_short, "w", encoding="utf-8") as f:
        f.write("Layer_Name | Min_L2 | Max_L2 | Mean_L2\n")
        f.write("-" * 50 + "\n")
        for layer_name, mag_tensor in ffn_magnitudes.items():
            f.write(f"{layer_name} | {mag_tensor.min():.6f} | {mag_tensor.max():.6f} | {mag_tensor.mean():.6f}\n")
    return ffn_magnitudes

def compute_vla_ffn_magnitude(mapping_list, model_finetuned, output_dir, compute_device="cuda:0"):
    print("🔍 正在计算 FFN 权重强度 (W_VLA)...")
    device = torch.device(compute_device)
    olm_state = model_finetuned.state_dict()
    ffn_magnitudes = {}

    for (plm_name, olm_name) in mapping_list:
        if "# NOT FOUND" in olm_name: continue
        if not any(x in olm_name for x in ["up_proj", "gate_proj", "down_proj"]): continue

        W_fine = olm_state[olm_name].to(device).float()
        if any(x in olm_name for x in ["up_proj", "gate_proj"]):
            mag = torch.norm(W_fine, p=2, dim=1)
        else:
            mag = torch.norm(W_fine, p=2, dim=0)
        ffn_magnitudes[olm_name] = mag.detach().cpu()

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "ffn_vla_magnitude_l2.txt")
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("Layer_Name | Min_L2 | Max_L2 | Mean_L2 | Channel_Values\n")
        f.write("-" * 80 + "\n")
        for layer_name, mag_tensor in ffn_magnitudes.items():
            m_min, m_max, m_mean = mag_tensor.min(), mag_tensor.max(), mag_tensor.mean()
            val_str = ",".join([f"{x:.6f}" for x in mag_tensor.tolist()])
            f.write(f"{layer_name} | {m_min:.6f} | {m_max:.6f} | {m_mean:.6f} | {val_str}\n")
    save_path_short = os.path.join(output_dir, "ffn_vla_magnitude_l2_short.txt")
    with open(save_path_short, "w", encoding="utf-8") as f:
        f.write("Layer_Name | Min_L2 | Max_L2 | Mean_L2\n")
        f.write("-" * 50 + "\n")
        for layer_name, mag_tensor in ffn_magnitudes.items():
            f.write(f"{layer_name} | {mag_tensor.min():.6f} | {mag_tensor.max():.6f} | {mag_tensor.mean():.6f}\n")
    return ffn_magnitudes

def compute_ffn_delta_magnitude(mapping_list, model_orig, model_finetuned, output_dir, compute_device="cuda:0"):
    print("🔍 正在计算 FFN 权重变化剧烈程度 (L2 of Delta W)...")
    device = torch.device(compute_device)
    plm_state = model_orig.state_dict()
    olm_state = model_finetuned.state_dict()
    ffn_deltas = {}

    for (plm_name, olm_name) in mapping_list:
        if "# NOT FOUND" in olm_name: continue
        if not any(x in olm_name for x in ["up_proj", "gate_proj", "down_proj"]): continue

        W_orig = plm_state[plm_name].to(device).float()
        W_fine = olm_state[olm_name].to(device).float()
        delta_W = W_fine - W_orig
        if any(x in olm_name for x in ["up_proj", "gate_proj"]):
            delta_mag = torch.norm(delta_W, p=2, dim=1)
        else:
            delta_mag = torch.norm(delta_W, p=2, dim=0)
        ffn_deltas[olm_name] = delta_mag.detach().cpu()

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "ffn_delta_magnitude_l2.txt")
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("Layer_Name | Min_Delta_L2 | Max_Delta_L2 | Mean_Delta_L2 | Delta_Values\n")
        f.write("-" * 80 + "\n")
        for layer_name, d_tensor in ffn_deltas.items():
            val_str = ",".join([f"{x:.6f}" for x in d_tensor.tolist()])
            f.write(f"{layer_name} | {d_tensor.min():.6f} | {d_tensor.max():.6f} | {d_tensor.mean():.6f} | {val_str}\n")
    save_path_short = os.path.join(output_dir, "ffn_delta_magnitude_l2_short.txt")
    with open(save_path_short, "w", encoding="utf-8") as f:
        f.write("Layer_Name | Min_Delta_L2 | Max_Delta_L2 | Mean_Delta_L2\n")
        f.write("-" * 50 + "\n")
        for layer_name, d_tensor in ffn_deltas.items():
            f.write(f"{layer_name} | {d_tensor.min():.6f} | {d_tensor.max():.6f} | {d_tensor.mean():.6f}\n")
    return ffn_deltas

def compute_ffn_cosine_similarity(mapping_list, model_orig, model_finetuned, output_dir, compute_device="cuda:0"):
    print("🔍 正在计算 FFN 权重方向变化 (Cosine Similarity)...")
    device = torch.device(compute_device)
    plm_state = model_orig.state_dict()
    olm_state = model_finetuned.state_dict()
    ffn_cosines = {}

    for (plm_name, olm_name) in mapping_list:
        if "# NOT FOUND" in olm_name: continue
        if not any(x in olm_name for x in ["up_proj", "gate_proj", "down_proj"]): continue

        W_orig = plm_state[plm_name].to(device).float()
        W_fine = olm_state[olm_name].to(device).float()

        # 按行或列计算 Cosine Similarity
        if any(x in olm_name for x in ["up_proj", "gate_proj"]):
            # 行向量
            W_orig_flat = W_orig / (W_orig.norm(p=2, dim=1, keepdim=True) + 1e-8)
            W_fine_flat = W_fine / (W_fine.norm(p=2, dim=1, keepdim=True) + 1e-8)
            cos_sim = (W_orig_flat * W_fine_flat).sum(dim=1)
        else:
            # 列向量
            W_orig_flat = W_orig / (W_orig.norm(p=2, dim=0, keepdim=True) + 1e-8)
            W_fine_flat = W_fine / (W_fine.norm(p=2, dim=0, keepdim=True) + 1e-8)
            cos_sim = (W_orig_flat * W_fine_flat).sum(dim=0)

        ffn_cosines[olm_name] = cos_sim.detach().cpu()

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "ffn_cosine_similarity.txt")
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("Layer_Name | Min_Cos | Max_Cos | Mean_Cos | Cos_Values\n")
        f.write("-" * 80 + "\n")
        for layer_name, c_tensor in ffn_cosines.items():
            val_str = ",".join([f"{x:.6f}" for x in c_tensor.tolist()])
            f.write(f"{layer_name} | {c_tensor.min():.6f} | {c_tensor.max():.6f} | {c_tensor.mean():.6f} | {val_str}\n")
    save_path_short = os.path.join(output_dir, "ffn_cosine_similarity_short.txt")
    with open(save_path_short, "w", encoding="utf-8") as f:
        f.write("Layer_Name | Min_Cos | Max_Cos | Mean_Cos\n")
        f.write("-" * 50 + "\n")
        for layer_name, c_tensor in ffn_cosines.items():
            f.write(f"{layer_name} | {c_tensor.min():.6f} | {c_tensor.max():.6f} | {c_tensor.mean():.6f}\n")
    return ffn_cosines

def analyze_channel_diff(vlm_data, vla_data, delta_data, cos_data, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    proj_types = ["gate_proj", "up_proj", "down_proj"]

    # 辅助函数：根据层号和类型找到正确的 Key
    def get_key_by_idx(data_dict, layer_idx, proj_type):
        for k in data_dict.keys():
            # 使用正则或更严谨的判断，确保匹配到对应的层和投影类型
            # 匹配包含 .layers.{idx}. 且包含 proj_type 的 key
            if f".layers.{layer_idx}." in k and proj_type in k:
                return k
        return None

    # 自动提取 vlm_data 中所有的层号
    all_layers = []
    for k in vlm_data.keys():
        match = re.search(r'layers\.(\d+)\.', k)
        if match:
            all_layers.append(int(match.group(1)))
    
    unique_layers = sorted(list(set(all_layers)))

    for proj in proj_types:
        layers_axis = []
        vlm_means, vla_means, delta_means, cos_means = [], [], [], []

        for idx in unique_layers:
            # 尝试在四个数据源中定位对应的 Key
            k_vlm = get_key_by_idx(vlm_data, idx, proj)
            k_vla = get_key_by_idx(vla_data, idx, proj)
            k_delta = get_key_by_idx(delta_data, idx, proj)
            k_cos = get_key_by_idx(cos_data, idx, proj)

            # 只有当四个文件都存在该层数据时才进行收集
            if all([k_vlm, k_vla, k_delta, k_cos]):
                layers_axis.append(idx)
                vlm_means.append(vlm_data[k_vlm].mean().item())
                vla_means.append(vla_data[k_vla].mean().item()) # 已修正：只添加一次
                delta_means.append(delta_data[k_delta].mean().item())
                cos_means.append(cos_data[k_cos].mean().item())

        if not layers_axis:
            print(f"⚠️ 警告: 没有找到属于 {proj} 的匹配数据，跳过绘图。")
            continue

        # --- 绘图逻辑 ---
        plt.figure(figsize=(12, 6))
        
        # 绘制四条对比曲线
        plt.plot(layers_axis, vlm_means, label='VLM (Original) Mean L2', marker='o', alpha=0.8, color='#1f77b4')
        plt.plot(layers_axis, vla_means, label='VLA (Finetuned) Mean L2', marker='s', alpha=0.8, color='#ff7f0e')
        plt.plot(layers_axis, delta_means, label='Delta (Weight Change) L2', marker='^', alpha=0.8, color='#2ca02c')
        plt.plot(layers_axis, cos_means, label='Cosine Similarity', marker='x', color='#d62728', linestyle='--')

        plt.title(f"FFN Channel Importance Analysis: {proj.upper()}", fontsize=14)
        plt.xlabel("Layer Index", fontsize=12)
        plt.ylabel("Score / Magnitude", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(loc='best')
        
        # 优化 X 轴刻度显示
        if len(layers_axis) > 10:
            plt.xticks(layers_axis[::2]) # 如果层数太多，每隔一层显示一个标签
        else:
            plt.xticks(layers_axis)

        plt.tight_layout()
        save_path = os.path.join(output_dir, f"ffn_analysis_{proj}.png")
        plt.savefig(save_path, dpi=200)
        plt.close()
        print(f"✅ 已生成分析图表: {save_path}")


import matplotlib.pyplot as plt
import seaborn as sns

def plot_channel_score_distribution(flat_scores, output_dir):
    """
    改进版：聚焦主体分布，自动剔除远端离群值（Outliers）
    """
    import numpy as np
    scores_np = flat_scores.detach().cpu().numpy()
    
    # --- 核心改进：计算 99.5% 分位数，过滤掉极少数极大值 ---
    # 这样可以确保横坐标聚焦在 0 到绝大多数数据所在的范围
    upper_limit = np.percentile(scores_np, 99.99) 
    filtered_scores = scores_np[scores_np <= upper_limit]

    plt.figure(figsize=(10, 6))
    
    # 使用更细腻的 bins=150 让山峰更平滑
    sns.histplot(filtered_scores, kde=True, color='royalblue', bins=150, alpha=0.6)
    
    # 动态设置 x 轴范围，稍微留白
    plt.xlim(left=0, right=upper_limit * 1.05)
    
    plt.xlabel("Scores of Channels", fontsize=12)
    plt.ylabel("Frequency / Density", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # 添加一个简单的文本说明，告知离群值情况
    # num_outliers = len(scores_np) - len(filtered_scores)
    # plt.text(upper_limit * 0.7, plt.ylim()[1] * 0.8, 
    #          f"Outliers excluded: {num_outliers}\nMax score: {scores_np.max():.2f}", 
    #          bbox=dict(facecolor='white', alpha=0.5))

    plot_path = os.path.join(output_dir, "ffn_score_main_distribution.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 主体分布图已保存（已过滤极大值）: {plot_path}")

def prune_channel(vlm_data, vla_data, delta_data, cos_data, output_dir, ratio=0.2, by_smallest=True):
    """
    纯 Delta 剪枝逻辑：
    只利用 delta_data (微调权重差异) 作为评分标准。
    by_smallest=True: 剪掉变化最小的 (Delta最小, 保留新知识)
    by_smallest=False: 剪掉变化最大的 (Delta最大, 剔除变动最剧烈的)
    """
    pruning_masks = {}
    all_channel_scores = []
    layer_metadata = {}

    # 1. 提取所有层号
    all_keys = list(delta_data.keys())
    unique_layers = sorted(list(set([
        int(re.search(r'layers\.(\d+)\.', k).group(1)) 
        for k in all_keys if re.search(r'layers\.(\d+)\.', k)
    ])))

    print(f"\n[剪枝配置] 目标比例: {ratio:.2%}")
    print(f"[剪枝配置] 评分逻辑: 仅使用 Delta_Magnitude (L2差异)")
    print(f"[剪枝配置] 剪枝方向: {'剪最小 (Smallest Delta)' if by_smallest else '剪最大 (Largest Delta)'}")

    # 2. 计算得分 (仅使用 delta_data)
    for idx in unique_layers:
        k_gate = next((k for k in delta_data.keys() if f".layers.{idx}." in k and "gate_proj" in k), None)
        k_up = next((k for k in delta_data.keys() if f".layers.{idx}." in k and "up_proj" in k), None)
        k_down = next((k for k in delta_data.keys() if f".layers.{idx}." in k and "down_proj" in k), None)
        
        if not all([k_gate, k_up, k_down]): continue
        
        is_protected = (idx <= 4 or idx >= 30)
        
        # --- 核心修改：只计算 Delta 的强度 ---
        # 融合 gate/up/down 三层的一致性差异
        delta_intensity = delta_data[k_gate] + delta_data[k_up] + delta_data[k_down]
        
        # 这里的得分就是 Delta 的量级
        combined_scores = delta_intensity

        layer_metadata[idx] = {
            'keys': (k_gate, k_up, k_down),
            'num_channels': len(combined_scores),
            'is_protected': is_protected,
            'scores': combined_scores
        }

        if not is_protected:
            all_channel_scores.append(combined_scores)

    # 3. 计算全局阈值
    if not all_channel_scores:
        print("❌ 错误：没有可剪枝层")
        return {}
        
    flat_scores = torch.cat(all_channel_scores)

    # 调用绘图函数（现在画的是 Delta 的分布）
    plot_channel_score_distribution(flat_scores, output_dir)
    
    # 确定阈值
    if by_smallest:
        # 剪掉变动最小的，保留 >= threshold
        num_to_prune_global = int(len(flat_scores) * ratio)
        sorted_scores, _ = torch.sort(flat_scores)
        threshold = sorted_scores[num_to_prune_global].item()
    else:
        # 剪掉变动最大的，保留 <= threshold
        num_to_prune_global = int(len(flat_scores) * (1 - ratio))
        sorted_scores, _ = torch.sort(flat_scores)
        threshold = sorted_scores[num_to_prune_global].item()

    print(f"[配置] Delta 强度阈值: {threshold:.6e}")
    print(f"{'层号':<10} | {'状态':<10} | {'剪除数':<10} | {'局部比例':<10}")
    print("-" * 55)

    # 4. 生成掩码
    for idx in unique_layers:
        if idx not in layer_metadata: continue
        meta = layer_metadata[idx]
        
        k_gate = meta['keys'][0].replace("module.", "")
        k_up = meta['keys'][1].replace("module.", "")
        k_down = meta['keys'][2].replace("module.", "")
        total_len = meta['num_channels']
        
        if meta['is_protected']:
            mask = np.ones(total_len, dtype=bool)
            status = "PROTECTED"
        else:
            if by_smallest:
                mask = (meta['scores'] >= threshold).numpy()
            else:
                mask = (meta['scores'] <= threshold).numpy()
            status = "PRUNABLE"

        pruned_count = int(np.sum(~mask))
        print(f"Layer {idx:<5} | {status:<10} | {pruned_count:<10d} | {pruned_count/total_len:>10.2%}")

        entry = {'layer_num': idx, 'pruned_channels': pruned_count, 'total_channels': total_len}
        pruning_masks[k_gate] = {**entry, 'output_mask': mask, 'input_mask': None, 'layer_type': 'gate'}
        pruning_masks[k_up] = {**entry, 'output_mask': mask, 'input_mask': None, 'layer_type': 'first'}
        pruning_masks[k_down] = {**entry, 'output_mask': None, 'input_mask': mask, 'layer_type': 'down'}

    # 5. 保存结果
    mode_str = "delta_smallest" if by_smallest else "delta_largest"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"ffn_pruning_masks_ratio_{ratio}_{mode_str}.pth")
    torch.save(pruning_masks, save_path)
    
    return pruning_masks

def load_prismatic_vlm(model_gpu_id=1):
    """使用prismatic库加载底层VLM，支持CPU或GPU，优先本地"""
    device_str = f"GPU:{model_gpu_id}" if model_gpu_id != "cpu" else "CPU"
    print(f"正在使用prismatic库加载底层VLM: prism-dinosiglip-224px+7b 到 {device_str}...")

    model_id = "prism-dinosiglip-224px+7b"
    local_dir = f"path/to/local/{model_id}"  # 替换为实际的本地路径

    try:
        # 尝试加载HF令牌(如果有)
        try:
            with open(".hf_token", "r") as f:
                hf_token = f.read().strip()
        except:
            hf_token = None
            print("未找到HF令牌文件，尝试无令牌加载")

        vlm = load(model_id, hf_token=hf_token)


        # 移动到指定设备
        if model_gpu_id == "cpu":
            vlm.to("cpu")
        else:
            vlm.to(f"cuda:{model_gpu_id}", dtype=torch.bfloat16)

        print(f"成功加载 {model_id} 到 {device_str}!")
        return vlm

    except Exception as e:
        print(f"通过prismatic库加载模型失败: {e}")
        print("尝试备选方法...")
        return load_siglip_fallback(model_gpu_id)

def load_siglip_fallback(model_gpu_id=1):
    """备选方法：使用subfolder参数加载子目录中的模型，支持CPU或GPU"""
    from transformers import AutoModel
    device_str = f"GPU:{model_gpu_id}" if model_gpu_id != "cpu" else "CPU"
    print(f"尝试使用备选方法加载模型: siglip-224px+7b 到 {device_str}...")
    
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained("google/siglip-base-patch16-224")
        config.model_type = "siglip"
        model = AutoModel.from_pretrained(
            "TRI-ML/prismatic-vlms",
            subfolder="prism-dinosiglip-224px+7b",
            config=config,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        if model_gpu_id == "cpu":
            model.to("cpu")
        else:
            model.to(f"cuda:{model_gpu_id}")
            
        return model
    except Exception as e:
        print(f"备选方法加载失败: {e}")
        print("尝试加载官方SigLIP模型作为替代...")
        
        try:
            # 不使用device_map='auto'参数
            model = AutoModel.from_pretrained(
                "google/siglip-base-patch16-224",
                torch_dtype=torch.float16
            )
            
            if model_gpu_id == "cpu":
                model.to("cpu")
            else:
                model.to(f"cuda:{model_gpu_id}")
                
            return model
        except Exception as e2:
            print(f"加载官方SigLIP模型也失败: {e2}")
            
            # 最后尝试
            try:
                print("尝试加载其他官方视觉模型作为替代...")
                model = AutoModel.from_pretrained(
                    "openai/clip-vit-base-patch16",
                    torch_dtype=torch.float16
                )
                
                if model_gpu_id == "cpu":
                    model.to("cpu")
                else:
                    model.to(f"cuda:{model_gpu_id}")
                    
                return model
            except Exception as e3:
                print(f"所有视觉模型加载尝试都失败: {e3}")
                return None

def load_openvla(model_gpu_id=1):
    """加载openvla，支持CPU或GPU"""
    from transformers import AutoModelForVision2Seq

    
    device_str = f"GPU:{model_gpu_id}" if model_gpu_id != "cpu" else "CPU"
    print(f"正在加载模型：openvla 到 {device_str}")
    
    # 首先释放显存缓存
    torch.cuda.empty_cache()

    # 设置环境变量以帮助内存管理
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    try:
        # 根据不同设备选择加载策略
        if model_gpu_id == "cpu":
            print("警告: 在CPU上加载大型模型可能会很慢且内存占用较大")
            model = AutoModelForVision2Seq.from_pretrained(
                "openvla/openvla-7b-finetuned-libero-spatial",
                torch_dtype=torch.float32,  # CPU上使用float32
                low_cpu_mem_usage=True,
                trust_remote_code=True
            ).to("cpu")
        else:
            # 尝试直接加载到指定GPU，不使用flash_attention_2
            model = AutoModelForVision2Seq.from_pretrained(
                "openvla/openvla-7b-finetuned-libero-spatial",
                torch_dtype=torch.bfloat16, 
                low_cpu_mem_usage=True, 
                trust_remote_code=True
            ).to(f"cuda:{model_gpu_id}")
            
            # 使用 DataParallel 将模型分布到指定GPU
            model = torch.nn.DataParallel(model, device_ids=[model_gpu_id])
            
        return model
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print(f"GPU:{model_gpu_id} 内存不足，尝试在CPU上加载")
            # 在CPU上加载
            try:
                model = AutoModelForVision2Seq.from_pretrained(
                    "openvla/openvla-7b-finetuned-libero-spatial",
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                ).to("cpu")
                return model
            except Exception as e2:
                print(f"在CPU上加载OpenVLA模型也失败: {e2}")
                return None
        else:
            print(f"加载OpenVLA模型时出错: {e}")
            return None


def analyze_and_generate_ffn_masks(model_orig, model_finetuned, compute_device, output_dir, ratio=0.2, by_smallest=True, prune_target="all"):
    """分析模型并生成剪裁掩码，支持选择剪枝目标 (attention/ffn/all)"""
    print("第1步: 匹配模型参数...")
    # 假设 match_parameters 和 compute_channel_diff 已经定义
    mapping_list = match_parameters(model_orig, model_finetuned, output_dir)
    
    print("\n第2步: 计算channel维度的权重差异...")
    # channel_diffs = compute_channel_diff(mapping_list, model_orig, model_finetuned, output_dir, compute_device)
    vlm_channel = compute_vlm_ffn_magnitude(mapping_list, model_orig, output_dir, compute_device)
    vla_channel = compute_vla_ffn_magnitude(mapping_list, model_finetuned, output_dir, compute_device)
    diff_l2_channel = compute_ffn_delta_magnitude(mapping_list, model_orig, model_finetuned, output_dir, compute_device)
    diff_cos_channel = compute_ffn_cosine_similarity(mapping_list, model_orig, model_finetuned, output_dir, compute_device)


    print("\n第3步: 分析通道差异分布...")
    # analyze_channel_diff(channel_diffs, output_dir)
    analyze_channel_diff(vlm_channel, vla_channel, diff_l2_channel, diff_cos_channel, "plots")

    print("\n第4步: 生成剪裁掩码...")
    pruning_results = prune_channel(vlm_channel, vla_channel, diff_l2_channel, diff_cos_channel, output_dir, ratio=ratio, by_smallest=by_smallest)
    print("生成的剪枝掩码信息:", pruning_results)
    return pruning_results 
    
    

def save_model_weights_info(model, filename):
    """
    将模型所有权重参数的名称与尺寸保存到文本文件
    :param model: torch.nn.Module
    :param filename: 输出文件名
    """
    if model is None:
        print(f"模型为 None，跳过保存 {filename}")
        return

    lines = []
    for name, param in model.named_parameters():
        lines.append(f"{name}  {tuple(param.shape)}")

    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"已保存模型权重信息到 {filename}，共 {len(lines)} 条参数。")








def main():
    # 设置环境变量
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # 优先释放 GPU 内存
    torch.cuda.empty_cache()
    
    print(f"CUDA是否可用: {torch.cuda.is_available()}")
    gpu_count = torch.cuda.device_count()
    print(f"可用GPU数量: {gpu_count}")
    
    # 从命令行参数中获取剪裁比例和剪裁模式
    import argparse
    parser = argparse.ArgumentParser(description='FFN中间维度剪裁工具')
    parser.add_argument('--ratio', type=float, default=0.2, help='剪裁比例（0.0-1.0）')
    parser.add_argument('--prune-target', type=str, default='ffn', choices=['attention', 'ffn', 'all'], 
                        help='选择剪枝目标: attention (仅剪枝attention), ffn (仅剪枝MLP), all (全部剪枝)')
    parser.add_argument('--by-smallest', action='store_true', help='剪裁变化最小的通道 (默认)')
    parser.add_argument('--by-largest', action='store_false', dest='by_smallest', help='剪裁变化最大的通道')
    parser.add_argument('--output-dir', type=str, default='./results_pruning_only_deltaW', help='输出目录')
    parser.add_argument('--compute-gpu', type=int, default=2, help='用于计算的GPU ID')
    parser.add_argument('--orig-gpu', type=int, default=1, help='加载原始模型的GPU ID')
    parser.add_argument('--fine-gpu', type=int, default=3, help='加载微调模型的GPU ID')
    parser.add_argument('--skip-orig-model', action='store_true', help='跳过加载原始模型(用于调试)')
    parser.add_argument('--skip-fine-model', action='store_true', help='跳过加载微调模型(用于调试)')
    parser.set_defaults(by_smallest=True)
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    model_orig = None
    model_finetuned = None
    
    # 加载第一个模型
    if not args.skip_orig_model:
        print(f"固定使用GPU {args.orig_gpu}加载第一个模型")
        model_orig = load_prismatic_vlm(args.orig_gpu)
    else:
        print("跳过加载原始模型 (调试模式)")
        
    # 加载第二个模型
    if not args.skip_fine_model:
        print(f"固定使用GPU {args.fine_gpu}加载第二个模型")
        model_finetuned = load_openvla(args.fine_gpu)
    else:
        print("跳过加载微调模型 (调试模式)")

    # 打印并保存权重信息
    save_model_weights_info(model_orig,  os.path.join(args.output_dir, "orig_model_weights.txt"))
    save_model_weights_info(model_finetuned, os.path.join(args.output_dir, "fine_model_weights.txt"))
    
    # 检查模型加载情况
    if (not args.skip_orig_model and model_orig is None) or (not args.skip_fine_model and model_finetuned is None):
        print("警告：至少一个模型加载失败。")
        
        # 进入模拟调试模式
        print("是否进入模拟调试模式继续分析？(输入y表示是，任意键表示否)")
        response = input().strip().lower()
        
        if response != 'y':
            print("退出程序。")
            return
        
        print("进入模拟调试模式，使用随机权重模拟模型...")
        
        # 创建简单模型结构进行调试
        if model_orig is None and not args.skip_orig_model:
            from torch import nn
            model_orig = nn.Sequential(
                nn.Linear(768, 3072),
                nn.GELU(),
                nn.Linear(3072, 768),
            )
            print("创建了模拟原始模型")
        
        if model_finetuned is None and not args.skip_fine_model:
            from torch import nn
            model_finetuned = nn.Sequential(
                nn.Linear(768, 3072),
                nn.GELU(),
                nn.Linear(3072, 768),
            )
            print("创建了模拟微调模型")
    
    # 记录模型所在的设备
    if model_orig is not None:
        orig_device = f"cuda:{args.orig_gpu}" if args.orig_gpu != "cpu" else "cpu"
        print(f"原始模型在设备: {orig_device}")
    
    if model_finetuned is not None:
        fine_device = f"cuda:{args.fine_gpu}" if args.fine_gpu != "cpu" else "cpu"
        print(f"微调模型在设备: {fine_device}")
    
    # 提取模型信息
    if model_orig is not None:
        model_orig_info = extract_model_info(model_orig, "原始模型")
    else:
        model_orig_info = None
        
    if model_finetuned is not None:
        model_finetuned_info = extract_model_info(model_finetuned, "微调模型")
    else:
        model_finetuned_info = None
    
    if model_orig_info and model_finetuned_info:
        print(f"=== 模型参数数量比较 ===")
        print(f"{model_orig_info['name']} - 总参数量: {model_orig_info['num_parameters']}, 可训练参数量: {model_orig_info['trainable_parameters']}")
        print(f"{model_finetuned_info['name']} - 总参数量: {model_finetuned_info['num_parameters']}, 可训练参数量: {model_finetuned_info['trainable_parameters']}")
    else:
        print("无法提取完整的模型信息，跳过参数比较。")
    
    # 固定使用指定GPU进行计算
    compute_device = f"cuda:{args.compute_gpu}"
    print(f"固定使用 {compute_device} 进行参数差异计算")


    analyze_and_generate_ffn_masks(
            model_orig, 
            model_finetuned, 
            compute_device, 
            output_dir = args.output_dir,
            ratio=args.ratio, 
            by_smallest=args.by_smallest,
            prune_target=args.prune_target
        )
    

if __name__ == "__main__":
    main()