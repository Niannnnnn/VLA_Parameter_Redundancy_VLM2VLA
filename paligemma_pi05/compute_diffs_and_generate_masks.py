import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict
from pathlib import Path
from datetime import datetime
from safetensors.torch import load_file
from transformers import AutoModel

# ==========================================
# 配置与路径
# ==========================================
# VLA_PATH = Path("/home/intern/zhangfengnian/checkpoints/pi05_libero_pytorch/model.safetensors")
VLA_PATH = Path("/mnt/afs/huangtao/intern/zhangfengnian/checkpoints/pi05_libero_pytorch/model.safetensors")
VLM_NAME = "google/paligemma-3b-mix-224"
OUTPUT_DIR = Path("./analysis_results_pi05")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# PaLiGemma 3B 架构参数
V_HIDDEN = 1152
V_HEADS = 16
L_HIDDEN = 2048
L_Q_HEADS = 8
L_KV_HEADS = 1

# ==========================================
# 核心计算函数
# ==========================================

def get_importance_score(vla_weight, vlm_weight, calc_dim=1):
    """计算 L2 Norm 差异"""
    delta_w = vla_weight.to(torch.float32) - vlm_weight.to(torch.float32)
    score = torch.norm(delta_w, p=2, dim=calc_dim) 
    return score.cpu().numpy()

def process_pi05_importance():
    print("🚀 正在加载权重并计算差异 (VLM vs VLA)...")
    vla_sd = load_file(VLA_PATH)
    vlm_model = AutoModel.from_pretrained(VLM_NAME, torch_dtype=torch.float16)
    vlm_sd = vlm_model.state_dict()

    importance_data = {
        "vision_ffn": {}, "vision_attn": {},
        "llm_ffn": {}, "llm_attn": {}
    }

    # 1. Vision Tower (27层)
    for i in range(27):
        vla_pre = f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}"
        vlm_pre = f"vision_tower.vision_model.encoder.layers.{i}"
        
        # FFN (fc1)
        importance_data["vision_ffn"][f"layer.{i}"] = get_importance_score(
            vla_sd[f"{vla_pre}.mlp.fc1.weight"], vlm_sd[f"{vlm_pre}.mlp.fc1.weight"])

        # ATTN (MHA) -> 聚合为 Head
        q_score = get_importance_score(
            vla_sd[f"{vla_pre}.self_attn.q_proj.weight"], vlm_sd[f"{vlm_pre}.self_attn.q_proj.weight"])
        importance_data["vision_attn"][f"layer.{i}"] = q_score.reshape(V_HEADS, -1).mean(axis=1)

    # 2. Language Model (18层)
    for i in range(18):
        vla_pre = f"paligemma_with_expert.paligemma.model.language_model.layers.{i}"
        vlm_pre = f"language_model.layers.{i}"

        # FFN (gate_proj)
        importance_data["llm_ffn"][f"layer.{i}"] = get_importance_score(
            vla_sd[f"{vla_pre}.mlp.gate_proj.weight"], vlm_sd[f"{vlm_pre}.mlp.gate_proj.weight"])

        # ATTN (GQA) -> 聚合为 Head Group
        q_score = get_importance_score(
            vla_sd[f"{vla_pre}.self_attn.q_proj.weight"], vlm_sd[f"{vlm_pre}.self_attn.q_proj.weight"])
        k_score = get_importance_score(
            vla_sd[f"{vla_pre}.self_attn.k_proj.weight"], vlm_sd[f"{vlm_pre}.self_attn.k_proj.weight"])
        
        q_heads = q_score.reshape(L_Q_HEADS, -1).mean(axis=1)
        importance_data["llm_attn"][f"layer.{i}"] = (q_heads + k_score.mean()) / 2

    return importance_data

# ==========================================
# 自动化掩码生成与动态命名
# ==========================================

def generate_pruning_masks(importance_dict, ratio=0.2, by_smallest=True, component="llm_ffn"):
    """
    by_smallest=True: 剪掉差异最小的 (保留核心变化)
    by_smallest=False: 剪掉差异最大的 (测试性能受损)
    """
    all_scores = []
    layer_keys = sorted(importance_dict.keys(), key=lambda x: int(x.split('.')[-1]))
    for k in layer_keys: all_scores.append(importance_dict[k])
    
    flat_scores = np.concatenate(all_scores)
    num_total = len(flat_scores)
    num_to_prune = int(num_total * ratio)
    
    # 策略确定
    sorted_indices = np.argsort(flat_scores)
    strategy = "smallest" if by_smallest else "biggest"
    
    if by_smallest:
        prune_indices = sorted_indices[:num_to_prune]
    else:
        prune_indices = sorted_indices[-num_to_prune:]
        
    global_mask = np.ones(num_total, dtype=bool)
    global_mask[prune_indices] = False
    
    # 构建包含元数据的丰富信息字典
    pruning_results = {
        "metadata": {
            "component": component,
            "ratio": ratio,
            "strategy": strategy,
            "total_units": num_total,
            "pruned_units": num_to_prune,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
        },
        "layers": {}
    }
    
    curr = 0
    for k in layer_keys:
        n = len(importance_dict[k])
        layer_mask = global_mask[curr : curr + n]
        curr += n
        pruning_results["layers"][k] = {
            "mask": layer_mask,
            "pruned_count": int(np.sum(~layer_mask)),
            "kept_count": int(np.sum(layer_mask)),
            "total_count": n
        }

    # 自动生成文件名
    filename = f"masks_{component}_r{ratio}_{strategy}.pth"
    save_path = OUTPUT_DIR / filename
    torch.save(pruning_results, save_path)
    
    print(f"✅ 保存掩码: {filename} | 剪枝比例: {ratio*100:.1f}% | 策略: {strategy}")
    return pruning_results

# ==========================================
# 可视化 (OpenVLA 风格)
# ==========================================
'''
def plot_pi05_heatmaps(data):
    titles = {
        "vision_ffn": "Vision Tower FFN (fc1) Importance",
        "vision_attn": "Vision Tower Attention (Head) Importance",
        "llm_ffn": "Language Model FFN (gate) Importance",
        "llm_attn": "Language Model Attention (GQA) Importance"
    }
    
    for key, title in titles.items():
        layers = sorted(data[key].keys(), key=lambda x: int(x.split('.')[-1]))
        matrix = np.array([data[key][l] for l in layers])
        
        plt.figure(figsize=(15, 6))
        sns.heatmap(matrix, cmap="YlGnBu", robust=True, vmax=np.percentile(matrix, 98), rasterized=True)
        # plt.title(f"Pi0.5: {title}")
        plt.xlabel("Channel / Head Index")
        plt.ylabel("Layer Index")
        # plt.savefig(OUTPUT_DIR / f"heatmap_{key}.png", bbox_inches='tight', dpi=300)
        save_path = OUTPUT_DIR / f"heatmap_{key}.pdf"
        plt.savefig(save_path, bbox_inches='tight') # PDF 不需要指定 dpi
        plt.close()
    print(f"📊 热力图已保存至 {OUTPUT_DIR}")
'''


from matplotlib.ticker import MaxNLocator

def plot_pi05_heatmaps(data):
    LABEL_SIZE = 18 
    TICK_SIZE = 14  
    CBAR_SIZE = 14  

    keys = ["vision_ffn", "vision_attn", "llm_ffn", "llm_attn"]
    
    for key in keys:
        if key not in data or not data[key]: continue
        
        layers = sorted(data[key].keys(), key=lambda x: int(x.split('.')[-1]))
        matrix = np.array([data[key][l] for l in layers])
        
        plt.figure(figsize=(12, 5))
        
        # 1. 绘图：设置 xticklabels=False 让 Seaborn 不要自动生成密集的刻度
        ax = sns.heatmap(
            matrix, 
            cmap="YlGnBu", 
            robust=True, 
            vmax=np.percentile(matrix, 98), 
            rasterized=True,
            xticklabels=False, # 暂时关闭，由我们手动精准控制
            cbar_kws={'shrink': 0.8}
        )
        
        plt.xlabel("Channel / Head Index", fontsize=LABEL_SIZE, labelpad=10)
        plt.ylabel("Layer Index", fontsize=LABEL_SIZE, labelpad=10)

        # 2. 纵轴刻度控制 (每 5 层显示一个)
        y_step = 5
        y_indices = np.arange(0, len(layers), y_step)
        plt.yticks(y_indices + 0.5, [int(layers[i].split('.')[-1]) for i in y_indices], 
                   rotation=0, fontsize=TICK_SIZE)
        
        # 3. 横轴刻度控制 (重点改进)
        num_channels = matrix.shape[1]
        
        # 自动计算步长：目标是显示 5-6 个刻度
        # 使用 MaxNLocator 自动寻找如 2000, 4000 这样“漂亮”的整分位
        locator = MaxNLocator(nbins=5, integer=True)
        x_ticks = locator.tick_values(0, num_channels)
        
        # 过滤掉超出边界的刻度
        x_ticks = [t for t in x_ticks if t < num_channels]
        
        # 设置刻度位置和标签
        # 如果维度很大，可以将标签格式化为 '2k', '4k' 等，或者保持原样
        plt.xticks(np.array(x_ticks) + 0.5, [f"{int(t)}" for t in x_ticks], 
                   rotation=0, fontsize=TICK_SIZE)

        # 4. 颜色条字号
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=CBAR_SIZE)
        
        save_path = OUTPUT_DIR / f"heatmap_{key}.pdf"
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
    print(f"📊 改进后的 PDF (已解决 FFN 刻度重叠) 已保存至 {OUTPUT_DIR}")




# ==========================================
# 执行实验循环
# ==========================================
if __name__ == "__main__":
    # 1. 计算所有组件的重要性得分
    scores = process_pi05_importance()
    
    # 2. 生成所有组件的热力图
    plot_pi05_heatmaps(scores)
    
    # 3. 自动化实验循环：定义你想测试的比例和策略
    # test_ratios = [0.1, 0.2, 0.5]
    test_ratios = [0.1]
    test_strategies = [True, False] # True=Smallest, False=Biggest
    # target_components = ["llm_ffn", "llm_attn", "vision_ffn", "vision_attn"]
    target_components = ["llm_ffn", "llm_attn", "vision_ffn", "vision_attn"]

    print("\n" + "="*50 + "\n🧪 开始生成多参数剪枝实验掩码...\n" + "="*50)

    for comp in target_components:
        for r in test_ratios:
            for strat in test_strategies:
                generate_pruning_masks(
                    scores[comp], 
                    ratio=r, 
                    by_smallest=strat, 
                    component=comp
                )

    print("\n" + "="*50 + f"\n🎉 分析与实验生成全部完成！\n结果目录: {OUTPUT_DIR.absolute()}")