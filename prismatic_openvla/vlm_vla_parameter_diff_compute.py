import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import re
import pandas as pd
import argparse
import seaborn as sns
from utils import * 

def evaluate_vla_importance(mapping_list, model_orig, model_finetuned, output_dir, 
                            component="llm", target_module="ffn", attn_mode="channel",
                            metric_type="relative", compute_device="cuda:0", save_detail=False):
    """
    评估通道/头部重要性。
    修复了 LLM FFN 维度对应错误（应该是 11008 维度）和 Key 前缀丢失的问题。
    """
    print(f"🔍 评估组件: [{component}] | 目标: [{target_module}] | 模式: [{attn_mode}] | 指标: [{metric_type}] ...")
    device = torch.device(compute_device)
    plm_state = model_orig.state_dict()
    olm_state = model_finetuned.state_dict()
    importance_scores = {}

    # 1. 定义关键字过滤
    if component == "llm":
        target_keywords = ["gate_proj", "up_proj", "down_proj"] if target_module == "ffn" else ["q_proj"]
    elif "vision" in component:
        target_keywords = ["mlp.fc1"] if target_module == "ffn" else ["attn.qkv"]
    elif component == "projector":
        target_keywords = ["fc1", "fc2", "projector.0", "projector.2"] 
    else:
        target_keywords = ["weight"]

    # 用于存放 LLM FFN 每一层各子模块的得分
    ffn_accumulator = defaultdict(dict)

    for (plm_name, olm_name) in mapping_list:
        if "# NOT FOUND" in olm_name: continue
        if "bias" in olm_name: continue 
        if not any(x in olm_name for x in target_keywords): continue

        # 加载权重
        W_orig = plm_state[plm_name].to(device).float()
        W_fine = olm_state[olm_name].to(device).float()
        delta_W = W_fine - W_orig
        
        # --- 核心修复 1: 确定计算维度的方向 ---
        if component == "llm":
            if target_module == "ffn":
                # gate_proj (11008, 4096) -> 剪输出 (dim 0)，计算 dim 1
                # up_proj   (11008, 4096) -> 剪输出 (dim 0)，计算 dim 1
                # down_proj (4096, 11008) -> 剪输入 (dim 1)，计算 dim 0
                calc_dim = 0 if "down_proj" in olm_name else 1 
            else:
                # Attention: q_proj (4096, 4096) -> 剪输出 (dim 0)，计算 dim 1
                calc_dim = 1 if "o_proj" in olm_name else 1
        elif "vision" in component or component == "projector":
            calc_dim = 1 
        else:
            calc_dim = 0 
        
        # 3. 计算基础得分
        # 这里的 calc_dim 是“被聚合掉的维度”，保留下来的就是重要性向量
        # 例如 gate_proj (11008, 4096) 对 dim 1 求 norm，得到 [11008]
        diff_norm = torch.norm(delta_W, p=2, dim=calc_dim)
        if metric_type == "relative":
            orig_norm = torch.norm(W_orig, p=2, dim=calc_dim) + 1e-8
            score = diff_norm / orig_norm
        else:
            score = diff_norm

        # --- 特殊处理：LLM FFN 聚合逻辑 ---
        if component == "llm" and target_module == "ffn":
            match = re.search(r'layers\.(\d+)\.', olm_name)
            if match:
                l_idx = int(match.group(1))
                sub_module_name = next(x for x in ["gate_proj", "up_proj", "down_proj"] if x in olm_name)
                ffn_accumulator[l_idx][sub_module_name] = score.detach().cpu()
            continue 
        
        # --- 其他模块（Vision/Projector/LLM Attn） ---
        if target_module == "attention" and attn_mode == "head":
            hidden_size = W_orig.shape[1]
            num_heads = 16 if hidden_size in [1024, 1152] else 32
            head_dim = hidden_size // num_heads
            if "qkv" in olm_name:
                score = score.view(3, num_heads, head_dim).mean(dim=[0, 2])
            else:
                score = score.view(num_heads, head_dim).mean(dim=1)

        importance_scores[olm_name] = score.detach().cpu()

    # --- 核心修复 2: 聚合与 Key 前缀保留 ---
    if component == "llm" and target_module == "ffn":
        print(f"合并 LLM FFN 子模块得分 (Gate + Up + Down)...")
        
        # 建立索引，确保获取带 'module.language_model.' 等完整前缀的 gate_proj 名称
        full_name_map = {}
        for _, olm_name in mapping_list:
            if "gate_proj" in olm_name:
                match = re.search(r'layers\.(\d+)\.', olm_name)
                if match:
                    full_name_map[int(match.group(1))] = olm_name

        for l_idx, sub_scores in ffn_accumulator.items():
            if not sub_scores: continue
            
            # 聚合三个投影层在 11008 维度上的得分
            combined_score = sum(sub_scores.values())
            
            if l_idx in full_name_map:
                rep_name = full_name_map[l_idx]
            else:
                # 最后的兜底
                rep_name = f"module.language_model.model.layers.{l_idx}.mlp.gate_proj.weight"
            
            importance_scores[rep_name] = combined_score

    # 5. 保存统计数据
    os.makedirs(output_dir, exist_ok=True)
    stats_path = os.path.join(output_dir, f"importance_stats_{component}_{target_module}_{metric_type}.csv")
    records = []
    for k, v in importance_scores.items():
        records.append({
            "Layer_Name": k, 
            "Min": v.min().item(), 
            "Max": v.max().item(), 
            "Mean": v.mean().item(), 
            "Count": len(v)
        })
    pd.DataFrame(records).to_csv(stats_path, index=False)
    
    return importance_scores

def generate_pruning_masks(delta_data, output_dir, ratio=0.2, by_smallest=True, 
                            component="llm", target_module="ffn", attn_mode="channel", metric_type="relative"):
    if not delta_data: return {}
    
    # ==========================================================
    # 局部配置：层保护开关
    # ==========================================================
    is_protected = False  # 手动修改此处：True 开启保护，False 关闭保护
    # protected_layers = [0, 1, 2, 31] # LLM 需要保护的层号
    protected_layers = [0, 1, 2, 3, 4, 30, 31] # LLM 需要保护的层号
    # protected_layers = [0, 1, 2, 3, 4, 5, 29, 30, 31] # LLM 需要保护的层号
    # ==========================================================

    pruning_masks = {}
    layer_scores = defaultdict(list)
    layer_to_keys = defaultdict(list)
    
    # 1. 解析层号
    for k, v in delta_data.items():
        if component == "projector":
            if "fc1" in k or "projector.0" in k: l_idx = 1
            elif "fc2" in k or "projector.2" in k: l_idx = 2
            elif "fc3" in k or "projector.4" in k: l_idx = 3
            else: l_idx = 0
        else:
            match = re.search(r'(layers|blocks)\.(\d+)\.', k)
            l_idx = int(match.group(2)) if match else 0
        
        # 保护逻辑：如果是 LLM 组件且开启了保护，将指定层的得分设为无穷大（或极小值）
        # 这样在全局排序时，这些单元永远不会被选中剪枝
        v_to_sort = v.clone()
        if is_protected and component == "llm" and l_idx in protected_layers:
            if by_smallest:
                # 如果是删掉得分最小的，我们就把保护层得分设为极大
                v_to_sort = torch.full_like(v, float('inf'))
            else:
                # 如果是删掉得分最大的，我们就把保护层得分设为极小
                v_to_sort = torch.full_like(v, float('-inf'))
        
        layer_scores[l_idx].append(v_to_sort)
        layer_to_keys[l_idx].append(k)

    # 2. 全局排序
    sorted_layer_ids = sorted(layer_scores.keys())
    all_scores_list = [torch.cat(layer_scores[i]) for i in sorted_layer_ids]
    flat_scores = torch.cat(all_scores_list)
    num_total = len(flat_scores)
    num_to_prune = int(num_total * ratio)
    
    strategy_str = 'smallest' if by_smallest else 'biggest'
    print(f"\n" + "="*60)
    print(f"📊 剪枝任务详情 [{component.upper()}]: 总单元数 = {num_total} | 策略 = {strategy_str} | 层保护 = {is_protected}")
    if is_protected and component == "llm":
        print(f"🛡️ 已保护 LLM 层: {protected_layers}")
    print("="*60)

    _, sorted_indices = torch.sort(flat_scores, descending=False)
    prune_indices = sorted_indices[:num_to_prune] if by_smallest else sorted_indices[-num_to_prune:]
    global_mask_bool = torch.ones(num_total, dtype=torch.bool)
    global_mask_bool[prune_indices] = False

    # 3. 拆解并分发 (逻辑保持不变)
    current_pos = 0
    for idx in sorted_layer_ids:
        n_elements = len(torch.cat(layer_scores[idx]))
        layer_mask = global_mask_bool[current_pos : current_pos + n_elements].numpy()
        current_pos += n_elements
        
        pruned_count_val = int(np.sum(~layer_mask))
        # 打印信息辅助验证
        prot_tag = " [PROTECTED]" if is_protected and component == "llm" and idx in protected_layers else ""
        print(f"Layer {idx:<2} | Pruned: {pruned_count_val:<5} | Total: {n_elements:<5}{prot_tag}")

        for k in layer_to_keys[idx]:
            k_clean = k.replace("module.", "")
            base_meta = {
                'layer_num': idx,
                'pruned_count': pruned_count_val,
                'total_count': n_elements,
                'target_module': target_module,
                'attn_mode': attn_mode,
                'component': component
            }

            # --- 以下是您原有的分发逻辑 (LLM, Vision, Projector) ---
            if component == "llm":
                if target_module == "ffn":
                    for suffix in ["gate_proj", "up_proj", "down_proj"]:
                        target_k = k_clean.replace("gate_proj", suffix)
                        is_down = "down_proj" in target_k
                        pruning_masks[target_k] = {
                            **base_meta,
                            'output_mask': None if is_down else layer_mask,
                            'input_mask': layer_mask if is_down else None,
                        }
                else:
                    h_dim = 128 
                    actual_mask = np.repeat(layer_mask, h_dim)
                    for suffix in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                        target_k = k_clean.replace("q_proj", suffix)
                        is_o = "o_proj" in target_k
                        pruning_masks[target_k] = {
                            **base_meta,
                            'total_count': len(actual_mask),
                            'pruned_count': int(np.sum(~actual_mask)),
                            'output_mask': None if is_o else actual_mask,
                            'input_mask': actual_mask if is_o else None,
                        }
            elif "vision" in component and target_module == "ffn":
                pruning_masks[k_clean] = {**base_meta, 'output_mask': layer_mask, 'input_mask': None}
                fc2_key = k_clean.replace("fc1", "fc2")
                pruning_masks[fc2_key] = {**base_meta, 'output_mask': None, 'input_mask': layer_mask}
            elif component == "projector":
                pruning_masks[k_clean] = {**base_meta, 'output_mask': layer_mask, 'input_mask': None}
                if "fc" in k_clean: next_k = k_clean.replace(f"fc{idx}", f"fc{idx+1}")
                else: next_k = k_clean.replace(f"projector.{(idx-1)*2}", f"projector.{(idx)*2}")
                if next_k not in pruning_masks:
                    pruning_masks[next_k] = {**base_meta, 'layer_num': idx+1, 'output_mask': None, 'input_mask': layer_mask}
                else:
                    pruning_masks[next_k]['input_mask'] = layer_mask
            else:
                is_input_side = any(x in k for x in ["attn.proj", "proj.weight"])
                actual_mask = layer_mask
                if target_module == "attention" and attn_mode == "head":
                    h_dim = 72 if "siglip" in component else (64 if "dino" in component else 128)
                    actual_mask = np.repeat(layer_mask, h_dim)
                pruning_masks[k_clean] = {
                    **base_meta,
                    'total_count': len(actual_mask),
                    'pruned_count': int(np.sum(~actual_mask)),
                    'output_mask': None if is_input_side else actual_mask,
                    'input_mask': actual_mask if is_input_side else None,
                }

    # 4. 保存
    prot_suffix = f"_protected_L{'_L'.join(map(str, protected_layers))}" if (is_protected and component == "llm") else ""
    save_name = f"masks_{component}_{target_module}_{attn_mode}_{ratio}_{strategy_str}_{metric_type}{prot_suffix}.pth"
    torch.save(pruning_masks, os.path.join(output_dir, save_name))
    print(f"💾 掩码文件已保存至: {os.path.join(output_dir, save_name)}\n" + "="*60 + "\n")
    return pruning_masks

# def plot_importance_heatmap(delta_data, output_dir, component="llm", target_module="ffn", attn_mode="channel", metric_type="relative"):
#     if not delta_data: return
#     print(f"📊 正在生成 {component} 热力图...")
    
#     layer_map = defaultdict(list)
#     for k, v in delta_data.items():
#         if component == "projector":
#             idx = 1 if ("fc1" in k or "projector.0" in k) else (2 if ("fc2" in k or "projector.2" in k) else 3)
#         else:
#             match = re.search(r'(layers|blocks)\.(\d+)\.', k)
#             idx = int(match.group(2)) if match else 0
#         layer_map[idx].append(v)
    
#     sorted_idx = sorted(layer_map.keys())
#     max_len = max([len(torch.cat(layer_map[i])) for i in sorted_idx])
    
#     processed_rows = []
#     for i in sorted_idx:
#         row_data = torch.cat(layer_map[i]).numpy()
#         if len(row_data) < max_len:
#             padded_row = np.full(max_len, np.nan)
#             padded_row[:len(row_data)] = row_data
#             processed_rows.append(padded_row)
#         else:
#             processed_rows.append(row_data)
    
#     heatmap_matrix = np.array(processed_rows)
#     plt.figure(figsize=(15, 6))
#     sns.heatmap(heatmap_matrix, cmap="YlGnBu", robust=True, 
#                 vmax=np.percentile(heatmap_matrix[~np.isnan(heatmap_matrix)], 98),
#                 mask=np.isnan(heatmap_matrix), rasterized=True)
    
#     # plt.title(f"{component.upper()} {target_module} Importance ({metric_type})")
#     plt.yticks(np.arange(len(sorted_idx)) + 0.5, sorted_idx, rotation=0)
#     # save_fig = f"heatmap_{component}_{target_module}_{attn_mode}_{metric_type}.png"
#     save_fig = f"heatmap_{component}_{target_module}_{attn_mode}.pdf"
#     # plt.savefig(os.path.join(output_dir, save_fig), bbox_inches='tight', dpi=300)
#     plt.savefig(os.path.join(output_dir, save_fig), bbox_inches='tight')
#     plt.close()




from matplotlib.ticker import MaxNLocator

def plot_importance_heatmap(delta_data, output_dir, component="llm", target_module="ffn", attn_mode="channel", metric_type="relative"):
    if not delta_data: return
    print(f"📊 正在生成 {component} 热力图 (论文格式)...")
    
    # --- 样式常量配置 ---
    LABEL_SIZE = 18    # 轴标题字号
    TICK_SIZE = 14     # 刻度数字字号
    CBAR_SIZE = 14     # 颜色条字号
    
    # 1. 数据对齐与预处理 (保持你的原有逻辑)
    layer_map = defaultdict(list)
    for k, v in delta_data.items():
        if component == "projector":
            idx = 1 if ("fc1" in k or "projector.0" in k) else (2 if ("fc2" in k or "projector.2" in k) else 3)
        else:
            match = re.search(r'(layers|blocks)\.(\d+)\.', k)
            idx = int(match.group(2)) if match else 0
        layer_map[idx].append(v)
    
    sorted_idx = sorted(layer_map.keys())
    max_len = max([len(torch.cat(layer_map[i])) for i in sorted_idx])
    
    processed_rows = []
    for i in sorted_idx:
        row_data = torch.cat(layer_map[i]).numpy()
        if len(row_data) < max_len:
            padded_row = np.full(max_len, np.nan)
            padded_row[:len(row_data)] = row_data
            processed_rows.append(padded_row)
        else:
            processed_rows.append(row_data)
    
    # 2. 绘图核心部分
    heatmap_matrix = np.array(processed_rows)
    plt.figure(figsize=(12, 5)) # 略微缩小宽度，提高文字在 PDF 中的相对占比
    
    # 使用 xticklabels=False 避免 Seaborn 尝试渲染上万个标签
    ax = sns.heatmap(
        heatmap_matrix, 
        cmap="YlGnBu", 
        robust=True, 
        vmax=np.percentile(heatmap_matrix[~np.isnan(heatmap_matrix)], 98),
        mask=np.isnan(heatmap_matrix), 
        rasterized=True,
        xticklabels=False, 
        cbar_kws={'shrink': 0.8}
    )
    
    # 3. 坐标轴与标签美化
    plt.xlabel("Channel / Head Index", fontsize=LABEL_SIZE, labelpad=10)
    plt.ylabel("Layer Index", fontsize=LABEL_SIZE, labelpad=10)

    # 纵轴：层号刻度
    # 如果层数较多（如 32 层），每 5 层显示一个；如果很少（如 Projector），则全部显示
    y_step = 5 if len(sorted_idx) > 10 else 1
    y_indices = np.arange(0, len(sorted_idx), y_step)
    plt.yticks(y_indices + 0.5, [sorted_idx[i] for i in y_indices], 
               rotation=0, fontsize=TICK_SIZE)

    # 横轴：使用 MaxNLocator 自动控制 5-6 个刻度，彻底解决 FFN 万级通道重叠问题
    num_channels = heatmap_matrix.shape[1]
    locator = MaxNLocator(nbins=5, integer=True)
    x_ticks = locator.tick_values(0, num_channels)
    x_ticks = [t for t in x_ticks if t < num_channels] # 过滤掉越界的刻度
    
    plt.xticks(np.array(x_ticks) + 0.5, [f"{int(t)}" for t in x_ticks], 
               rotation=0, fontsize=TICK_SIZE)

    # 4. 颜色条字号调整
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=CBAR_SIZE)
    
    # 5. 保存
    save_fig = f"heatmap_{component}_{target_module}_{attn_mode}.pdf"
    plt.savefig(os.path.join(output_dir, save_fig), bbox_inches='tight')
    plt.close()
    print(f"✅ 已成功保存 PDF 热力图: {save_fig}")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--component', type=str, default='vision_siglip', choices=['llm', 'vision_dino', 'vision_siglip', 'projector'])
    parser.add_argument('--target-module', type=str, default='attention', choices=['ffn', 'attention'])
    parser.add_argument('--attn-mode', type=str, default='head', choices=['channel', 'head'])
    parser.add_argument('--ratio', type=float, default=0.2)
    parser.add_argument('--by-biggest', action='store_false', dest='by_smallest')
    parser.add_argument('--metric-type', type=str, default='relative', choices=['absolute', 'relative'])
    parser.add_argument('--output-dir', type=str, default='./analysis_results')
    parser.add_argument('--compute-gpu', type=int, default=0)
    parser.add_argument('--orig-gpu', type=int, default=1)
    parser.add_argument('--fine-gpu', type=int, default=2)
    parser.set_defaults(by_smallest=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    model_orig = load_prismatic_vlm(args.orig_gpu)
    model_finetuned = load_openvla(args.fine_gpu)
    
    mapping_list = match_parameters(model_orig, model_finetuned, args.output_dir, component=args.component)
    
    importance_data = evaluate_vla_importance(
        mapping_list, model_orig, model_finetuned, args.output_dir,
        component=args.component, target_module=args.target_module,
        attn_mode=args.attn_mode, metric_type=args.metric_type,
        compute_device=f"cuda:{args.compute_gpu}"
    )

    plot_importance_heatmap(importance_data, args.output_dir, args.component, args.target_module, args.attn_mode, args.metric_type)

    generate_pruning_masks(
        importance_data, args.output_dir, ratio=args.ratio, by_smallest=args.by_smallest, 
        component=args.component, target_module=args.target_module,
        attn_mode=args.attn_mode, metric_type=args.metric_type
    )

if __name__ == "__main__":
    main()