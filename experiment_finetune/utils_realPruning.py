import torch
import torch.nn as nn
import copy
import os
import re
from transformers import AutoModelForVision2Seq, AutoProcessor

def load_pretrained_model():
    checkpoint = "openvla/openvla-7b-finetuned-libero-spatial"
    # checkpoint = "openvla/openvla-7b"
    model = AutoModelForVision2Seq.from_pretrained(
        checkpoint,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="auto",
        load_in_8bit=False, 
        load_in_4bit=False,
    )
    processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
    return model, processor


def prune_linear_layer(layer: nn.Linear, dim: int, index: torch.Tensor):
    """根据通道索引裁剪Linear层"""
    W = layer.weight.data.clone()
    b = layer.bias.data.clone() if layer.bias is not None else None

    if dim == 0:
        W = W[index, :]
        if b is not None:
            b = b[index]
    elif dim == 1:
        W = W[:, index]
    else:
        raise ValueError("dim must be 0 or 1")

    new_layer = nn.Linear(W.shape[1], W.shape[0], bias=(b is not None), dtype=W.dtype)
    new_layer.weight.data = W.clone()
    if b is not None:
        new_layer.bias.data = b.clone()

    return new_layer


def build_pruned_model(original_model, pruning_mask):
    """在不改动原模型的情况下，返回一个剪枝后新模型"""
    # 深拷贝一个新模型（结构与参数独立）
    pruned_model = copy.deepcopy(original_model)

    for i, layer in enumerate(pruned_model.language_model.model.layers):
        ffn = layer.mlp
        up_key = f"language_model.model.layers.{i}.mlp.up_proj.weight"
        gate_key = f"language_model.model.layers.{i}.mlp.gate_proj.weight"
        down_key = f"language_model.model.layers.{i}.mlp.down_proj.weight"

        if up_key not in pruning_mask:
            print(f"Layer {i}: no pruning info, skip.")
            continue

        # 获取up_proj的mask信息
        up_info = pruning_mask[up_key]
        gate_info = pruning_mask[gate_key]
        down_info = pruning_mask[down_key]
        
        # 检查output_mask是否为None
        if (up_info["output_mask"] is None or 
            gate_info["output_mask"] is None or 
            down_info["input_mask"] is None):
            print(f"Layer {i}: one or more masks are None (up_output: {up_info['output_mask'] }, "
                  f"gate_output: {gate_info['output_mask'] }, "
                  f"down_input: {down_info['input_mask'] }), skip pruning this layer.")
            continue

        up_mask = torch.tensor(pruning_mask[up_key]["output_mask"]).bool()
        gate_mask = torch.tensor(pruning_mask[gate_key]["output_mask"]).bool()
        down_mask = torch.tensor(pruning_mask[down_key]["input_mask"]).bool()
        # 验证一致性
        assert up_mask.shape == gate_mask.shape == down_mask.shape, \
            f"mask shape mismatch in layer {i}"

        kept_idx = torch.where(up_mask)[0]
        print(f"Layer {i}: keep : {len(kept_idx)} channels, total : {len(up_mask)} channels, prune : {len(up_mask)-len(kept_idx)} channels")

        # 替换新Linear
        ffn.up_proj = prune_linear_layer(ffn.up_proj, dim=0, index=kept_idx)
        ffn.gate_proj = prune_linear_layer(ffn.gate_proj, dim=0, index=kept_idx)
        ffn.down_proj = prune_linear_layer(ffn.down_proj, dim=1, index=kept_idx)

    return pruned_model


if __name__ == "__main__":
    model, processor = load_pretrained_model()
    pruning_mask_path = (
        "path/to/your/pruning_mask.pt"  # 替换为实际的 pruning_mask.pt 路径
    )
    pruning_mask = torch.load(pruning_mask_path)

    # 构建新模型（不改原模型）
    pruned_model = build_pruned_model(model, pruning_mask)

    # 保存为 HuggingFace 可加载的模型目录
    base = os.path.splitext(os.path.basename(pruning_mask_path))[0]
    match = re.search(r"(ratio_[0-9.]+)", base)

    ratio_tag = match.group(1) if match else "pruned"
    save_dir = os.path.join(os.getcwd(), f"openvla_pruned_model_only_deltaW_smallest_{ratio_tag}")
    os.makedirs(save_dir, exist_ok=True)  
    pruned_model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)

    print(f"pruned model saved to: {save_dir}")
