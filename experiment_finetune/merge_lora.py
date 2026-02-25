import os
import torch
import json
from pathlib import Path
from peft import PeftModel
from transformers import AutoConfig, AutoProcessor, AutoModelForVision2Seq
from transformers.models.llama.modeling_llama import LlamaModel, LlamaDecoderLayer, LlamaRMSNorm
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

# --- 这里粘贴你之前的辅助函数 ---

def load_pruned_model_offline(vla_path, layer_ffn_dims_file):
    """复用你训练代码中的剪枝加载逻辑"""
    model_dir = str(vla_path)
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    pruning_info = torch.load(layer_ffn_dims_file, map_location="cpu")
    
    # 计算每层维度
    num_layers = getattr(config, "num_hidden_layers", 32)
    layer_ffn_dims = []
    for layer_idx in range(num_layers):
        up_proj_key = f'language_model.model.layers.{layer_idx}.mlp.up_proj.weight'
        if up_proj_key in pruning_info:
            info = pruning_info[up_proj_key]
            total_channels = info.get("total_channels", config.text_config.intermediate_size)
            pruned_channels = info.get("pruned_channels", 0)
            layer_ffn_dims.append(int(total_channels - pruned_channels))
        else:
            layer_ffn_dims.append(config.text_config.intermediate_size)

    # Patch 逻辑
    LlamaModel_init_orig = LlamaModel.__init__
    def patched_init(self, config_local):
        super(LlamaModel, self).__init__(config_local)
        orig_intermediate = getattr(config_local, "intermediate_size", None)
        layers_list = []
        for i in range(config_local.num_hidden_layers):
            new_inter = layer_ffn_dims[i] if i < len(layer_ffn_dims) else orig_intermediate
            setattr(config_local, "intermediate_size", new_inter)
            layers_list.append(LlamaDecoderLayer(config_local, i))
        setattr(config_local, "intermediate_size", orig_intermediate)
        self.layers = torch.nn.ModuleList(layers_list)
        self.embed_tokens = torch.nn.Embedding(config_local.vocab_size, config_local.hidden_size)
        self.norm = LlamaRMSNorm(config_local.hidden_size, eps=config_local.rms_norm_eps)

    LlamaModel.__init__ = patched_init
    vla = AutoModelForVision2Seq.from_config(config, trust_remote_code=True)
    
    # 加载权重
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    with open(index_path, "r") as f:
        weight_map = json.load(f).get("weight_map", {})
    
    state_dict = {}
    for shard_file in set(weight_map.values()):
        state_dict.update(load_file(os.path.join(model_dir, shard_file)))
    
    vla.load_state_dict(state_dict, strict=False)
    LlamaModel.__init__ = LlamaModel_init_orig # 还原
    return vla

# --- 主程序逻辑 ---

base_model_path = "/path/to/base_model"
ffn_dims_file = "/path/to/ffn_dims_file"
adapter_path = "/path/to/lora_adapter"

print("Step 1: Loading pruned base model...")
base_model = load_pruned_model_offline(base_model_path, ffn_dims_file)

print("Step 2: Loading LoRA adapter...")
# 加载 LoRA 时，PEFT 会根据 base_model 现在的维度自动匹配 LoRA 矩阵的维度
model = PeftModel.from_pretrained(base_model, adapter_path)

print("Step 3: Merging...")
merged_model = model.merge_and_unload()
merged_model.to(torch.bfloat16) # 确保精度正确，防止变成 float32 导致体积翻倍

# print("Step 4: Saving merged model...")
print("Step 4: Saving merged model...")
save_path = "/path/to/save_merged_model"
os.makedirs(save_path, exist_ok=True)

# 1. 获取合并后的纯净 state_dict
# 这一步会确保我们拿到的只有当前模型的一份权重
merged_state_dict = merged_model.state_dict()
print(f"✅ Merged model has {len(merged_state_dict)} parameters.")
print(merged_state_dict.keys())

# 2. 保存权重文件 (使用 safetensors)
# 建议手动分为几个 shard 或者直接存成单文件（如果内存允许）
from safetensors.torch import save_file
save_file(merged_state_dict, os.path.join(save_path, "model.safetensors"))

# 3. 保存配置文件 (这不会保存权重，只保存 json)
merged_model.config.save_pretrained(save_path)

# 4. 生成 index 文件（让 HF 识别）
index_dict = {
    "metadata": {"total_size": sum(p.numel() * 2 for p in merged_model.parameters())}, # 假设是 bf16
    "weight_map": {k: "model.safetensors" for k in merged_state_dict.keys()}
}
with open(os.path.join(save_path, "model.safetensors.index.json"), "w") as f:
    json.dump(index_dict, f, indent=2)

# 5. 别忘了存一份处理器
processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
processor.save_pretrained(save_path)

print(f"✅ All done! Saved cleaned model to {save_path}")

# 别忘了存一份处理器和配置，方便直接运行
processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
processor.save_pretrained(save_path)

print(f"✅ All done! Saved to {save_path}")