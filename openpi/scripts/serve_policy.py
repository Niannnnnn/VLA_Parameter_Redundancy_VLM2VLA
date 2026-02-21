import dataclasses
import enum
import logging
import socket
import torch
import numpy as np
from pathlib import Path
import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config

# ==========================================================
# 剪枝应用核心逻辑
# ==========================================================
def apply_pruning_mask(model: torch.nn.Module, mask_path: str):
    logging.info(f"正在从 {mask_path} 加载剪枝掩码...")
    mask_data = torch.load(mask_path, weights_only=False)
    
    metadata = mask_data["metadata"]
    component = metadata["component"]
    layers_mask = mask_data["layers"]
    
    logging.info(f"检测到组件: {component}, 策略: {metadata['strategy']}, 比例: {metadata['ratio']}")
    prefix = "paligemma_with_expert.paligemma.model"
    
    total_zeros = 0
    total_non_zeros = 0

    with torch.no_grad():
        for layer_key, info in layers_mask.items():
            # 获取层索引，例如 "layer.2" -> "2"
            idx = layer_key.split('.')[-1]
            mask = torch.from_numpy(info["mask"]) # 形状通常为 (num_heads,)
            
            # --- 处理 Vision Attention (Head-level Pruning) ---
            if component == "vision_attn":
                # 定义该层所有相关的权重矩阵
                # 注意：ViT 通常会对 q,k,v,out 的输出/输入维度按 head 切分
                attn_prefix = f"{prefix}.vision_tower.vision_model.encoder.layers.{idx}.self_attn"
                # 我们同时剪枝 q, k, v 的输出以及 out_proj 的输入
                target_weights = [
                    (f"{attn_prefix}.q_proj.weight", 0), # 剪行 (output dim)
                    (f"{attn_prefix}.k_proj.weight", 0),
                    (f"{attn_prefix}.v_proj.weight", 0),
                    (f"{attn_prefix}.out_proj.weight", 1), # 剪列 (input dim)
                ]

                num_heads = mask.shape[0]
                
                for p_name, dim in target_weights:
                    found = False
                    for name, param in model.named_parameters():
                        if name == p_name:
                            # 计算每个 head 的维度 (e.g., 1152 / 16 = 72)
                            head_dim = param.shape[dim] // num_heads
                            
                            # 将参数重塑以方便对 head 进行操作
                            # 如果 dim=0, shape -> (num_heads, head_dim, -1)
                            # 如果 dim=1, shape -> (-1, num_heads, head_dim)
                            if dim == 0:
                                view_shape = (num_heads, head_dim, -1)
                                param_view = param.data.view(*view_shape)
                                for h in range(num_heads):
                                    if not mask[h]:
                                        param_view[h] = 0.0
                            else:
                                # 对于 out_proj.weight (dim=1), 需要交换维度处理
                                view_shape = (-1, num_heads, head_dim)
                                param_view = param.data.view(*view_shape)
                                for h in range(num_heads):
                                    if not mask[h]:
                                        param_view[:, h] = 0.0
                            
                            found = True
                            break
                    
                    if not found:
                        logging.warning(f"未能在模型中找到参数: {p_name}")

                # 统计当前层的 Head 状态
                pruned_heads = torch.sum(~mask).item()
                kept_heads = torch.sum(mask).item()
                logging.info(f"  [统计] Vision Layer {idx}: 已剪枝 {pruned_heads} Heads | 保留 {kept_heads} Heads")
                total_zeros += pruned_heads
                total_non_zeros += kept_heads
                continue # 进入下一层循环

            # --- 处理其他组件 (Channel-level Pruning) ---
            target_params = []
            if component == "llm_ffn":
                target_params.append(f"{prefix}.language_model.layers.{idx}.mlp.gate_proj.weight")
                target_params.append(f"{prefix}.language_model.layers.{idx}.mlp.up_proj.weight")
                target_params.append((f"{prefix}.language_model.layers.{idx}.mlp.down_proj.weight", 1))
            
            # --- 修复后的 LLM Attn 逻辑 (针对 GQA) ---
            # --- 修复后的 LLM Attn 逻辑 (针对 GQA 并包含统计) ---
            elif component == "llm_attn":
                # Pi0.5 (Gemma-2B): 8 Q-heads, 1 KV-head
                q_name = f"{prefix}.language_model.layers.{idx}.self_attn.q_proj.weight"
                o_name = f"{prefix}.language_model.layers.{idx}.self_attn.o_proj.weight"
                
                # 记录本层剪枝数量用于最终统计
                pruned_this_layer = torch.sum(~mask).item()
                kept_this_layer = torch.sum(mask).item()

                # 1. 处理 Q 投影 (剪行)
                for name, param in model.named_parameters():
                    if name == q_name:
                        num_q_heads = mask.shape[0] # 应为 8
                        head_dim = param.shape[0] // num_q_heads
                        param_view = param.data.view(num_q_heads, head_dim, -1)
                        for h in range(num_q_heads):
                            if not mask[h]: 
                                param_view[h] = 0.0
                
                # 2. 处理 O 投影 (对应剪列)
                for name, param in model.named_parameters():
                    if name == o_name:
                        num_q_heads = mask.shape[0] # 应为 8
                        head_dim = param.shape[1] // num_q_heads
                        param_view = param.data.view(-1, num_q_heads, head_dim)
                        for h in range(num_q_heads):
                            if not mask[h]: 
                                param_view[:, h] = 0.0
                
                # 更新全局统计变量
                total_zeros += pruned_this_layer
                total_non_zeros += kept_this_layer

                logging.info(f"  [LLM Attn] Layer {idx} Q-heads pruned: {pruned_this_layer} | Kept: {kept_this_layer}")
                
                # KV 逻辑：GQA 模式下 KV 通常不随 Q 剪枝，除非整个 Group 消失
                continue

            elif component == "vision_ffn":
                target_params.append(f"{prefix}.vision_tower.vision_model.encoder.layers.{idx}.mlp.fc1.weight")
                target_params.append((f"{prefix}.vision_tower.vision_model.encoder.layers.{idx}.mlp.fc2.weight", 1))

            # 执行常规的 Channel-level 统计和应用
            for p_item in target_params:
                dim = 0
                if isinstance(p_item, tuple): p_name, dim = p_item
                else: p_name = p_item
                
                found = False
                for name, param in model.named_parameters():
                    if name == p_name:
                        m = mask.to(param.device)
                        if m.shape[0] < param.shape[dim]:
                            num_per_head = param.shape[dim] // m.shape[0]
                            m = m.repeat_interleave(num_per_head)

                        if dim == 0:
                            param.data.mul_(m.view(-1, *([1] * (param.ndim - 1))))
                        else:
                            param.data.mul_(m.view(1, -1, *([1] * (param.ndim - 2))))
                        
                        channel_norms = torch.norm(param.data.float(), p=2, dim=1 if dim==0 else 0)
                        num_zero_channels = torch.sum(channel_norms == 0).item()
                        num_active_channels = torch.sum(channel_norms > 0).item()
                        
                        logging.info(f"  [统计] 参数: {p_name.split('.')[-2:]} | 置零通道: {num_zero_channels}")
                        total_zeros += num_zero_channels
                        total_non_zeros += num_active_channels
                        found = True; break

    logging.info(f"✅ 掩码应用完成。总计已置零单元: {total_zeros}, 总计活跃单元: {total_non_zeros}")
    
# ==========================================================
# 原有代码逻辑增强
# ==========================================================

class EnvMode(enum.Enum):
    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"

@dataclasses.dataclass
class Checkpoint:
    config: str
    dir: str

@dataclasses.dataclass
class Default:
    pass

@dataclasses.dataclass
class Args:
    env: EnvMode = EnvMode.LIBERO
    default_prompt: str | None = None
    port: int = 8000
    record: bool = False
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)
    
    # 新增掩码路径参数
    mask_path: str | None = None 

DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(config="pi05_aloha", dir="gs://openpi-assets/checkpoints/pi05_base"),
    EnvMode.ALOHA_SIM: Checkpoint(config="pi0_aloha_sim", dir="gs://openpi-assets/checkpoints/pi0_aloha_sim"),
    EnvMode.DROID: Checkpoint(config="pi05_droid", dir="gs://openpi-assets/checkpoints/pi05_droid"),
    EnvMode.LIBERO: Checkpoint(config="pi05_libero", dir="gs://openpi-assets/checkpoints/pi05_libero"),
}

def create_policy(args: Args) -> _policy.Policy:
    # 1. 确定配置和目录
    # 当使用 policy:checkpoint 时，args.policy 会是一个 Checkpoint 实例
    if isinstance(args.policy, Checkpoint):
        config_obj = _config.get_config(args.policy.config)
        checkpoint_dir = args.policy.dir
    else:
        # 当使用默认模式时
        cp = DEFAULT_CHECKPOINT.get(args.env)
        config_obj = _config.get_config(cp.config)
        checkpoint_dir = cp.dir

    # 2. 创建策略
    policy = _policy_config.create_trained_policy(
        config_obj, checkpoint_dir, default_prompt=args.default_prompt
    )

    # 3. 应用掩码
    if args.mask_path:
        inner_model = getattr(policy, "_model", None)
        if inner_model:
            apply_pruning_mask(inner_model, args.mask_path)
            
    return policy

def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata

    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))