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
# 剪枝应用核心逻辑 (增强版：支持累计统计)
# ==========================================================
def apply_multiple_masks(model: torch.nn.Module, mask_paths: list[str]):
    """循环应用多个掩码文件"""
    overall_zeros = 0
    overall_active = 0
    
    for path in mask_paths:
        logging.info(f"--- 正在处理掩码文件: {path} ---")
        zeros, active = apply_single_mask(model, path)
        overall_zeros += zeros
        overall_active = active # 活跃单元以最后一次状态为准，或根据需求累加
        
    logging.info(f"✨ 所有掩码应用完成。")
    logging.info(f"✅ 最终统计 -> 总计置零单元: {overall_zeros}")

def apply_single_mask(model: torch.nn.Module, mask_path: str) -> tuple[int, int]:
    """应用单个掩码文件并返回该次操作的统计"""
    mask_data = torch.load(mask_path, weights_only=False)
    
    metadata = mask_data["metadata"]
    component = metadata["component"]
    layers_mask = mask_data["layers"]
    
    logging.info(f"检测到组件: {component} | 策略: {metadata['strategy']} | 目标比例: {metadata['ratio']}")
    prefix = "paligemma_with_expert.paligemma.model"
    
    total_zeros = 0
    total_non_zeros = 0

    with torch.no_grad():
        for layer_key, info in layers_mask.items():
            idx = layer_key.split('.')[-1]
            mask = torch.from_numpy(info["mask"]) 
            
            # --- 1. Vision Attention (Head-level) ---
            if component == "vision_attn":
                attn_prefix = f"{prefix}.vision_tower.vision_model.encoder.layers.{idx}.self_attn"
                target_weights = [
                    (f"{attn_prefix}.q_proj.weight", 0),
                    (f"{attn_prefix}.k_proj.weight", 0),
                    (f"{attn_prefix}.v_proj.weight", 0),
                    (f"{attn_prefix}.out_proj.weight", 1),
                ]
                num_heads = mask.shape[0]
                for p_name, dim in target_weights:
                    for name, param in model.named_parameters():
                        if name == p_name:
                            head_dim = param.shape[dim] // num_heads
                            if dim == 0:
                                param_view = param.data.view(num_heads, head_dim, -1)
                                for h in range(num_heads):
                                    if not mask[h]: param_view[h] = 0.0
                            else:
                                param_view = param.data.view(-1, num_heads, head_dim)
                                for h in range(num_heads):
                                    if not mask[h]: param_view[:, h] = 0.0
                
                pruned = torch.sum(~mask).item()
                total_zeros += pruned
                total_non_zeros += torch.sum(mask).item()
                logging.info(f"  [Vision Attn] Layer {idx}: Pruned {pruned} heads")
                continue

            # --- 2. LLM Attention (GQA Head-level) ---
            elif component == "llm_attn":
                q_name = f"{prefix}.language_model.layers.{idx}.self_attn.q_proj.weight"
                o_name = f"{prefix}.language_model.layers.{idx}.self_attn.o_proj.weight"
                num_q_heads = mask.shape[0]
                
                for name, param in model.named_parameters():
                    if name == q_name:
                        head_dim = param.shape[0] // num_q_heads
                        param_view = param.data.view(num_q_heads, head_dim, -1)
                        for h in range(num_q_heads):
                            if not mask[h]: param_view[h] = 0.0
                    if name == o_name:
                        head_dim = param.shape[1] // num_q_heads
                        param_view = param.data.view(-1, num_q_heads, head_dim)
                        for h in range(num_q_heads):
                            if not mask[h]: param_view[:, h] = 0.0
                
                pruned = torch.sum(~mask).item()
                total_zeros += pruned
                total_non_zeros += torch.sum(mask).item()
                logging.info(f"  [LLM Attn] Layer {idx}: Pruned {pruned} Q-heads")
                continue

            # --- 3. FFN Components (Channel-level) ---
            target_params = []
            if component == "llm_ffn":
                target_params.append(f"{prefix}.language_model.layers.{idx}.mlp.gate_proj.weight")
                target_params.append(f"{prefix}.language_model.layers.{idx}.mlp.up_proj.weight")
                target_params.append((f"{prefix}.language_model.layers.{idx}.mlp.down_proj.weight", 1))
            elif component == "vision_ffn":
                target_params.append(f"{prefix}.vision_tower.vision_model.encoder.layers.{idx}.mlp.fc1.weight")
                target_params.append((f"{prefix}.vision_tower.vision_model.encoder.layers.{idx}.mlp.fc2.weight", 1))

            layer_pruned_channels = 0
            for p_item in target_params:
                p_name = p_item[0] if isinstance(p_item, tuple) else p_item
                dim = p_item[1] if isinstance(p_item, tuple) else 0
                
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
                        
                        # 计算当前参数被置零的通道
                        num_zero_channels = torch.sum(torch.norm(param.data.float(), p=2, dim=1 if dim==0 else 0) == 0).item()
                        # 注意：FFN的一个掩码对应多个参数，这里我们只在每层最后统计一次
                        layer_pruned_channels = num_zero_channels 
                        break
            
            # 更新全局统计并打印当前层结果
            total_zeros += layer_pruned_channels
            total_non_zeros += (mask.shape[0] - layer_pruned_channels) # 基于掩码长度计算活跃单元
            logging.info(f"  [{component.upper()}] Layer {idx}: Pruned {layer_pruned_channels} channels")

    return total_zeros, total_non_zeros
    
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
    # 关键修改：处理掩码列表
    if args.mask_path:
        logging.info(f"DEBUG: 接收到的掩码路径列表: {args.mask_path}")
        inner_model = getattr(policy, "_model", None)
        if inner_model:
            # 关键：手动解析逗号分隔的路径
            path_list = [p.strip() for p in args.mask_path.split(",")]
            logging.info(f"DEBUG: 解析后的路径列表: {path_list}")
            apply_multiple_masks(inner_model, path_list)
        else:
            logging.warning("无法在策略对象中找到 _model，跳过剪枝应用。")
            
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