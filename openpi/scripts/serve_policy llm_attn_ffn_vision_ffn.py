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
    
    # 用于汇总统计
    total_zeros = 0
    total_non_zeros = 0

    with torch.no_grad():
        for layer_key, info in layers_mask.items():
            idx = layer_key.split('.')[-1]
            mask = torch.from_numpy(info["mask"])
            
            target_params = []
            if component == "llm_ffn":
                target_params.append(f"{prefix}.language_model.layers.{idx}.mlp.gate_proj.weight")
                target_params.append(f"{prefix}.language_model.layers.{idx}.mlp.up_proj.weight")
                target_params.append((f"{prefix}.language_model.layers.{idx}.mlp.down_proj.weight", 1))
            # elif component == "llm_attn":
            #     target_params.append(f"{prefix}.language_model.layers.{idx}.self_attn.q_proj.weight")
            #     target_params.append(f"{prefix}.language_model.layers.{idx}.self_attn.k_proj.weight")
            #     target_params.append(f"{prefix}.language_model.layers.{idx}.self_attn.v_proj.weight")
            
            elif component == "llm_attn":
                q_proj_name = f"{prefix}.language_model.layers.{idx}.self_attn.q_proj.weight"
                
                # 只处理 q_proj，k/v 不剪
                found = False
                for name, param in model.named_parameters():
                    if name == q_proj_name:
                        mask = torch.from_numpy(info["mask"])  # shape: (8,)
                        num_heads = mask.shape[0]
                        head_dim = param.shape[0] // num_heads  # 假设 q_proj 输出维度 = num_heads * head_dim

                        # 重塑为 (num_heads, head_dim, hidden_dim)
                        param_reshaped = param.data.view(num_heads, head_dim, -1)

                        # 根据掩码置零被剪枝的 head
                        for head_id in range(num_heads):
                            if not mask[head_id]:
                                param_reshaped[head_id] = 0.0

                        # 写回原参数
                        param.data.copy_(param_reshaped.view_as(param))

                        # 统计
                        num_zero_heads = torch.sum(~mask).item()
                        num_kept_heads = torch.sum(mask).item()
                        logging.info(f"  [统计] 参数: {name.split('.')[-2:]}")
                        logging.info(f"         已剪枝 heads: {num_zero_heads} | 保留 heads: {num_kept_heads}")

                        total_zeros += num_zero_heads
                        total_non_zeros += num_kept_heads
                        found = True
                        break

                if not found:
                    logging.warning(f"未能在模型中找到参数: {q_proj_name}")


            elif component == "vision_ffn":
                target_params.append(f"{prefix}.vision_tower.vision_model.encoder.layers.{idx}.mlp.fc1.weight")
                target_params.append((f"{prefix}.vision_tower.vision_model.encoder.layers.{idx}.mlp.fc2.weight", 1))
            elif component == "vision_attn":
                target_params.append(f"{prefix}.vision_tower.vision_model.encoder.layers.{idx}.self_attn.q_proj.weight")
                target_params.append(f"{prefix}.vision_tower.vision_model.encoder.layers.{idx}.self_attn.k_proj.weight")
                target_params.append(f"{prefix}.vision_tower.vision_model.encoder.layers.{idx}.self_attn.v_proj.weight")

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

                        # 应用掩码
                        if dim == 0:
                            param.data.mul_(m.view(-1, *([1] * (param.ndim - 1))))
                        else:
                            param.data.mul_(m.view(1, -1, *([1] * (param.ndim - 2))))
                        
                        # --- 新增统计逻辑 ---
                        # 计算当前参数在指定维度（通道）上的范数，如果范数为0，说明该通道被整体置零
                        # 这样统计的是“被剪掉的通道数”，而不是单个权重的个数，更符合你的掩码逻辑
                        channel_norms = torch.norm(param.data.float(), p=2, dim=1 if dim==0 else 0)
                        num_zero_channels = torch.sum(channel_norms == 0).item()
                        num_active_channels = torch.sum(channel_norms > 0).item()
                        
                        logging.info(f"  [统计] 参数: {p_name.split('.')[-2:]}")
                        logging.info(f"         已置零通道: {num_zero_channels} | 活跃通道: {num_active_channels}")
                        
                        total_zeros += num_zero_channels
                        total_non_zeros += num_active_channels
                        # --------------------
                        
                        found = True
                        break
                if not found:
                    logging.warning(f"未能在模型中找到参数: {p_name}")

    logging.info(f"✅ 掩码应用完成。总计已置零通道: {total_zeros}, 总计活跃通道: {total_non_zeros}")
    
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