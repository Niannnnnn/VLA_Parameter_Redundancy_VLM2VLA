"""
finetune_fsdp.py

FSDP-based fine-tuning script for OpenVLA models with parameter-efficient training support.

Key changes from DDP version:
- Uses FSDP (Fully Sharded Data Parallel) instead of DDP
- Shards model parameters, gradients, and optimizer states across GPUs
- Supports mixed precision training with FSDP
- Compatible with LoRA and full fine-tuning
- Includes proper FSDP checkpoint saving/loading

Run with:
    torchrun --standalone --nnodes 1 --nproc-per-node $K finetune_fsdp.py \
        --vla_path "/path/to/model" \
        --data_root_dir /path/to/data \
        --dataset_name libero_spatial_no_noops \
        --run_root_dir ./runs \
        --batch_size 4 \
        --learning_rate 5e-4
"""

import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
import functools

import draccus
import torch
import torch.distributed as dist
import tqdm
from accelerate import PartialState
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from transformers import AutoConfig, AutoImageProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast
import datetime

# FSDP imports
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
    FullStateDictConfig,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy,
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)

import wandb
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

import json
from transformers.models.llama.modeling_llama import LlamaModel, LlamaDecoderLayer, LlamaRMSNorm
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class FinetuneConfig:
    # fmt: off
    vla_path: str = "openvla/openvla-7b"
    
    # Directory Paths
    data_root_dir: Path = Path("datasets/open-x-embodiment")
    dataset_name: str = "droid_wipe"
    run_root_dir: Path = Path("runs")
    adapter_tmp_dir: Path = Path("adapter-tmp")
    
    # Fine-tuning Parameters
    batch_size: int = 16
    max_steps: int = 200_000
    save_steps: int = 5000
    learning_rate: float = 5e-4
    grad_accumulation_steps: int = 1
    image_aug: bool = True
    shuffle_buffer_size: int = 100_000
    save_latest_checkpoint_only: bool = True
    
    # LoRA Arguments
    use_lora: bool = True
    lora_rank: int = 32
    lora_dropout: float = 0.0
    use_quantization: bool = False
    
    # FSDP-specific Parameters
    fsdp_sharding_strategy: str = "FULL_SHARD"  # Options: FULL_SHARD, SHARD_GRAD_OP, NO_SHARD, HYBRID_SHARD
    fsdp_cpu_offload: bool = False  # Offload params to CPU when not in use
    fsdp_activation_checkpointing: bool = False  # Enable activation checkpointing
    fsdp_auto_wrap_min_params: int = 1e8  # Minimum parameters for auto-wrapping
    
    # Tracking Parameters
    # wandb_project: str = "openvla"
    # wandb_entity: str = "stanford-voltron"
    run_id_note: Optional[str] = None
    
    # Pruning Information
    layer_ffn_dims_file: Union[str, Path] = ""
    # fmt: on


def setup_fsdp_config(cfg: FinetuneConfig, model):
    """Setup FSDP configuration based on config parameters."""
    
    # Mixed Precision Policy
    mixed_precision_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    
    # Sharding Strategy
    sharding_strategy_map = {
        "FULL_SHARD": ShardingStrategy.FULL_SHARD,
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
        "NO_SHARD": ShardingStrategy.NO_SHARD,
        "HYBRID_SHARD": ShardingStrategy.HYBRID_SHARD,
    }
    sharding_strategy = sharding_strategy_map.get(
        cfg.fsdp_sharding_strategy, ShardingStrategy.FULL_SHARD
    )
    
    # Auto-wrap policy - wraps layers based on parameter count or transformer blocks
    # For transformer models, we use transformer_auto_wrap_policy
    try:
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                LlamaDecoderLayer,  # Wrap each Llama decoder layer
            },
        )
    except Exception:
        # Fallback to size-based wrapping
        auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy,
            min_num_params=cfg.fsdp_auto_wrap_min_params,
        )
    
    fsdp_config = {
        "sharding_strategy": sharding_strategy,
        "mixed_precision": mixed_precision_policy,
        "auto_wrap_policy": auto_wrap_policy,
        "device_id": torch.cuda.current_device(),
        "limit_all_gathers": True,  # Useful for memory optimization
        "use_orig_params": True,  # Required for optimizer state checkpointing with LoRA
    }
    
    # CPU offload configuration
    if cfg.fsdp_cpu_offload:
        from torch.distributed.fsdp import CPUOffload
        fsdp_config["cpu_offload"] = CPUOffload(offload_params=True)
    
    return fsdp_config


def apply_fsdp_activation_checkpointing(model):
    """Apply activation checkpointing to reduce memory usage."""
    non_reentrant_wrapper = functools.partial(
        checkpoint_wrapper,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )
    
    check_fn = lambda submodule: isinstance(submodule, LlamaDecoderLayer)
    
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=non_reentrant_wrapper,
        check_fn=check_fn,
    )


def save_fsdp_checkpoint(model, optimizer, run_dir, gradient_step_idx, cfg, processor, is_main_process):
    """Save FSDP model checkpoint."""
    
    if cfg.save_latest_checkpoint_only:
        checkpoint_dir = run_dir
    else:
        checkpoint_dir = Path(str(run_dir) + f"--{gradient_step_idx}_chkpt")
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    # FSDP state dict saving
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    
    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        save_policy,
    ):
        model_state_dict = model.state_dict()
        
        if is_main_process:
            # Save model weights
            if cfg.use_lora:
                # For LoRA, save adapter weights
                # adapter_dir = cfg.adapter_tmp_dir / checkpoint_dir.name
                # os.makedirs(adapter_dir, exist_ok=True)
                # model.module.save_pretrained(adapter_dir)
                # --- 修改后的 LoRA 保存逻辑 ---
                adapter_dir = cfg.adapter_tmp_dir / checkpoint_dir.name
                os.makedirs(adapter_dir, exist_ok=True)
                
                # 1. 提取只属于 LoRA 的权重 (通过 key 过滤)
                # 只有包含 "lora_" 关键字的权重才是我们需要保存的 adapter
                lora_state_dict = {k: v for k, v in model_state_dict.items() if "lora_" in k}
                print(f"🔍 Extracted {len(lora_state_dict)} LoRA parameters for saving.")
                print(f"Sample LoRA parameters keys: {list(lora_state_dict.keys())[:10]}")
                
                # 2. 使用 PEFT 的方法保存
                # 注意：这里调用 model.module.save_pretrained，但传入我们已经拿到的 state_dict
                # 这样它就不会自己再去同步获取，避免了死锁
                model.module.save_pretrained(
                    adapter_dir, 
                    state_dict=lora_state_dict
                )
                print(f"✅ Saved LoRA adapters to {adapter_dir}")
            else:
                # Save full model
                # torch.save(model_state_dict, checkpoint_dir / "pytorch_model.bin")
                # processor.save_pretrained(checkpoint_dir)
                # model.module.config.save_pretrained(checkpoint_dir)
                from safetensors.torch import save_file

                safetensors_path = checkpoint_dir / "model.safetensors"
                save_file(model_state_dict, safetensors_path)

                processor.save_pretrained(checkpoint_dir)
                model.module.config.save_pretrained(checkpoint_dir)

                # Also write an index file (optional but nice for HF compatibility)
                index_path = checkpoint_dir / "model.safetensors.index.json"
                with open(index_path, "w") as f:
                    import json
                    json.dump({"weight_map": {k: "model.safetensors" for k in model_state_dict.keys()}}, f)
            
            print(f"✅ Saved model checkpoint at step {gradient_step_idx}")
    
    # # Save optimizer state
    # optimizer_state = FSDP.full_optim_state_dict(model, optimizer)
    # if is_main_process:
    #     torch.save(optimizer_state, checkpoint_dir / "optimizer.pt")
    #     print(f"✅ Saved optimizer state at step {gradient_step_idx}")
    
    dist.barrier()


def load_pruned_model_with_fsdp_prep(cfg):
    """Load pruned model and prepare for FSDP wrapping."""
    
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    
    if not hasattr(cfg, "layer_ffn_dims_file") or not cfg.layer_ffn_dims_file:
        print("No pruning file provided - using standard model loading")
        vla = AutoModelForVision2Seq.from_pretrained(
            cfg.vla_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        return processor, vla
    
    # Pruned model loading logic
    print(f">>> Loading pruned model from {cfg.vla_path}")
    
    model_dir = str(cfg.vla_path)
    if os.path.isdir(model_dir):
        index_path = os.path.join(model_dir, "model.safetensors.index.json")
    else:
        index_path = hf_hub_download(repo_id=model_dir, filename="model.safetensors.index.json")
    
    # Load config and pruning info
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    pruning_info = torch.load(cfg.layer_ffn_dims_file, map_location="cpu")
    
    # Compute per-layer FFN dimensions
    num_layers = getattr(config, "num_hidden_layers", 32)
    layer_ffn_dims = []
    
    for layer_idx in range(num_layers):
        up_proj_key = f'language_model.model.layers.{layer_idx}.mlp.up_proj.weight'
        if up_proj_key in pruning_info:
            info = pruning_info[up_proj_key]
            total_channels = info.get("total_channels", config.text_config.intermediate_size)
            pruned_channels = info.get("pruned_channels", 0)
            remaining = int(total_channels - pruned_channels)
            layer_ffn_dims.append(remaining)
            print(f"  Layer {layer_idx}: remaining={remaining}")
        else:
            layer_ffn_dims.append(config.text_config.intermediate_size)
    
    # Patch LlamaModel.__init__
    LlamaModel_init_orig = LlamaModel.__init__
    
    def patched_init(self, config_local):
        super(LlamaModel, self).__init__(config_local)
        orig_intermediate = getattr(config_local, "intermediate_size", None)
        layers_list = []
        
        for i in range(config_local.num_hidden_layers):
            new_inter = layer_ffn_dims[i] if i < len(layer_ffn_dims) else orig_intermediate
            setattr(config_local, "intermediate_size", new_inter)
            layer = LlamaDecoderLayer(config_local, i)
            layers_list.append(layer)
        
        setattr(config_local, "intermediate_size", orig_intermediate)
        self.layers = torch.nn.ModuleList(layers_list)
        self.embed_tokens = torch.nn.Embedding(config_local.vocab_size, config_local.hidden_size)
        self.norm = LlamaRMSNorm(config_local.hidden_size, eps=config_local.rms_norm_eps)
        self.gradient_checkpointing = False
    
    LlamaModel.__init__ = patched_init
    
    # Create empty model
    vla = AutoModelForVision2Seq.from_config(config, trust_remote_code=True)
    
    # Load weights from shards
    with open(index_path, "r") as f:
        index = json.load(f)
    
    weight_map = index.get("weight_map", {})
    state_dict = {}
    
    for tensor_name, shard_file in weight_map.items():
        if os.path.isdir(model_dir):
            shard_path = os.path.join(model_dir, shard_file)
        else:
            shard_path = hf_hub_download(repo_id=model_dir, filename=shard_file)
        
        shard_state = load_file(shard_path)
        state_dict.update(shard_state)
    
    missing, unexpected = vla.load_state_dict(state_dict, strict=False)
    print(f">>> Loaded weights - Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    
    # Restore original init
    LlamaModel.__init__ = LlamaModel_init_orig
    
    return processor, vla


@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    print(f"🚀 Fine-tuning OpenVLA with FSDP: `{cfg.vla_path}` on `{cfg.dataset_name}`")
    
    # Setup distributed environment
    assert torch.cuda.is_available(), "FSDP training requires CUDA!"
    
    # Initialize distributed
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=3600))
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    is_main_process = global_rank == 0
    
    print(f"Process {global_rank}/{world_size} on device {device}")
    
    # Configure experiment ID
    exp_id = (
        f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}"
        f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
        f"+lr-{cfg.learning_rate}"
        f"+fsdp-{cfg.fsdp_sharding_strategy}"
    )
    if cfg.use_lora:
        exp_id += f"+lora-r{cfg.lora_rank}"
    if cfg.image_aug:
        exp_id += "+img_aug"
    if cfg.run_id_note:
        exp_id += f"--{cfg.run_id_note}"
    
    run_dir = cfg.run_root_dir / exp_id
    adapter_dir = cfg.adapter_tmp_dir / exp_id
    
    if is_main_process:
        os.makedirs(run_dir, exist_ok=True)
    
    dist.barrier()
    
    # Load model and processor
    processor, vla = load_pruned_model_with_fsdp_prep(cfg)
    
    # LoRA setup (before FSDP wrapping)
    # if cfg.use_lora:
    #     lora_config = LoraConfig(
    #         r=cfg.lora_rank,
    #         lora_alpha=min(cfg.lora_rank, 16),
    #         lora_dropout=cfg.lora_dropout,
    #         target_modules=["gate_proj", "down_proj", "up_proj"],
    #         init_lora_weights="gaussian",
    #     )
    #     vla = get_peft_model(vla, lora_config)
    #     if is_main_process:
    #         vla.print_trainable_parameters()







    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_rank, # 通常 alpha 设为与 r 一致或 r*2
            lora_dropout=cfg.lora_dropout,
            # 确保只针对 FFN 模块
            target_modules=["gate_proj", "up_proj", "down_proj"], 
            init_lora_weights="gaussian",
            bias="none",
            task_type="CAUSAL_LM", # 明确任务类型
        )
        vla = get_peft_model(vla, lora_config)
        if is_main_process:
            vla.print_trainable_parameters()

    









    # Setup FSDP
    fsdp_config = setup_fsdp_config(cfg, vla)
    
    # Wrap model with FSDP
    vla = FSDP(vla, **fsdp_config)
    
    # Apply activation checkpointing if enabled
    if cfg.fsdp_activation_checkpointing:
        apply_fsdp_activation_checkpointing(vla)
        if is_main_process:
            print("✅ Applied FSDP activation checkpointing")
    
    # Create optimizer
    optimizer = AdamW(
        [p for p in vla.parameters() if p.requires_grad],
        lr=cfg.learning_rate,
    )
    
    # Create dataset
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
    )
    
    vla_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.module.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
    )
    
    if is_main_process:
        save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)
    
    # Create dataloader
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length,
        processor.tokenizer.pad_token_id,
        padding_side="right",
    )
    
    dataloader = DataLoader(
        vla_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,
    )
    
    # # Initialize W&B
    # if is_main_process:
    #     wandb.init(
    #         entity=cfg.wandb_entity,
    #         project=cfg.wandb_project,
    #         name=f"fsdp+{exp_id}",
    #     )
    
    # Training metrics
    recent_losses = deque(maxlen=cfg.grad_accumulation_steps)
    recent_action_accuracies = deque(maxlen=cfg.grad_accumulation_steps)
    recent_l1_losses = deque(maxlen=cfg.grad_accumulation_steps)
    
    # Training loop
    with tqdm.tqdm(total=cfg.max_steps, disable=not is_main_process) as progress:
        vla.train()
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(dataloader):
            # Forward pass
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output: CausalLMOutputWithPast = vla(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device),
                    labels=batch["labels"].to(device),
                )
                loss = output.loss
                print(f"Step {batch_idx}, Loss: {loss.item()}")
            
            # Normalize loss
            normalized_loss = loss / cfg.grad_accumulation_steps
            print(f"Normalized Loss: {normalized_loss.item()}")
            
            # Backward pass
            normalized_loss.backward()
            print("Backward pass done")
            
            # Compute metrics
            action_logits = output.logits[:, vla.module.vision_backbone.featurizer.patch_embed.num_patches : -1]
            action_preds = action_logits.argmax(dim=2)
            action_gt = batch["labels"][:, 1:].to(action_preds.device)
            mask = action_gt > action_tokenizer.action_token_begin_idx
            
            correct_preds = (action_preds == action_gt) & mask
            action_accuracy = correct_preds.sum().float() / mask.sum().float()
            
            continuous_actions_pred = torch.tensor(
                action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())
            )
            continuous_actions_gt = torch.tensor(
                action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy())
            )
            action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)
            
            recent_losses.append(loss.item())
            recent_action_accuracies.append(action_accuracy.item())
            recent_l1_losses.append(action_l1_loss.item())
            
            gradient_step_idx = batch_idx // cfg.grad_accumulation_steps
            
            # Optimizer step
            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                progress.update()
                
                # # Log metrics
                # if is_main_process and gradient_step_idx % 10 == 0:
                #     wandb.log(
                #         {
                #             "train_loss": sum(recent_losses) / len(recent_losses),
                #             "action_accuracy": sum(recent_action_accuracies) / len(recent_action_accuracies),
                #             "l1_loss": sum(recent_l1_losses) / len(recent_l1_losses),
                #         },
                #         step=gradient_step_idx,
                #     )
            
            # Save checkpoint
            if gradient_step_idx > 0 and gradient_step_idx % cfg.save_steps == 0:
                save_fsdp_checkpoint(
                    vla, optimizer, run_dir, gradient_step_idx,
                    cfg, processor, is_main_process
                )
            
            # Stop at max steps
            if gradient_step_idx >= cfg.max_steps:
                dist.barrier()
                print(f"✅ Reached max steps {cfg.max_steps}")
                save_fsdp_checkpoint(
                    vla, optimizer, run_dir, gradient_step_idx,
                    cfg, processor, is_main_process
                )
                dist.barrier()
                break
    
    # if is_main_process:
    #     wandb.finish()
    
    dist.destroy_process_group()


if __name__ == "__main__":
    finetune()