"""
run_pruned_libero_eval.py

Runs a model with FFN pruning in a LIBERO simulation environment.

Usage:
    # OpenVLA with pruning:
    python experiments/robot/libero/run_pruned_libero_eval.py \
        --model_family openvla \
        --pretrained_checkpoint <CHECKPOINT_PATH> \
        --task_suite_name [ libero_spatial | libero_object | libero_goal | libero_10 | libero_90 ] \
        --center_crop [ True | False ] \
        --run_id_note <OPTIONAL TAG FOR LOGGING> \
        --use_wandb [ True | False ] \
        --wandb_project <PROJECT> \
        --wandb_entity <ENTITY>
"""

import os
import sys
import torch
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union, Dict, Any, List

import draccus
import numpy as np
import tqdm
from libero.libero import benchmark

import wandb

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization
    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    
    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_spatial"          # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 10                    # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_project: str = "YOUR_WANDB_PROJECT"        # Name of W&B project to log to (use default!)
    wandb_entity: str = "YOUR_WANDB_ENTITY"          # Name of entity to log under

    seed: int = 7                                    # Random Seed (for reproducibility)

    #################################################################################################################
    # Prune Information
    #################################################################################################################
    # layer_ffn_dims_file: Union[str, Path] = ""  # 每层剪枝后的 FFN 维度文件路径（.pth）
    layer_ffn_dims_file: Union[List[str], str] = ""

    # fmt: on

    pruning_mode: str = "real"  # 选项: "real" (物理删除通道), "masked" (原模型+掩码置零)

import torch
import numpy as np
import re
from typing import Dict, Any, Union, List, Tuple, Optional
from transformers import AutoConfig, AutoModelForVision2Seq, AutoProcessor
from safetensors.torch import load_file
import json
from transformers.models.llama.modeling_llama import LlamaModel, LlamaDecoderLayer, LlamaRMSNorm
from torch.profiler import profile, record_function, ProfilerActivity
import warnings
# from ptflops import get_model_complexity_info
from PIL import Image
from huggingface_hub import hf_hub_download


def print_model_dims(model):
    print("\n" + "=="*40)
    print(f"{'MODEL DIMENSION & MASKED SPARSITY REPORT':^80}")
    print("=="*40)

    # 内部辅助函数：计算稀疏度并格式化输出
    def get_layer_stats(layer):
        if layer is None or not hasattr(layer, 'weight'): 
            return "N/A"
        weight = layer.weight.data
        total_params = weight.numel()
        # 计算权重为 0 的个数
        zero_cnt = (weight == 0).sum().item()
        sparsity = zero_cnt / total_params if total_params > 0 else 0
        shape_str = str(list(weight.shape))
        return f"{shape_str:<18} | Sparsity: {sparsity:>5.1%}"

    # 内部辅助函数：仅获取稀疏度百分比（用于 LLM 紧凑表格）
    def get_s_pct(layer):
        if layer is None or not hasattr(layer, 'weight'): return "N/A"
        w = layer.weight.data
        total = w.numel()
        return f"{(w == 0).sum().item() / total:.1%}" if total > 0 else "0.0%"

    # --- 1. Vision Backbone (逐层打印) ---
    if hasattr(model, "vision_backbone"):
        vb = model.vision_backbone
        branches = []
        if hasattr(vb, "featurizer"): branches.append(("DINO", vb.featurizer))
        if hasattr(vb, "fused_featurizer"): branches.append(("SigLIP", vb.fused_featurizer))
        
        for b_name, branch in branches:
            print(f"\n[{b_name} Branch - Full Architecture]")
            print(f"{'Layer':<6} | {'Component':<8} | {'Shape':<18} | {'Sparsity'}")
            print("-" * 60)
            for i, block in enumerate(branch.blocks):
                # 打印 QKV 
                print(f"L{i:02d}   | qkv      | {get_layer_stats(block.attn.qkv)}")
                # 打印 MLP 第一层 (fc1)
                print(f"L{i:02d}   | fc1      | {get_layer_stats(block.mlp.fc1)}")
                # 如果你想看投影层，也可以加上 block.attn.proj

    # --- 2. Projector (全量检查) ---
    if hasattr(model, "projector"):
        print(f"\n[Projector Layers]")
        proj = model.projector
        # 按照常见的命名顺序检查
        possible_names = ["fc1", "fc2", "fc3", "projector.0", "projector.2", "projector.4"]
        for name in possible_names:
            layer = None
            if "." in name: # Sequential
                parts = name.split(".")
                sub = getattr(proj, parts[0], None)
                if sub and int(parts[1]) < len(sub):
                    layer = sub[int(parts[1])]
            else: # Attribute
                layer = getattr(proj, name, None)
            
            if layer and hasattr(layer, "weight"):
                print(f"  {name:<11} : {get_layer_stats(layer)}")

    # --- 3. Language Model (逐层打印) ---
    if hasattr(model, "language_model"):
        print(f"\n[Language Model - All Layers]")
        print(f"{'Layer':<6} | {'Q / O Sparsity':<20} | {'Gate / Down Sparsity':<20}")
        print("-" * 60)
        
        layers = model.language_model.model.layers
        for i, layer in enumerate(layers):
            # 获取 Attention 和 MLP 核心矩阵的稀疏度
            q_s = get_s_pct(layer.self_attn.q_proj)
            o_s = get_s_pct(layer.self_attn.o_proj)
            g_s = get_s_pct(layer.mlp.gate_proj)
            d_s = get_s_pct(layer.mlp.down_proj)
            
            print(f"L{i:02d}   | {q_s + ' / ' + o_s:<20} | {g_s + ' / ' + d_s:<20}")

    print("\n" + "=="*40)


@draccus.wrap()
def eval_pruned_libero(cfg: GenerateConfig) -> None:
    print("Start eval_pruned_libero")
    print("Evaluating with configuration:", cfg)  # 打印配置

    assert cfg.pretrained_checkpoint, "cfg.pretrained_checkpoint must not be None!"
    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # [OpenVLA] Set action un-normalization key
    cfg.unnorm_key = cfg.task_suite_name
    
    # --- [MODIFIED START] ---
    # 0️⃣ 模式选择：通过 cfg 里的参数判断。如果你的 cfg 还没有这个参数，
    # 可以在命令行通过 --pruning_mode "masked" 或 "real" 传入
    pruning_mode = getattr(cfg, "pruning_mode", "real") 
    print(f">>> Execution Mode: {pruning_mode.upper()}")

    model_dir = str(cfg.pretrained_checkpoint)
    
    # 获取 index 文件路径
    if not os.path.isdir(model_dir):
        print(f"  Using HF cache for model: {model_dir}")
        index_path = hf_hub_download(repo_id=cfg.pretrained_checkpoint, filename="model.safetensors.index.json")
    else:
        index_path = os.path.join(model_dir, "model.safetensors.index.json")

    # 读取模型基础配置
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    print(f"config.text_config.intermediate_size: {config.text_config.intermediate_size}") 





    # # 判断是否启用剪枝
    # use_pruning = (cfg.layer_ffn_dims_file != "")





    # --- 核心修复逻辑：手动解析各种可能的输入格式 ---
    raw_files = cfg.layer_ffn_dims_file
    mask_files = []

    if isinstance(raw_files, str):
        # 尝试处理 JSON 字符串格式，如 '["path1", "path2"]'
        if raw_files.startswith('[') and raw_files.endswith(']'):
            try:
                mask_files = json.loads(raw_files)
            except:
                mask_files = [raw_files]
        elif raw_files.strip() == "":
            mask_files = []
        else:
            # 处理普通单路径字符串
            mask_files = [raw_files]
    elif isinstance(raw_files, list):
        mask_files = raw_files
    else:
        mask_files = []

    # 将处理后的列表写回 cfg，确保后续逻辑正常
    cfg.layer_ffn_dims_file = mask_files
    # --- 修复结束 ---

    # --- 鲁棒性处理开始 ---
    # 确保 layer_ffn_dims_file 始终是一个 List[str]
    mask_files = cfg.layer_ffn_dims_file
    if isinstance(mask_files, str):
        mask_files = [mask_files]
    elif mask_files is None:
        mask_files = []
    # --- 鲁棒性处理结束 ---

    use_pruning = (len(mask_files) > 0)


    
    if not use_pruning:
        # --- 正常加载原始模型 ---
        print(">>> Loading standard model (No Pruning)...")
        model = AutoModelForVision2Seq.from_pretrained(
            model_dir,
            trust_remote_code=True,
            load_in_8bit=cfg.load_in_8bit,
            load_in_4bit=cfg.load_in_4bit,
            torch_dtype=torch.bfloat16
        )
    else:
        # # 从 pth 文件加载剪枝掩码信息
        # print(f">>> Loading pruning info from: {cfg.layer_ffn_dims_file}")
        # pruning_info = torch.load(cfg.layer_ffn_dims_file, map_location='cpu')


        # =========================================================================
        # 合并多个掩码文件
        # =========================================================================
        combined_pruning_info = {}
        for mask_file in cfg.layer_ffn_dims_file:
            print(f">>> Loading pruning info from: {mask_file}")
            # 如果传入的是字符串而非列表，做个兼容
            current_info = torch.load(mask_file, map_location='cpu')
            
            # 合并字典。注意：如果多个文件包含同一个 key，后者会覆盖前者
            # 这正好符合“每个文件针对不同模块”的需求
            combined_pruning_info.update(current_info)
        
        pruning_info = combined_pruning_info
        print(f">>> Total masked keys collected: {len(pruning_info.keys())}")


        # =========================================================================
        # 情况 A: 物理剪枝模式 (Real Pruning)
        # =========================================================================
        if pruning_mode == "real":
            print(">>> Initializing model with Physical Pruning (Reduced Structure)...")
            # 计算每层的 FFN 维度
            layer_ffn_dims = []
            for layer_idx in range(32):
                up_proj_key = f'language_model.model.layers.{layer_idx}.mlp.up_proj.weight'
                if up_proj_key in pruning_info:
                    up_proj_info = pruning_info[up_proj_key]
                    remaining_channels = up_proj_info['total_channels'] - up_proj_info['pruned_channels']
                    layer_ffn_dims.append(remaining_channels)
                else:
                    layer_ffn_dims.append(config.text_config.intermediate_size)

            # 挂载 Patch 构造函数
            def patched_init(self, config):
                super(LlamaModel, self).__init__(config)
                orig_value = config.intermediate_size
                layers = []
                for i in range(config.num_hidden_layers):
                    config.intermediate_size = layer_ffn_dims[i]
                    layers.append(LlamaDecoderLayer(config, i))
                self.layers = torch.nn.ModuleList(layers)
                config.intermediate_size = orig_value
                self.embed_tokens = torch.nn.Embedding(config.vocab_size, config.hidden_size)
                self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
                self.gradient_checkpointing = False
                print(f"✅ Layer-wise FFN initialized for Real Pruning.")

            LlamaModel.__init__ = patched_init
            
            # 初始化结构并准备加载权重
            model = AutoModelForVision2Seq.from_config(config, trust_remote_code=True)

        # =========================================================================
        # 情况 B: 掩码剪枝模式 (Masked Pruning)
        # =========================================================================
        else:
            print(">>> Initializing model with Weight Masking (Full Structure)...")
            # 掩码模式下，我们直接使用原始配置初始化（或者直接 from_pretrained）
            # 为了保证加载速度和一致性，这里使用 from_config 初始化空架子
            model = AutoModelForVision2Seq.from_config(config, trust_remote_code=True)

        # 3️⃣ 权重加载逻辑 (两者通用，但 Masked 模式加载的是全量权重，Real 模式加载的是删减后的权重)
        with open(index_path, "r") as f:
            index = json.load(f)

        weight_map = index["weight_map"]
        state_dict = {}
        for tensor_name, shard_file in weight_map.items():
            if os.path.isdir(model_dir):
                shard_path = os.path.join(model_dir, shard_file)
            else:
                shard_path = hf_hub_download(repo_id=cfg.pretrained_checkpoint, filename=shard_file)
            
            if shard_path not in state_dict:
                shard_state = load_file(shard_path)
                state_dict.update(shard_state)

        # 加载权重到模型
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f">>> Load Complete. Missing: {len(missing)}, Unexpected: {len(unexpected)}")


        # =========================================================================
        # 核心步骤：如果是 Masked 模式，在此处手动置零权重 (覆盖 LLM, Vision, Projector)
        # =========================================================================
        if pruning_mode == "masked":
            print(">>> Applying Global Masking (Attention + FFN)...")
            model.to("cuda")
            with torch.no_grad():
                # --- 1. Language Model (Attention & FFN) ---
                for i, layer in enumerate(model.language_model.model.layers):
                    # FFN: up, gate (output side), down (input side)
                    for proj_name in ["up_proj", "gate_proj", "down_proj"]:
                        key = f"language_model.model.layers.{i}.mlp.{proj_name}.weight"
                        if key in pruning_info:
                            m_ts = torch.tensor(pruning_info[key]["output_mask"] if "down" not in proj_name else pruning_info[key]["input_mask"]).cuda()
                            proj = getattr(layer.mlp, proj_name)
                            if "down" in proj_name: 
                                proj.weight.data[:, ~m_ts] = 0
                            else:                   
                                proj.weight.data[~m_ts, :] = 0
                                if hasattr(proj, 'bias') and proj.bias is not None:
                                    proj.bias.data[~m_ts] = 0

                    # Attention: q, k, v (output side), o (input side)
                    for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                        key = f"language_model.model.layers.{i}.self_attn.{proj_name}.weight"
                        if key in pruning_info:
                            m_ts = torch.tensor(pruning_info[key]["output_mask"] if "o_proj" not in proj_name else pruning_info[key]["input_mask"]).cuda()
                            proj = getattr(layer.self_attn, proj_name)
                            if "o_proj" in proj_name: 
                                proj.weight.data[:, ~m_ts] = 0
                            else:                    
                                proj.weight.data[~m_ts, :] = 0
                                if hasattr(proj, 'bias') and proj.bias is not None:
                                    proj.bias.data[~m_ts] = 0

                # --- 2. Vision Backbone ---
                vb = model.vision_backbone
                configs = [
                    (vb.featurizer, "vision_backbone.featurizer.blocks"),
                    (vb.fused_featurizer, "vision_backbone.fused_featurizer.blocks")
                ]
                
                for branch_obj, prefix in configs:
                    for i, block in enumerate(branch_obj.blocks):
                        # --- A. Vision MLP (fc1, fc2) ---
                        for fc_name in ["fc1", "fc2"]:
                            key = f"{prefix}.{i}.mlp.{fc_name}.weight"
                            if key in pruning_info:
                                info = pruning_info[key]
                                m_ts = torch.tensor(info["output_mask"] if "fc1" in fc_name else info["input_mask"]).cuda()
                                fc = getattr(block.mlp, fc_name)
                                if "fc1" in fc_name: 
                                    fc.weight.data[~m_ts, :] = 0
                                    if hasattr(fc, 'bias') and fc.bias is not None: 
                                        fc.bias.data[~m_ts] = 0
                                else: 
                                    fc.weight.data[:, ~m_ts] = 0
                        
                        # --- B. Vision Attention ---
                        qkv_key = f"{prefix}.{i}.attn.qkv.weight"
                        if qkv_key in pruning_info:
                            m_ts = torch.tensor(pruning_info[qkv_key]["output_mask"]).cuda()
                            qkv_weight = block.attn.qkv.weight.data
                            if m_ts.shape[0] != qkv_weight.shape[0]:
                                m_ts_expanded = torch.cat([m_ts, m_ts, m_ts], dim=0)
                            else:
                                m_ts_expanded = m_ts

                            block.attn.qkv.weight.data[~m_ts_expanded, :] = 0
                            if hasattr(block.attn.qkv, 'bias') and block.attn.qkv.bias is not None:
                                block.attn.qkv.bias.data[~m_ts_expanded] = 0
                        
                        v_proj_key = f"{prefix}.{i}.attn.proj.weight"
                        if v_proj_key in pruning_info:
                            m_ts = torch.tensor(pruning_info[v_proj_key]["input_mask"]).cuda()
                            block.attn.proj.weight.data[:, ~m_ts] = 0

                # --- 3. Projector ---
                proj = model.projector
                for name in ["fc1", "fc2", "fc3"]:
                    key = f"projector.{name}.weight"
                    if key in pruning_info:
                        info = pruning_info[key]
                        layer = getattr(proj, name)
                        if info.get("output_mask") is not None:
                            m_out = torch.tensor(info["output_mask"]).cuda()
                            layer.weight.data[~m_out, :] = 0
                            if hasattr(layer, 'bias') and layer.bias is not None: 
                                layer.bias.data[~m_out] = 0
                        if info.get("input_mask") is not None:
                            m_in = torch.tensor(info["input_mask"]).cuda()
                            layer.weight.data[:, ~m_in] = 0

                print("✅ All components (LLM, Vision, Projector) masking applied successfully.")

    # 5️⃣ 统一移动到 GPU 并设置评估模式
    model.to("cuda", dtype=torch.bfloat16)
    model.eval()

    # 打印维度以验证剪枝
    print_model_dims(model)

    # 6️⃣ 加载处理器
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    print(">>> ✅ Model loading and preparation finished!")
    # --- [MODIFIED END] ---





    # [OpenVLA] Check that the model contains the action un-normalization key
    if cfg.model_family == "openvla":
        # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
        # with the suffix "_no_noops" in the dataset name)
        if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        assert cfg.unnorm_key in model.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"

    # # [OpenVLA] Get Hugging Face processor
    # processor = None
    # if cfg.model_family == "openvla":
    #     processor = get_processor(cfg)

    # Initialize local logging
    run_id = f"PRUNED-EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging as well
    if cfg.use_wandb:
        config_dict = {k: v for k, v in vars(cfg).items()}
            
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
            config=config_dict
        )

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.task_suite_name}")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")
    

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = get_libero_env(task, cfg.model_family, resolution=256)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")

            # Reset environment
            env.reset()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            if cfg.task_suite_name == "libero_spatial":
                max_steps = 220  # longest training demo has 193 steps
            elif cfg.task_suite_name == "libero_object":
                max_steps = 280  # longest training demo has 254 steps
            elif cfg.task_suite_name == "libero_goal":
                max_steps = 300  # longest training demo has 270 steps
            elif cfg.task_suite_name == "libero_10":
                max_steps = 520  # longest training demo has 505 steps
            elif cfg.task_suite_name == "libero_90":
                max_steps = 400  # longest training demo has 373 steps

            print(f"Starting episode {task_episodes+1}...")
            log_file.write(f"Starting episode {task_episodes+1}...\n")
            while t < max_steps + cfg.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < cfg.num_steps_wait:
                        obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                        t += 1
                        continue

                    # Get preprocessed image
                    img = get_libero_image(obs, resize_size)

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    # Prepare observations dict
                    # Note: OpenVLA does not take proprio state as input
                    observation = {
                        "full_image": img,
                        "state": np.concatenate(
                            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                        ),
                    }
                    # print(f"obs: {obs['robot0_eef_pos']}, {obs['robot0_eef_quat']}, {obs['robot0_gripper_qpos']}")
                    # print(f"obs shape: {obs['robot0_eef_pos'].shape}, {obs['robot0_eef_quat'].shape}, {obs['robot0_gripper_qpos'].shape}") # (3,), (4,), (2,)
                    # print(f"quat2axisangle shape: {quat2axisangle(obs['robot0_eef_quat']).shape}") # 3

                    # Query model to get action
                    action = get_action(
                        cfg,
                        model,
                        observation,
                        task_description,
                        processor=processor,
                    )

                    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
                    action = normalize_gripper_action(action, binarize=True)

                    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
                    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
                    if cfg.model_family == "openvla":
                        action = invert_gripper_action(action)

                    # print(">>> BEFORE env.step()", flush=True)

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())

                    # print(">>> AFTER env.step()", flush=True)


                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    print(f"Caught exception: {e}")
                    log_file.write(f"Caught exception: {e}\n")
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            save_rollout_video(
                replay_images, total_episodes, success=done, task_description=task_description, log_file=log_file
            )

            # Log current results
            print(f"Success: {done}")
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            log_file.write(f"Success: {done}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
            log_file.flush()

        # Log final results
        print(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        log_file.write(f"Current task success rate: {float(task_successes) / float(task_episodes)}\n")
        log_file.write(f"Current total success rate: {float(total_successes) / float(total_episodes)}\n")
        log_file.flush()
        if cfg.use_wandb:
            wandb.log(
                {
                    f"success_rate/{task_description}": float(task_successes) / float(task_episodes),
                    f"num_episodes/{task_description}": task_episodes,
                }
            )

    # Save local log file
    log_file.close()

    # Push total metrics and local log file to wandb
    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": float(total_successes) / float(total_episodes),
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)


if __name__ == "__main__":
    print("main!!!!!!!!!!!!!!!!!!!!!!!!!!")

    eval_pruned_libero()