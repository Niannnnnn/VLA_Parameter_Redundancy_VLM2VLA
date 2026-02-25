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
    num_trials_per_task: int = 50                    # Number of rollouts per task

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
    layer_ffn_dims_file: Union[str, Path] = ""  # 每层剪枝后的 FFN 维度文件路径（.pth）

    # fmt: on

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
from ptflops import get_model_complexity_info
from PIL import Image
from huggingface_hub import hf_hub_download


@draccus.wrap()
def eval_pruned_libero(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint, "cfg.pretrained_checkpoint must not be None!"
    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # [OpenVLA] Set action un-normalization key
    cfg.unnorm_key = cfg.task_suite_name

    # # Load model
    # model = get_model(cfg)
    # --- [MODIFIED START] ---
    print(">>> Loading sharded pruned model...")

    model_dir = str(cfg.pretrained_checkpoint)
    if not os.path.isdir(model_dir):  # 可能是 HF repo id
        print(f"  Using HF cache for model: {model_dir}")
        # 下载 index 文件
        # index_file_url = f"{model_dir}/model.safetensors.index.json"
        # index_path = cached_download(index_file_url)
        index_path = hf_hub_download(
            repo_id=cfg.pretrained_checkpoint,
            filename="model.safetensors.index.json"
        )
    else:
        # 本地路径
        index_path = os.path.join(model_dir, "model.safetensors.index.json")

    # 1️⃣ 读取模型配置
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    # print(f"config: {config}")
    # print(f"config.text_config: {config.text_config}")
    print(f"config.text_config.intermediate_size: {config.text_config.intermediate_size}") 

    # 从pth文件加载剪枝信息并计算每层FFN维度
    print(f">>> Loading pruning info from: {cfg.layer_ffn_dims_file}")
    pruning_info = torch.load(cfg.layer_ffn_dims_file, map_location='cpu')

    # 计算每层的FFN维度
    layer_ffn_dims = []
    total_layers = 32

    for layer_idx in range(total_layers):
        # 查找该层的up_proj权重信息
        up_proj_key = f'language_model.model.layers.{layer_idx}.mlp.up_proj.weight'
        
        if up_proj_key in pruning_info:
            up_proj_info = pruning_info[up_proj_key]
            total_channels = up_proj_info['total_channels']
            pruned_channels = up_proj_info['pruned_channels']
            remaining_channels = total_channels - pruned_channels
            layer_ffn_dims.append(remaining_channels)
            print(f"Layer {layer_idx}: total_channels={total_channels}, pruned_channels={pruned_channels}, remaining={remaining_channels}")
        else:
            # 如果没有找到该层的剪枝信息，使用原始维度
            original_dim = config.text_config.intermediate_size
            layer_ffn_dims.append(original_dim)
            print(f"Layer {layer_idx}: Using original dimension {original_dim} (no pruning info found)")

    print(f">>> Final layer_ffn_dims: {layer_ffn_dims}")

    # 保存原始构造函数
    LlamaModel_init_orig = LlamaModel.__init__

    # 重写 LlamaModel 构造函数
    def patched_init(self, config):
        super(LlamaModel, self).__init__(config)

        # 临时保存原始的 intermediate_size
        orig_value = config.intermediate_size

        # 逐层动态调整
        layers = []
        for i in range(config.num_hidden_layers):
            config.intermediate_size = layer_ffn_dims[i]  # ✅ 修改配置
            layer = LlamaDecoderLayer(config, i)
            layers.append(layer)
        self.layers = torch.nn.ModuleList(layers)

        # 恢复原值（防止影响其他地方）
        config.intermediate_size = orig_value

        # 复制原本 LlamaModel 其他初始化逻辑
        self.embed_tokens = torch.nn.Embedding(config.vocab_size, config.hidden_size)
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

        print("✅ Patched LlamaModel loaded with per-layer FFN sizes.")

    # 替换原始构造
    LlamaModel.__init__ = patched_init
    

    # 2️⃣ 初始化空模型（结构）
    model = AutoModelForVision2Seq.from_config(config, trust_remote_code=True)

    # 3️⃣ 合并所有 safetensors 分片
    # index_path = os.path.join(model_dir, "model.safetensors.index.json")
    with open(index_path, "r") as f:
        index = json.load(f)

    weight_map = index["weight_map"]
    state_dict = {}

    print(f"Total tensors to load: {len(weight_map)}")
    print("Unique shard files:", set(weight_map.values()))

    for tensor_name, shard_file in weight_map.items():
        # shard_path = os.path.join(model_dir, shard_file)
        if os.path.isdir(cfg.pretrained_checkpoint):
            print(f"  Using local shard for tensor: {tensor_name}")
            shard_path = os.path.join(cfg.pretrained_checkpoint, shard_file)
        else:
            print(f"  Downloading shard for tensor: {tensor_name}")
            # HF Hub 缓存路径
            # shard_path = cached_download(f"{cfg.pretrained_checkpoint}/{shard_file}")
            shard_path = hf_hub_download(
                repo_id=cfg.pretrained_checkpoint,
                filename=shard_file
            )
        if shard_path not in state_dict:
            print(f"[Cache used] {shard_path}")
            print(f"  Loading {shard_file} ...", flush=True)
            shard_state = load_file(shard_path)
            print(f"  ✅ Finished loading {shard_file} ({len(shard_state)} tensors)", flush=True)
            state_dict.update(shard_state)

    # 4️⃣ 加载权重（不严格匹配）
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f">>> Missing keys: {len(missing)}, {missing} ; Unexpected keys: {len(unexpected)}, {unexpected}")

    # # -----------------------------融入 LoRA adapter-----------------------------
    # from peft import PeftModel

    # # 假设你这时 model 已经加载完主模型权重
    # adapter_dir = "./first_success_noSaveModel_1028/adapters_pruned_0.2_1028/openvla-pruned-7b-spatial-3iters-10plans-0.2+libero_spatial_no_noops+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug"

    # print(f">>> Loading LoRA adapter from: {adapter_dir}")
    # model = PeftModel.from_pretrained(model, adapter_dir)
    # print("✅ LoRA adapter loaded into model.")

    # # 合并 LoRA 权重到原模型中（这样推理时不再依赖 LoRA 模块）
    # model = model.merge_and_unload()
    # print("✅ LoRA adapter merged into base model.")
    # # -----------------------------end-----------------------------

    # -----------------------------全量微调模型加载-----------------------------
    print(">>> No LoRA adapter detected — loading full fine-tuned model directly.")
    # 全量微调的权重已经写入 model.safetensors，因此无需加载 adapter。
    # 直接进入推理模式。
    # -----------------------------end-----------------------------

    # 5️⃣ 移动到 GPU
    model.to("cuda", dtype=torch.bfloat16)
    model.eval()
    # for i, layer in enumerate(model.language_model.model.layers):
    #     print(f"  Layer {i}: {layer}")
    #     print(f"  Layer {i} weights: {layer.state_dict().keys()}, {layer.state_dict().values()}")
    #     for k, v in layer.state_dict().items():
    #         print(f"  {k}: {v.shape}")

    

    # 6️⃣ 加载 processor
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)

    print(">>> ✅ Pruned sharded model successfully loaded!")
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

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
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
    eval_pruned_libero()