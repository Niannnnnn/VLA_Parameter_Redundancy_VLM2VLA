import os
import torch

from transformers import (
    AutoModel,
    AutoConfig,
    AutoModelForVision2Seq,
)

from prismatic import load

def extract_model_info(model, model_name):
    """提取模型的结构和参数信息"""
    try:
        # 获取模型结构
        structure = {
            'name': model_name,
            'num_parameters': sum(p.numel() for p in model.parameters()),  # 总参数数量
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),  # 可训练参数数量
        }
        return structure
    except Exception as e:
        print(f"提取模型信息时出错：{e}")
        return None
    
def show_model_params(model, save_path):
    # print(f"模型参数名称与尺寸如下：")
    state_dict = model.state_dict()
    # for name, param in state_dict.items():
    #     print(f"{name:60s} {tuple(param.shape)}")

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(str(model))

    
    print(f"✅ 参数信息已保存到 {save_path}")

def weight_mapping(model_orig, model_finetuned, save_path, component="llm"):
    plm_state = model_orig.state_dict()
    olm_state = model_finetuned.state_dict()

    # 动态定义映射关系
    if component == "llm":
        prefix_map = [("llm_backbone.llm.model.layers.", "module.language_model.model.layers.")]
        target_suffixes = ["q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight", 
                           "up_proj.weight", "gate_proj.weight", "down_proj.weight"]
    elif component == "vision_dino":
        prefix_map = [("vision_backbone.dino_featurizer.blocks.", "module.vision_backbone.featurizer.blocks.")]
        target_suffixes = ["attn.qkv.weight", "attn.proj.weight", "mlp.fc1.weight", "mlp.fc2.weight"]
    elif component == "vision_siglip":
        prefix_map = [("vision_backbone.siglip_featurizer.blocks.", "module.vision_backbone.fused_featurizer.blocks.")]
        target_suffixes = ["attn.qkv.weight", "attn.proj.weight", "mlp.fc1.weight", "mlp.fc2.weight"]
    elif component == "projector":
        prefix_map = [("projector.projector.0", "module.projector.fc1"),
                      ("projector.projector.2", "module.projector.fc2"),
                      ("projector.projector.4", "module.projector.fc3")]
        target_suffixes = ["weight"]

    mapping_list = []
    for plm_name in plm_state.keys():
        if not any(suffix in plm_name for suffix in target_suffixes): continue
        
        for plm_prefix, olm_prefix in prefix_map:
            if plm_name.startswith(plm_prefix):
                olm_name = plm_name.replace(plm_prefix, olm_prefix)
                if olm_name in olm_state:
                    mapping_list.append((plm_name, olm_name))
                break

    # 保存映射表逻辑...
    return mapping_list

def match_parameters(model_orig, model_finetuned, output_dir, component="llm"):
    print("正在匹配参数...")
    # 获取两个模型的状态字典
    orig_state_dict = model_orig.state_dict()
    # 处理 DataParallel 模型的状态字典
    if hasattr(model_finetuned, 'module'):
        finetuned_state_dict = model_finetuned.module.state_dict()
    else:
        finetuned_state_dict = model_finetuned.state_dict()

    print(f"原始模型参数数量: {len(orig_state_dict)}, 微调模型参数数量: {len(finetuned_state_dict)}")
    show_model_params(model_orig, os.path.join(output_dir, "model_orig.txt"))
    show_model_params(model_finetuned, os.path.join(output_dir, "model_finetuned.txt"))

    mapping_list = weight_mapping(model_orig, model_finetuned, 
                                 os.path.join(output_dir, f"mapping_{component}.txt"), 
                                 component=component)

    return mapping_list

def load_prismatic_vlm(model_gpu_id=1):
    """使用prismatic库加载底层VLM，支持CPU或GPU，优先本地"""
    device_str = f"GPU:{model_gpu_id}" if model_gpu_id != "cpu" else "CPU"
    print(f"正在使用prismatic库加载底层VLM: prism-dinosiglip-224px+7b 到 {device_str}...")

    model_id = "prism-dinosiglip-224px+7b"
    local_dir = f"/home/intern/zhangfengnian/checkpoints/{model_id}"

    try:
        # 尝试加载HF令牌(如果有)
        try:
            with open(".hf_token", "r") as f:
                hf_token = f.read().strip()
        except:
            hf_token = None
            print("未找到HF令牌文件，尝试无令牌加载")

        vlm = load(model_id, hf_token=hf_token)


        # 移动到指定设备
        if model_gpu_id == "cpu":
            vlm.to("cpu")
        else:
            vlm.to(f"cuda:{model_gpu_id}", dtype=torch.bfloat16)

        print(f"成功加载 {model_id} 到 {device_str}!")
        return vlm

    except Exception as e:
        print(f"通过prismatic库加载模型失败: {e}")
        print("尝试备选方法...")
        return load_siglip_fallback(model_gpu_id)

def load_siglip_fallback(model_gpu_id=1):
    """备选方法：使用subfolder参数加载子目录中的模型，支持CPU或GPU"""
    device_str = f"GPU:{model_gpu_id}" if model_gpu_id != "cpu" else "CPU"
    print(f"尝试使用备选方法加载模型: siglip-224px+7b 到 {device_str}...")
    
    try:
        config = AutoConfig.from_pretrained("google/siglip-base-patch16-224")
        config.model_type = "siglip"
        model = AutoModel.from_pretrained(
            "TRI-ML/prismatic-vlms",
            subfolder="prism-dinosiglip-224px+7b",
            config=config,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        if model_gpu_id == "cpu":
            model.to("cpu")
        else:
            model.to(f"cuda:{model_gpu_id}")
            
        return model
    except Exception as e:
        print(f"备选方法加载失败: {e}")
        print("尝试加载官方SigLIP模型作为替代...")
        
        try:
            # 不使用device_map='auto'参数
            model = AutoModel.from_pretrained(
                "google/siglip-base-patch16-224",
                torch_dtype=torch.float16
            )
            
            if model_gpu_id == "cpu":
                model.to("cpu")
            else:
                model.to(f"cuda:{model_gpu_id}")
                
            return model
        except Exception as e2:
            print(f"加载官方SigLIP模型也失败: {e2}")
            
            # 最后尝试
            try:
                print("尝试加载其他官方视觉模型作为替代...")
                model = AutoModel.from_pretrained(
                    "openai/clip-vit-base-patch16",
                    torch_dtype=torch.float16
                )
                
                if model_gpu_id == "cpu":
                    model.to("cpu")
                else:
                    model.to(f"cuda:{model_gpu_id}")
                    
                return model
            except Exception as e3:
                print(f"所有视觉模型加载尝试都失败: {e3}")
                return None

def load_openvla(model_gpu_id=1):
    """加载openvla，支持CPU或GPU"""

    
    device_str = f"GPU:{model_gpu_id}" if model_gpu_id != "cpu" else "CPU"
    print(f"正在加载模型：openvla 到 {device_str}")
    
    # 首先释放显存缓存
    torch.cuda.empty_cache()

    # 设置环境变量以帮助内存管理
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    try:
        # 根据不同设备选择加载策略
        if model_gpu_id == "cpu":
            print("警告: 在CPU上加载大型模型可能会很慢且内存占用较大")
            model = AutoModelForVision2Seq.from_pretrained(
                "openvla/openvla-7b-finetuned-libero-spatial",
                torch_dtype=torch.float32,  # CPU上使用float32
                low_cpu_mem_usage=True,
                trust_remote_code=True
            ).to("cpu")
        else:
            # 尝试直接加载到指定GPU，不使用flash_attention_2
            model = AutoModelForVision2Seq.from_pretrained(
                "openvla/openvla-7b-finetuned-libero-spatial",
                torch_dtype=torch.bfloat16, 
                low_cpu_mem_usage=True, 
                trust_remote_code=True
            ).to(f"cuda:{model_gpu_id}")
            
            # 使用 DataParallel 将模型分布到指定GPU
            model = torch.nn.DataParallel(model, device_ids=[model_gpu_id])
            
        return model
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print(f"GPU:{model_gpu_id} 内存不足，尝试在CPU上加载")
            # 在CPU上加载
            try:
                model = AutoModelForVision2Seq.from_pretrained(
                    "openvla/openvla-7b-finetuned-libero-spatial",
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                ).to("cpu")
                return model
            except Exception as e2:
                print(f"在CPU上加载OpenVLA模型也失败: {e2}")
                return None
        else:
            print(f"加载OpenVLA模型时出错: {e}")
            return None


def save_model_weights_info(model, filename):
    """
    将模型所有权重参数的名称与尺寸保存到文本文件
    :param model: torch.nn.Module
    :param filename: 输出文件名
    """
    if model is None:
        print(f"模型为 None，跳过保存 {filename}")
        return

    lines = []
    for name, param in model.named_parameters():
        lines.append(f"{name}  {tuple(param.shape)}")

    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"已保存模型权重信息到 {filename}，共 {len(lines)} 条参数。")





