import torch
from collections import defaultdict
from pathlib import Path


# ====== PaLiGemma (VLM) ======
from transformers import AutoModel


########################################
# Global output directory
########################################

OUTPUT_DIR = Path("/home/intern/zhangfengnian/workspace/vlm_vla/paligemma_pi0.5/outputs_pi05_libero")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


########################################
# Utils
########################################

def dump_model_params_from_state_dict(state_dict, save_path):
    save_path = OUTPUT_DIR / save_path

    total = 0
    with save_path.open("w") as f:
        for name, tensor in state_dict.items():
            n = tensor.numel()
            total += n
            f.write(
                f"{name}\tshape={tuple(tensor.shape)}\t"
                f"dtype={tensor.dtype}\tnumel={n}\n"
            )

    print(f"[OK] Saved params to {save_path}")
    print(f"[INFO] Total params: {total/1e6:.2f}M")

    return total



def collect_top_level_stats_from_state_dict(state_dict):
    stats = defaultdict(int)
    for name, tensor in state_dict.items():
        top = name.split(".")[0]
        stats[top] += tensor.numel()
    return stats



def print_stats(title, stats):
    print(f"\n===== {title} =====")
    for k, v in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"{k:30s}: {v/1e6:6.2f} M")


########################################
# 1. Load pi05_libero (PyTorch VLA)
########################################

print("\n=== Loading pi05_libero (PyTorch VLA) ===")

from safetensors.torch import load_file

PI05_LIBERO_PT_DIR = Path(
    "/home/intern/zhangfengnian/checkpoints/pi05_libero_pytorch"
)

# 1. load state_dict
libero_state_dict = load_file(
    PI05_LIBERO_PT_DIR / "model.safetensors"
)

print(f"[OK] Loaded pi05_libero PyTorch state_dict")
print(f"[INFO] #params tensors = {len(libero_state_dict)}")

# 2. dump & stats
libero_total = dump_model_params_from_state_dict(
    libero_state_dict,
    "pi05_libero_params.txt"
)

libero_stats = collect_top_level_stats_from_state_dict(libero_state_dict)
print_stats("pi05_libero top-level param stats", libero_stats)

########################################
# 2. Load PaLiGemma (VLM)
########################################

print("\n=== Loading PaLiGemma (VLM) ===")

paligemma_model_name = "google/paligemma-3b-mix-224"

paligemma_model = AutoModel.from_pretrained(
    paligemma_model_name,
    torch_dtype=torch.float16,
)

paligemma_state_dict = paligemma_model.state_dict()

paligemma_total = dump_model_params_from_state_dict(
    paligemma_state_dict,
    "paligemma_params.txt"
)

paligemma_stats = collect_top_level_stats_from_state_dict(paligemma_state_dict)
print_stats("PaLiGemma top-level param stats", paligemma_stats)



########################################
# 3. Parameter name matching (VLM vs VLA)
########################################

print("\n=== Matching parameters by name ===")

libero_param_names = set(libero_state_dict.keys())
paligemma_param_names = set(paligemma_state_dict.keys())

common = libero_param_names & paligemma_param_names
only_libero = libero_param_names - paligemma_param_names
only_pali = paligemma_param_names - libero_param_names

print(f"Common params      : {len(common)}")
print(f"Only in pi05_libero: {len(only_libero)}")
print(f"Only in PaLiGemma  : {len(only_pali)}")

with (OUTPUT_DIR / "param_name_matching.txt").open("w") as f:
    f.write("=== Common parameters ===\n")
    for name in sorted(common):
        f.write(name + "\n")

    f.write("\n=== Only in pi05_libero ===\n")
    for name in sorted(only_libero):
        f.write(name + "\n")

    f.write("\n=== Only in PaLiGemma ===\n")
    for name in sorted(only_pali):
        f.write(name + "\n")

print("[OK] Saved parameter name matching to param_name_matching.txt")


########################################
# 4. Weight-level matching (VLM backbone)
########################################

def is_valid_weight(name: str):
    if not name.endswith(".weight"):
        return False
    if "layer_norm" in name or "layernorm" in name:
        return False
    return True


def canonicalize_libero_name(name: str):
    prefix = "paligemma_with_expert.paligemma.model."
    if not name.startswith(prefix):
        return None
    core = name[len(prefix):]
    if core.startswith(("vision_tower.", "language_model.", "multi_modal_projector.")):
        return core
    return None


def match_vlm_weights(
    libero_sd,
    pali_sd,
    save_path="vlm_vla_weight_matching.txt"
):
    save_path = OUTPUT_DIR / save_path

    matched, shape_mismatch, missing = [], [], []

    for vla_name, vla_tensor in libero_sd.items():

        if not is_valid_weight(vla_name):
            continue

        vlm_name = canonicalize_libero_name(vla_name)
        if vlm_name is None:
            continue

        if vlm_name not in pali_sd:
            missing.append((vlm_name, vla_name))
            continue

        vlm_tensor = pali_sd[vlm_name]

        if vla_tensor.shape == vlm_tensor.shape:
            matched.append((vlm_name, vla_name, vla_tensor.numel()))
        else:
            shape_mismatch.append(
                (vlm_name, vla_name, vlm_tensor.shape, vla_tensor.shape)
            )

    with save_path.open("w") as f:
        for vlm, vla, n in matched:
            f.write(f"{vlm}\t{vla}\tnumel={n}\n")

        f.write("\n=== Shape mismatch ===\n")
        for x in shape_mismatch:
            f.write(str(x) + "\n")

        f.write("\n=== Missing ===\n")
        for x in missing:
            f.write(str(x) + "\n")

    print("\n=== VLM ↔ VLA WEIGHT Matching Summary ===")
    print(f"Matched weights : {len(matched)}")
    print(f"Shape mismatch  : {len(shape_mismatch)}")
    print(f"Missing         : {len(missing)}")
    print(f"[OK] Saved to {save_path}")

    return matched, shape_mismatch, missing

matched, shape_mismatch, missing = match_vlm_weights(
    libero_state_dict,
    paligemma_state_dict
)

