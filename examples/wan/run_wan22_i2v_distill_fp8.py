"""
Wan2.2 Distill FP8 - Image-to-Video generation
RTX 5090 Laptop (24GB VRAM, 32GB RAM, i9)

Corrected script with explicit paths for all model components:
- FP8 quantized distill high/low noise models
- T5 text encoder (in non-standard subdirectory)
- CLIP image encoder (in non-standard subdirectory)  
- VAE encoder/decoder
"""

import os
from lightx2v import LightX2VPipeline

# ============================================================
# PATHS - All explicit, no guessing
# ============================================================
MODEL_ROOT = "/home/martinelcheikh/LightX2V/models/Wan2.2-I2V"

HIGH_NOISE_CKPT = os.path.join(
    MODEL_ROOT, "high_noise_model",
    "wan2.2_i2v_A14b_high_noise_scaled_fp8_e4m3_lightx2v_4step.safetensors"
)
LOW_NOISE_CKPT = os.path.join(
    MODEL_ROOT, "low_noise_model",
    "wan2.2_i2v_A14b_low_noise_scaled_fp8_e4m3_lightx2v_4step.safetensors"
)
T5_CKPT = os.path.join(
    MODEL_ROOT, "models_t5",
    "models_t5_umt5-xxl-enc-bf16.pth"
)
CLIP_CKPT = os.path.join(
    MODEL_ROOT, "models_clip",
    "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
)

IMAGE_PATH = "/home/martinelcheikh/LightX2V/assets/img_lightx2v.png"
SAVE_PATH = "/home/martinelcheikh/LightX2V/save_results/output_wan22_fp8.mp4"

# ============================================================
# VALIDATE ALL PATHS EXIST
# ============================================================
paths_to_check = {
    "Model root": MODEL_ROOT,
    "High noise FP8 ckpt": HIGH_NOISE_CKPT,
    "Low noise FP8 ckpt": LOW_NOISE_CKPT,
    "T5 encoder": T5_CKPT,
    "CLIP encoder": CLIP_CKPT,
    "Input image": IMAGE_PATH,
    "Google tokenizer": os.path.join(MODEL_ROOT, "google", "umt5-xxl"),
    "VAE": os.path.join(MODEL_ROOT, "Wan2.1_VAE.pth"),
}

print("=" * 60)
print("Validating paths...")
all_ok = True
for name, path in paths_to_check.items():
    exists = os.path.exists(path)
    status = "✓" if exists else "✗ MISSING"
    print(f"  {status} {name}: {path}")
    if not exists:
        all_ok = False

if not all_ok:
    print("\n✗ Some paths are missing! Fix them before running.")
    exit(1)

print("✓ All paths validated!")
print("=" * 60)

# ============================================================
# CREATE PIPELINE
# ============================================================
print("\n[1/5] Creating pipeline...")
pipe = LightX2VPipeline(
    model_path=MODEL_ROOT,
    model_cls="wan2.2_moe_distill",
    task="i2v",
)

# ============================================================
# ENABLE FP8 QUANTIZATION with explicit ckpt paths
# ============================================================
print("[2/5] Enabling FP8 quantization...")
pipe.enable_quantize(
    dit_quantized=True,
    quant_scheme="fp8-triton",
    high_noise_quantized_ckpt=HIGH_NOISE_CKPT,
    low_noise_quantized_ckpt=LOW_NOISE_CKPT,
)

# Set T5 and CLIP paths explicitly (they're in non-standard subdirs)
pipe.t5_original_ckpt = T5_CKPT
pipe.clip_original_ckpt = CLIP_CKPT

# ============================================================
# ENABLE OFFLOADING for 24GB VRAM
# ============================================================
print("[3/5] Enabling CPU offload...")
pipe.enable_offload(
    cpu_offload=True,
    offload_granularity="phase",      # Wan 2.2 MoE uses 'phase' offload
    text_encoder_offload=True,
    image_encoder_offload=False,
    vae_offload=False,
)

# ============================================================
# CREATE GENERATOR
# ============================================================
print("[4/5] Creating generator (this loads the models)...")
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

pipe.create_generator(
    attn_mode="torch_sdpa",       # Most compatible with RTX 5090
    infer_steps=4,                # Distilled model = 4 steps
    height=480,
    width=832,
    num_frames=81,                # ~5 seconds at 16fps
    guidance_scale=1.0,           # CFG disabled for distill
    sample_shift=5.0,
)

# ============================================================
# GENERATE VIDEO
# ============================================================
print("[5/5] Generating video...")
prompt = (
    "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. "
    "The fluffy-furred feline gazes directly at the camera with a relaxed expression. "
    "Blurred beach scenery forms the background featuring crystal-clear waters, "
    "distant green hills, and a blue sky dotted with white clouds."
)
negative_prompt = (
    "镜头晃动，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，"
    "静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
    "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)

pipe.generate(
    seed=42,
    image_path=IMAGE_PATH,
    prompt=prompt,
    negative_prompt=negative_prompt,
    save_result_path=SAVE_PATH,
)

print(f"\n✓ Video saved to: {SAVE_PATH}")
