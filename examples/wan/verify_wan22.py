import os
import torch
from lightx2v import LightX2VPipeline

# Configuration paths
MODEL_PATH = "/home/martinelcheikh/LightX2V/models/Wan2.2-I2V-A14B"
LOW_NOISE_CKPT = os.path.join(MODEL_PATH, "low_noise_model/diffusion_pytorch_model.safetensors")
HIGH_NOISE_CKPT = os.path.join(MODEL_PATH, "high_noise_model/diffusion_pytorch_model.safetensors")
IMAGE_PATH = "/home/martinelcheikh/LightX2V/assets/img_lightx2v.png"
SAVE_PATH = "/home/martinelcheikh/LightX2V/save_results/output_verify.mp4"

# Ensure save directory exists
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

print(f"Initializing pipeline with model_path: {MODEL_PATH}")

pipe = LightX2VPipeline(
    model_path=MODEL_PATH,
    model_cls="wan2.2_moe_distill",
    task="i2v",
    low_noise_original_ckpt=LOW_NOISE_CKPT,
    high_noise_original_ckpt=HIGH_NOISE_CKPT,
)

# Enable offloading (Optimized for 32GB RAM / 24GB VRAM)
pipe.enable_offload(
    cpu_offload=True,
    offload_granularity="model", # Using model granularity for lazy-loading stability
    text_encoder_offload=True,
    image_encoder_offload=False,
    vae_offload=False,
)

# Create generator
pipe.create_generator(
    attn_mode="torch_sdpa",
    infer_steps=4,
    height=480,
    width=832,
    num_frames=81,
    guidance_scale=1.0,
    sample_shift=5.0,
)

prompt = "A cinematic video of a futuristic city with neon lights, high quality, 4k."
negative_prompt = "low quality, blurry, static, distorted"

print("Starting video generation...")
pipe.generate(
    seed=42,
    image_path=IMAGE_PATH,
    prompt=prompt,
    negative_prompt=negative_prompt,
    save_result_path=SAVE_PATH,
)

print(f"Generation complete! Result saved to: {SAVE_PATH}")
