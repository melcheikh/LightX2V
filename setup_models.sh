#!/bin/bash
# Script de descarga automática de modelos para LightX2V
# Uso: bash setup_models.sh

set -e  # Detener si hay error

echo "============================================"
echo "  LightX2V - Descarga Automática de Modelos"
echo "============================================"
echo ""

# Verificar que huggingface-cli está instalado
if ! command -v huggingface-cli &> /dev/null; then
    echo "❌ ERROR: huggingface-cli no está instalado"
    echo "Instalalo con: pip install huggingface_hub"
    exit 1
fi

# Crear directorio de modelos
mkdir -p models

# ---------------------------------------------------------
# Wan2.2-I2V-A14B Distill FP8 (modelos actualmente funcionando)
# ---------------------------------------------------------
echo "📦 Descargando Wan2.2-I2V-A14B modelos distill FP8..."
mkdir -p models/Wan2.2-I2V/{high_noise_model,low_noise_model,models_t5,models_clip,google/umt5-xxl}

# High noise FP8 (15 GB)
huggingface-cli download lightx2v/Wan2.2-Distill-Models \
    wan2.2_i2v_A14b_high_noise_scaled_fp8_e4m3_lightx2v_4step.safetensors \
    --local-dir models/Wan2.2-I2V/high_noise_model

# Low noise FP8 (15 GB)
huggingface-cli download lightx2v/Wan2.2-Distill-Models \
    wan2.2_i2v_A14b_low_noise_scaled_fp8_e4m3_lightx2v_4step.safetensors \
    --local-dir models/Wan2.2-I2V/low_noise_model

# T5 encoder (11 GB)
huggingface-cli download Wan-AI/Wan2.2-I2V-A14B \
    models_t5_umt5-xxl-enc-bf16.pth \
    --local-dir models/Wan2.2-I2V/models_t5

# CLIP encoder (4.7 GB) - Opcional, se puede deshabilitar
# huggingface-cli download Wan-AI/Wan2.2-I2V-A14B \
#     models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
#     --local-dir models/Wan2.2-I2V/models_clip

# VAE (2.8 GB)
huggingface-cli download Wan-AI/Wan2.2-I2V-A14B \
    Wan2.1_VAE.pth \
    --local-dir models/Wan2.2-I2V

# Tokenizer T5
huggingface-cli download Wan-AI/Wan2.2-I2V-A14B \
    --include "google/umt5-xxl/*" \
    --local-dir models/Wan2.2-I2V

# Config.json (ya está modificado en el repo, este es para referencia)
# huggingface-cli download Wan-AI/Wan2.2-I2V-A14B \
#     configuration.json \
#     --local-dir models/Wan2.2-I2V

# ---------------------------------------------------------
# OPCIÓN: Wan2.1-I2V-14B Distill FP8 (Videos más largos, 10-12s)
# ---------------------------------------------------------
# echo "📦 Descargando Wan2.1-I2V-14B modelo único (Videos 10-12s)..."
# mkdir -p models/Wan2.1-I2V/{models_t5,models_clip,google/umt5-xxl}
# huggingface-cli download lightx2v/Wan2.1-Distill-Models \
#     wan2.1_i2v_720p_scaled_fp8_e4m3_lightx2v_4step.safetensors \
#     --local-dir models/Wan2.1-I2V
# # (T5, VAE y Tokenizers son compartidos o similares, consultar docs)

echo ""
echo "✅ Descarga completa!"
echo ""
echo "📁 Modelos descargados en: $(pwd)/models/Wan2.2-I2V"
echo "💾 Espacio total usado: ~45-50 GB"
echo ""
echo "🚀 Siguiente paso:"
echo "   conda activate lightx2v"
echo "   python examples/wan/run_wan22_i2v_distill_fp8.py"
echo ""
