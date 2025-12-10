#!/bin/bash
# Script to build Whisper TensorRT engine for TensorRT-LLM v0.18.2+
# Usage: bash build_whisper_tensorrt.sh [model_name] [quantization] [output_base_dir]
# Example: bash build_whisper_tensorrt.sh large-v3 int8 /app/models

set -e

# Configuration
MODEL_NAME=${1:-"large-v3"}
QUANTIZATION=${2:-"int8"}
OUTPUT_BASE_DIR=${3:-"/app/models"}
TENSORRT_LLM_PATH="/app/TensorRT-LLM-examples/examples/whisper"

INFERENCE_PRECISION="float16"
MAX_BEAM_WIDTH=4
MAX_BATCH_SIZE=8

echo "=========================================="
echo "Building Whisper TensorRT Engine"
echo "=========================================="
echo "Model Name: $MODEL_NAME"
echo "Quantization: $QUANTIZATION"
echo "Output Directory: $OUTPUT_BASE_DIR"
echo "=========================================="

cd $TENSORRT_LLM_PATH

# Step 1: Download assets if not exist
echo "[1/5] Downloading assets..."
mkdir -p assets

if [ ! -f "assets/multilingual.tiktoken" ]; then
    wget --directory-prefix=assets https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/multilingual.tiktoken
fi

if [ ! -f "assets/mel_filters.npz" ]; then
    wget --directory-prefix=assets https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/mel_filters.npz
fi

# Step 2: Download model weights
echo "[2/5] Downloading model weights..."
MODEL_URL=""
case $MODEL_NAME in
    "tiny")
        MODEL_URL="https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt"
        ;;
    "base")
        MODEL_URL="https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt"
        ;;
    "small")
        MODEL_URL="https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt"
        ;;
    "medium")
        MODEL_URL="https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt"
        ;;
    "large-v2")
        MODEL_URL="https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt"
        ;;
    "large-v3")
        MODEL_URL="https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt"
        ;;
    *)
        echo "Unknown model: $MODEL_NAME"
        exit 1
        ;;
esac

if [ ! -f "assets/${MODEL_NAME}.pt" ]; then
    wget --directory-prefix=assets -O assets/${MODEL_NAME}.pt $MODEL_URL
fi

# Step 3: Convert checkpoint
echo "[3/5] Converting checkpoint to TensorRT-LLM format..."
checkpoint_dir="whisper_${MODEL_NAME//-/_}_weights_${QUANTIZATION}"

# convert_checkpoint.py expects model in assets/ directory
if [ "$QUANTIZATION" == "float16" ]; then
    python3 convert_checkpoint.py \
        --output_dir $checkpoint_dir
else
    python3 convert_checkpoint.py \
        --use_weight_only \
        --weight_only_precision $QUANTIZATION \
        --output_dir $checkpoint_dir
fi

# Step 4: Build TensorRT engines
echo "[4/5] Building TensorRT engines (this will take 10-20 minutes)..."
output_dir="${OUTPUT_BASE_DIR}/whisper_${MODEL_NAME//-/_}_${QUANTIZATION}"

echo "Building encoder..."
trtllm-build --checkpoint_dir ${checkpoint_dir}/encoder \
              --output_dir ${output_dir}/encoder \
              --moe_plugin disable \
              --max_batch_size ${MAX_BATCH_SIZE} \
              --gemm_plugin disable \
              --bert_attention_plugin ${INFERENCE_PRECISION} \
              --max_input_len 3000 --max_seq_len=3000

echo "Building decoder..."
trtllm-build --checkpoint_dir ${checkpoint_dir}/decoder \
              --output_dir ${output_dir}/decoder \
              --moe_plugin disable \
              --max_beam_width ${MAX_BEAM_WIDTH} \
              --max_batch_size ${MAX_BATCH_SIZE} \
              --max_seq_len 114 \
              --max_input_len 14 \
              --max_encoder_input_len 3000 \
              --gemm_plugin ${INFERENCE_PRECISION} \
              --bert_attention_plugin ${INFERENCE_PRECISION} \
              --gpt_attention_plugin ${INFERENCE_PRECISION}

# Step 5: Copy assets to app directory
echo "[5/5] Copying assets..."
cp -r assets /app/

echo "=========================================="
echo "✅ TensorRT engine built successfully!"
echo "=========================================="
echo "Model: $MODEL_NAME"
echo "Quantization: $QUANTIZATION"
echo "Engine path: $output_dir"
echo ""
echo "Encoder: $output_dir/encoder/"
echo "Decoder: $output_dir/decoder/"
echo ""
echo "To use this engine, set environment variable:"
echo "export TRT_MODEL_PATH=$output_dir"
echo "=========================================="
