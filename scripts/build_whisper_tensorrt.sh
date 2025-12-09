#!/bin/bash
# Script to build Whisper TensorRT engine
# Usage: bash build_whisper_tensorrt.sh <tensorrt_llm_path> <model_name> [quantization]
# Example: bash build_whisper_tensorrt.sh /app/TensorRT-LLM-examples large-v3 float16

set -e

TENSORRT_LLM_PATH=${1:-"/app/TensorRT-LLM-examples"}
MODEL_NAME=${2:-"large-v3"}
QUANTIZATION=${3:-"float16"}

echo "=================================="
echo "Building Whisper TensorRT Engine"
echo "=================================="
echo "TensorRT-LLM Path: $TENSORRT_LLM_PATH"
echo "Model Name: $MODEL_NAME"
echo "Quantization: $QUANTIZATION"
echo "=================================="

cd $TENSORRT_LLM_PATH/examples/whisper

# Download model if not exists
if [ ! -d "assets" ]; then
    echo "Downloading Whisper model assets..."
    python3 download_model.py --model $MODEL_NAME
fi

# Build TensorRT engine
echo "Building TensorRT engine..."
OUTPUT_DIR="whisper_${MODEL_NAME//-/_}_${QUANTIZATION}"

if [ "$QUANTIZATION" == "int8" ]; then
    python3 build.py --model_name $MODEL_NAME \
                     --output_dir $OUTPUT_DIR \
                     --use_gpt_attention_plugin \
                     --use_gemm_plugin \
                     --enable_context_fmha \
                     --weight_only_precision int8
elif [ "$QUANTIZATION" == "int4" ]; then
    python3 build.py --model_name $MODEL_NAME \
                     --output_dir $OUTPUT_DIR \
                     --use_gpt_attention_plugin \
                     --use_gemm_plugin \
                     --enable_context_fmha \
                     --weight_only_precision int4
else
    # float16 (default)
    python3 build.py --model_name $MODEL_NAME \
                     --output_dir $OUTPUT_DIR \
                     --use_gpt_attention_plugin \
                     --use_gemm_plugin \
                     --enable_context_fmha
fi

echo "=================================="
echo "TensorRT engine built successfully!"
echo "Model path: $(pwd)/$OUTPUT_DIR"
echo "=================================="
echo ""
echo "To use this engine, set TRT_MODEL_PATH environment variable:"
echo "export TRT_MODEL_PATH=$(pwd)/$OUTPUT_DIR"
