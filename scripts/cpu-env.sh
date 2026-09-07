#!/usr/bin/env bash
# Source before starting Ollama on CPU-only / AMD iGPU machines:
#   source scripts/cpu-env.sh
#   ollama serve
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-60m}"
export OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-2048}"
export OLLAMA_KV_CACHE_TYPE="${OLLAMA_KV_CACHE_TYPE:-q8_0}"
export OLLAMA_MAX_LOADED_MODELS="${OLLAMA_MAX_LOADED_MODELS:-1}"
export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:--1}"
echo "Ollama CPU env applied: keep_alive=$OLLAMA_KEEP_ALIVE ctx=$OLLAMA_CONTEXT_LENGTH"
