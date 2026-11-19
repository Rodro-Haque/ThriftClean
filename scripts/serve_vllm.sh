#!/usr/bin/env bash
# Serve Qwen3-8B with vLLM (RTX 3090) from the separate .venv-vllm. Run from anywhere; blocks until killed.
#   one-time setup:  python3 -m venv .venv-vllm && .venv-vllm/bin/pip install vllm
#   then, in the shell that runs the pipeline:  export LLM_BASE_URL=http://localhost:8000/v1 LLM_MODEL=Qwen/Qwen3-8B
# The Jellyfish-8B baseline needs a second serve with that model name instead (MODEL=NECOUDBFM/Jellyfish-8B scripts/serve_vllm.sh).
set -euo pipefail
cd "$(dirname "$0")/.."

CUDA_HOME=$(pwd)/.venv-vllm/lib/python3.12/site-packages/nvidia/cu13 \
PATH="$(pwd)/.venv-vllm/bin:$(pwd)/.venv-vllm/lib/python3.12/site-packages/nvidia/cu13/bin:$PATH" \
HF_HOME=~/.cache/huggingface \
VLLM_USE_FLASHINFER_SAMPLER=0 \
.venv-vllm/bin/vllm serve "${MODEL:-Qwen/Qwen3-8B}" --dtype bfloat16 --max-model-len 4096 \
    --gpu-memory-utilization 0.90 --port 8000
