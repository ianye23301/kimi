#!/bin/bash
set -euo pipefail

# Launch ATOM with all Kimi K2.5 optimizations enabled.
# Run inside the kimi-atom container.
#
# Usage:
#   bash kimi/scripts/run_optimized.sh [PORT] [TP]
#
# Environment variables (all default to "1" = enabled):
#   ATOM_MLA_TRITON_DECODE - Triton MLA decode for bs<=32 (no head padding)
#   ATOM_MLA_FUSED_KV_BMM  - Fuse k_up_proj + RoPE + KV cache store
#   ATOM_ENABLE_DS_INPUT_RMSNORM_QUANT_FUSION - Fused RMSNorm + quant
#   ATOM_ENABLE_ALLREDUCE_RMSNORM_FUSION - Fuse AllReduce into RMSNorm

PORT=${1:-8100}
TP=${2:-8}
MODEL="/models/Kimi-K2.5-MXFP4"

echo "=== ATOM Optimized Kimi K2.5 ==="
echo "Port: $PORT, TP: $TP"
echo "Optimizations:"
echo "  MLA Triton decode (bs<=32): ${ATOM_MLA_TRITON_DECODE:-1}"
echo "  Fused kv_bmm: ${ATOM_MLA_FUSED_KV_BMM:-1}"
echo "  Input RMSNorm+Quant fusion: ${ATOM_ENABLE_DS_INPUT_RMSNORM_QUANT_FUSION:-1}"
echo "  AllReduce+RMSNorm fusion: ${ATOM_ENABLE_ALLREDUCE_RMSNORM_FUSION:-1}"
echo ""

# Ensure optimizations are enabled
export ATOM_MLA_TRITON_DECODE=${ATOM_MLA_TRITON_DECODE:-1}
export ATOM_MLA_FUSED_KV_BMM=${ATOM_MLA_FUSED_KV_BMM:-1}
export ATOM_ENABLE_DS_INPUT_RMSNORM_QUANT_FUSION=${ATOM_ENABLE_DS_INPUT_RMSNORM_QUANT_FUSION:-1}
export ATOM_ENABLE_ALLREDUCE_RMSNORM_FUSION=${ATOM_ENABLE_ALLREDUCE_RMSNORM_FUSION:-1}

# Suppress noisy aiter logging
export AITER_LOG_LEVEL=WARNING

# Clear stale compile cache
rm -rf /root/.cache/atom/*

python -m atom.entrypoints.openai_server \
    --model "$MODEL" \
    --kv_cache_dtype fp8 \
    --trust_remote_code \
    -tp "$TP" \
    --port "$PORT" \
    --max_model_len 10000 \
    "$@"
