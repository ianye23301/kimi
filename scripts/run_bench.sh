#!/bin/bash
set -euo pipefail

PORT=${1:-8000}
NUM_GPUS=${2:-4}
TAG=${3:-"test"}
BENCH_MODULE=${4:-"atom.benchmarks.benchmark_serving"}

MODEL="/models/Kimi-K2.5-MXFP4"
ISL=8000
OSL=1024

LOG_FILE="/tmp/results_${TAG}.txt"
rm -f "$LOG_FILE"

curl -sf "http://localhost:$PORT/health" > /dev/null || { echo "Server not running on port $PORT"; exit 1; }
echo "Server healthy on port $PORT" | tee -a "$LOG_FILE"

for CONC in 4 32 128; do
    NUM_PROMPTS=$((CONC * 10))
    echo "=========================================" | tee -a "$LOG_FILE"
    echo "CONC=$CONC  ISL=$ISL  OSL=$OSL  PROMPTS=$NUM_PROMPTS  GPUS=$NUM_GPUS" | tee -a "$LOG_FILE"
    echo "=========================================" | tee -a "$LOG_FILE"

    python -m "$BENCH_MODULE" \
        --model="$MODEL" \
        --backend=vllm \
        --base-url="http://localhost:$PORT" \
        --trust-remote-code \
        --dataset-name=random \
        --random-input-len="$ISL" \
        --random-output-len="$OSL" \
        --random-range-ratio 0.8 \
        --num-prompts="$NUM_PROMPTS" \
        --max-concurrency="$CONC" \
        --request-rate=inf \
        --ignore-eos \
        --percentile-metrics="ttft,tpot,itl,e2el" 2>&1 | tee -a "$LOG_FILE"

    echo -e "\n" | tee -a "$LOG_FILE"
done

echo "=== DONE: $TAG ===" | tee -a "$LOG_FILE"
