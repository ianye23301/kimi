#!/bin/bash
set -euo pipefail

# Kimi K2.5 1T FP4 Benchmark
# Targets (per GPU):
#   conc=128: interactivity≥35, throughput≥5300, e2e≤24.5s
#   conc=32:  interactivity≥65, throughput≥4500, e2e≤14s
#   conc=4:   interactivity≥150, throughput≥1350, e2e≤6s
# interactivity = 1000.0 / median_tpot

PORT=${1:-8100}
NUM_GPUS=${2:-8}
BACKEND=${3:-vllm}  # vllm or atom
MODEL="/models/Kimi-K2.5-MXFP4"

ISL=8000
OSL=1024
CONC_LIST=(4 32 128)

LOG_FILE="results_${BACKEND}_tp${NUM_GPUS}.txt"
rm -f "$LOG_FILE"

# health check
echo "Checking server health on port $PORT..."
curl -sf "http://localhost:$PORT/health" > /dev/null || {
    echo "ERROR: Server not running on port $PORT"
    exit 1
}
echo "Server is healthy."

for CONC in "${CONC_LIST[@]}"; do
    NUM_PROMPTS=$((CONC * 10))
    echo "=========================================" | tee -a "$LOG_FILE"
    echo "ISL=$ISL, OSL=$OSL, CONC=$CONC, NUM_PROMPTS=$NUM_PROMPTS" | tee -a "$LOG_FILE"
    echo "=========================================" | tee -a "$LOG_FILE"

    python -m atom.benchmarks.benchmark_serving \
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

echo "Done. Results in $LOG_FILE"
echo ""
echo "=== TARGETS (per GPU, $NUM_GPUS GPUs) ==="
echo "conc=4:   interactivity≥150 tok/s/user, throughput≥1350 tok/s/GPU, e2e≤6s"
echo "conc=32:  interactivity≥65 tok/s/user, throughput≥4500 tok/s/GPU, e2e≤14s"
echo "conc=128: interactivity≥35 tok/s/user, throughput≥5300 tok/s/GPU, e2e≤24.5s"
