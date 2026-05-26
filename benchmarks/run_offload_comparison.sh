#!/usr/bin/env bash
# =============================================================================
# Full Offload Comparison Benchmark Runner
# =============================================================================
# Runs all four systems (DirectKV, Pie, Neo, FlexGen) across three models,
# collects latency-vs-rate (Figure 10) and long-context (Figure 11) data,
# and produces all plots.
#
# Prerequisites:
#   - SGLang installed with DirectKV and Pie backends available
#   - Neo and FlexGen repos cloned and configured (see comparison.md)
#   - H200 or GH200 GPU with ≥256GB CPU memory
#   - CUDA ≥12.1, PyTorch ≥2.1
#
# Usage:
#   bash benchmarks/run_offload_comparison.sh [OPTIONS]
#
# Options:
#   --models    "llama-3.1-8b opt-6.7b opt-30b" (space-separated)
#   --systems   "DirectKV Pie Neo FlexGen"
#   --rates     "5 10 15 20 25 30"
#   --seq-lens  "1024 2048 4096 8192"
#   --dry-run   Print commands without executing
#   --skip-fig10  Skip Experiment 1
#   --skip-fig11  Skip Experiment 2
# =============================================================================

set -euo pipefail

# Defaults
MODELS="${MODELS:-llama-3.1-8b}"
SYSTEMS="${SYSTEMS:-DirectKV Pie}"
RATES="${RATES:-5 10 15 20 25 30}"
SEQ_LENS="${SEQ_LENS:-1024 2048 4096 8192}"
NUM_REQUESTS="${NUM_REQUESTS:-1000}"
WARMUP="${WARMUP:-100}"
RUNS="${RUNS:-3}"
BASE_URL="${BASE_URL:-http://127.0.0.1:30000}"
RESULTS_DIR="${RESULTS_DIR:-results}"
DRY_RUN="${DRY_RUN:-false}"
SKIP_FIG10="${SKIP_FIG10:-false}"
SKIP_FIG11="${SKIP_FIG11:-false}"
SERVER_PORT="${SERVER_PORT:-30000}"

# Model ID mapping
declare -A MODEL_IDS=(
    ["llama-3.1-8b"]="meta-llama/Llama-3.1-8B"
    ["opt-6.7b"]="facebook/opt-6.7b"
    ["opt-30b"]="facebook/opt-30b"
)

# Parse CLI args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --models)   MODELS="$2"; shift 2;;
        --systems)  SYSTEMS="$2"; shift 2;;
        --rates)    RATES="$2"; shift 2;;
        --seq-lens) SEQ_LENS="$2"; shift 2;;
        --dry-run)  DRY_RUN=true; shift;;
        --skip-fig10) SKIP_FIG10=true; shift;;
        --skip-fig11) SKIP_FIG11=true; shift;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

mkdir -p "$RESULTS_DIR/plots"

# ---------------------------------------------------------------------------
# Helper: run a command or print it (dry-run)
# ---------------------------------------------------------------------------
run_cmd() {
    if [ "$DRY_RUN" = "true" ]; then
        echo "[dry-run] $*"
    else
        echo "[run] $*"
        eval "$@"
    fi
}

# ---------------------------------------------------------------------------
# Helper: launch SGLang server for a given system
# ---------------------------------------------------------------------------
launch_sglang_server() {
    local system="$1"
    local model_id="$2"

    local server_args="--model-path $model_id --tp 1 --port $SERVER_PORT"

    case "$system" in
        DirectKV)
            server_args="$server_args --attention-backend directkv --disable-radix-cache --disable-cuda-graph"
            ;;
        Pie)
            # Pie uses the standard attention backend with the Pie KV pool hook
            server_args="$server_args --attention-backend flashinfer --disable-radix-cache --disable-cuda-graph"
            ;;
        NoOffload)
            server_args="$server_args --attention-backend flashinfer"
            ;;
        Neo|FlexGen)
            echo "[WARN] $system requires its own server launcher — see comparison.md"
            echo "       Assuming server is already running at $BASE_URL"
            return 0
            ;;
    esac

    echo "Launching SGLang server: python -m sglang.launch_server $server_args"
    if [ "$DRY_RUN" = "false" ]; then
        python -m sglang.launch_server $server_args &
        SERVER_PID=$!
        echo "Server PID: $SERVER_PID"
        echo "Waiting for server to be ready..."
        sleep 30  # Allow server startup
        # Health check
        for i in $(seq 1 30); do
            if curl -s "$BASE_URL/health" > /dev/null 2>&1; then
                echo "Server is ready."
                return 0
            fi
            sleep 2
        done
        echo "[ERROR] Server failed to start within 90s"
        kill $SERVER_PID 2>/dev/null || true
        return 1
    fi
}

kill_server() {
    if [ -n "${SERVER_PID:-}" ]; then
        echo "Killing server PID $SERVER_PID"
        kill $SERVER_PID 2>/dev/null || true
        wait $SERVER_PID 2>/dev/null || true
        unset SERVER_PID
        sleep 5  # Let GPU memory free
    fi
}
trap kill_server EXIT

# ---------------------------------------------------------------------------
# Step 0: Record hardware info
# ---------------------------------------------------------------------------
echo "============================================================"
echo "Offload Comparison Benchmark"
echo "============================================================"
echo "Date: $(date)"
echo "Models: $MODELS"
echo "Systems: $SYSTEMS"
echo "Rates: $RATES"
echo "Seq lengths: $SEQ_LENS"
echo ""

if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader \
        > "$RESULTS_DIR/gpu_info.txt" 2>/dev/null || true
    head -1 "$RESULTS_DIR/gpu_info.txt"
fi
echo ""

# ---------------------------------------------------------------------------
# Experiment 1: Latency vs Request Rate (Figure 10)
# ---------------------------------------------------------------------------
if [ "$SKIP_FIG10" = "false" ]; then
    echo "============================================================"
    echo "Experiment 1: Latency vs Request Rate"
    echo "============================================================"

    FIG10_CSV="$RESULTS_DIR/fig10_latency_vs_rate.csv"
    rm -f "$FIG10_CSV"  # fresh start

    for model in $MODELS; do
        model_id="${MODEL_IDS[$model]:-$model}"
        for system in $SYSTEMS; do
            echo ""
            echo "--- $system / $model ---"

            launch_sglang_server "$system" "$model_id"

            run_cmd python benchmarks/bench_offload_serving.py \
                --system "$system" \
                --model-id "$model_id" \
                --model-label "$model" \
                --dataset sharegpt \
                --request-rates $RATES \
                --num-requests "$NUM_REQUESTS" \
                --warmup-requests "$WARMUP" \
                --runs "$RUNS" \
                --base-url "$BASE_URL" \
                --output-csv "$FIG10_CSV"

            kill_server
        done
    done
fi

# ---------------------------------------------------------------------------
# Experiment 2: Long-Context Performance (Figure 11)
# ---------------------------------------------------------------------------
if [ "$SKIP_FIG11" = "false" ]; then
    echo ""
    echo "============================================================"
    echo "Experiment 2: Long-Context Performance"
    echo "============================================================"

    FIG11_CSV="$RESULTS_DIR/fig11_longctx.csv"
    rm -f "$FIG11_CSV"

    # Use the largest model for long-context stress test
    LONGCTX_MODEL="${LONGCTX_MODEL:-opt-30b}"
    LONGCTX_MODEL_ID="${MODEL_IDS[$LONGCTX_MODEL]:-$LONGCTX_MODEL}"

    # Run NoOffload baseline first
    echo ""
    echo "--- NoOffload baseline / $LONGCTX_MODEL ---"
    launch_sglang_server "NoOffload" "$LONGCTX_MODEL_ID"

    run_cmd python benchmarks/bench_offload_longctx.py \
        --system NoOffload \
        --model-id "$LONGCTX_MODEL_ID" \
        --model-label "$LONGCTX_MODEL" \
        --seq-lengths $SEQ_LENS \
        --num-requests 200 \
        --warmup-requests 20 \
        --runs "$RUNS" \
        --base-url "$BASE_URL" \
        --output-csv "$FIG11_CSV"

    kill_server

    # Run each offload system
    for system in $SYSTEMS; do
        echo ""
        echo "--- $system / $LONGCTX_MODEL ---"
        launch_sglang_server "$system" "$LONGCTX_MODEL_ID"

        run_cmd python benchmarks/bench_offload_longctx.py \
            --system "$system" \
            --model-id "$LONGCTX_MODEL_ID" \
            --model-label "$LONGCTX_MODEL" \
            --seq-lengths $SEQ_LENS \
            --num-requests 200 \
            --warmup-requests 20 \
            --runs "$RUNS" \
            --base-url "$BASE_URL" \
            --output-csv "$FIG11_CSV"

        kill_server
    done
fi

# ---------------------------------------------------------------------------
# Step 3: Generate plots
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "Generating Plots"
echo "============================================================"

run_cmd python benchmarks/plot_offload_results.py \
    --fig10-csv "$RESULTS_DIR/fig10_latency_vs_rate.csv" \
    --fig11-csv "$RESULTS_DIR/fig11_longctx.csv" \
    --output-dir "$RESULTS_DIR/plots/"

echo ""
echo "============================================================"
echo "DONE. Results in $RESULTS_DIR/"
echo "============================================================"
echo "CSVs:"
ls -la "$RESULTS_DIR"/*.csv 2>/dev/null || echo "  (no CSVs yet — was this a dry run?)"
echo ""
echo "Plots:"
ls -la "$RESULTS_DIR/plots/"*.pdf 2>/dev/null || echo "  (no plots yet)"
