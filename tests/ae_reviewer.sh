#!/usr/bin/env bash
# =============================================================================
# ae_reviewer.sh — DirectKV Artifact Evaluation (four systems)
#
# Reproduces Figure 10 (latency vs request rate) and Figure 11 (long-context)
# for DirectKV, Pie, Neo, and FlexGen on an NVIDIA GH200 480GB (aarch64).
#
# Figure 10 panels:
#   (a) LLaMA-3-8B:  DirectKV, Pie, Neo          (sections 5a–5c)
#   (b) OPT-6.7B:   DirectKV, Pie, Neo, FlexGen  (sections 5d–5g)
#   (c) OPT-30B:    DirectKV, FlexGen, Neo        (sections 5h–5j)
#
# Figure 11 (long-context):
#   LLaMA-3-8B:  DirectKV, Pie                   (sections 6a–6b)
#   OPT-6.7B:    DirectKV, Pie, FlexGen           (sections 6c–6e)
#   OPT-30B:     DirectKV                         (section  6f)
#
# Usage
# -----
#   bash tests/ae_reviewer.sh              # full run (~2.5 hours)
#   bash tests/ae_reviewer.sh --quick      # reduced params (~25 min, for sanity check)
#   bash tests/ae_reviewer.sh --skip-neo   # skip Neo (if not needed)
#   bash tests/ae_reviewer.sh --skip-flexgen
#   bash tests/ae_reviewer.sh --skip-opt30b # skip OPT-30B sections
#   bash tests/ae_reviewer.sh --fig10-only  # skip Fig 11
#   bash tests/ae_reviewer.sh --neo-only    # run Neo sections only
#
# Requirements
# ------------
#   - All model weights present in weights/  (see baseline_ready_ae.md)
#   - numactl installed  (sudo apt-get install -y numactl)
#   - Python env with: sglang==0.4.9.post6, swiftllm, flexllmgen, fastapi, uvicorn
#   - export NO_PROXY=localhost,127.0.0.1  (handled below)
#
# Output
# ------
#   results/<timestamp>/
#     fig10_serving.csv        — latency vs request rate (all systems)
#     fig11_longctx.csv        — long-context latency + memory saving
#     plots/figure10.pdf
#     plots/figure11.pdf
#     logs/                    — per-server and per-bench logs
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# 0. Parse flags
# ---------------------------------------------------------------------------
QUICK=false
SKIP_NEO=false
SKIP_FLEXGEN=false
SKIP_OPT30B=false
FIG10_ONLY=false
SKIP_LLAMA=false
NEO_ONLY=false

for arg in "$@"; do
  case $arg in
    --quick)        QUICK=true ;;
    --skip-neo)     SKIP_NEO=true ;;
    --skip-flexgen) SKIP_FLEXGEN=true ;;
    --skip-opt30b)  SKIP_OPT30B=true ;;
    --fig10-only)   FIG10_ONLY=true ;;
    --skip-llama)   SKIP_LLAMA=true ;;
    --neo-only)     NEO_ONLY=true ;;
    *) echo "[WARN] Unknown flag: $arg" ;;
  esac
done

# --neo-only implies: skip DirectKV/Pie/FlexGen and Fig 11
if $NEO_ONLY; then
  SKIP_FLEXGEN=true
  FIG10_ONLY=true
fi

# ---------------------------------------------------------------------------
# 1. Paths and environment
# ---------------------------------------------------------------------------
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export NO_PROXY=localhost,127.0.0.1
export no_proxy=localhost,127.0.0.1

TIMESTAMP="$(date +%Y-%m-%d_%H-%M)"
RESULTS="$ROOT/results/$TIMESTAMP"
LOGS="$RESULTS/logs"
PLOTS="$RESULTS/plots"
mkdir -p "$RESULTS" "$LOGS" "$PLOTS"

# Model paths (local — no HF Hub download)
LLAMA="$ROOT/weights/Llama/Llama-3-8B"
OPT_HF="$ROOT/weights/opt/opt-6.7b-hf"
OPT_NP_PARENT="$ROOT/weights/opt"          # parent of opt-6.7b-np/ and opt-30b-np/
OPT30B_HF="$ROOT/weights/opt/opt-30b-hf"   # OPT-30B HF weights (DirectKV + SGLang)

NEO_DIR="$ROOT/baseline/neo"
LIB="$NEO_DIR/pacpu/prebuilt/libpacpu-llama3_8b-tp1.so"
LIB_OPT6B="$NEO_DIR/pacpu/prebuilt/libpacpu-opt_6_7b-tp1.so"
LIB_OPT30B="$NEO_DIR/pacpu/prebuilt/libpacpu-opt_30b-tp1.so"
PROFILE_DIR="$NEO_DIR/profile_results"
mkdir -p "$PROFILE_DIR"

FIG10_CSV="$RESULTS/fig10_serving.csv"
FIG11_CSV="$RESULTS/fig11_longctx.csv"

# Benchmark parameters
if $QUICK; then
  RATES="2 5 10 15"              # Fig 10: lower rates for long (4K+1K) sequences
  NREQ=20; WARMUP=2; RUNS=1
  SEQ_LENS="4096"
  LC_NREQ=100; LC_WARMUP=5; LC_RUNS=1
else
  RATES="2 5 10 15 20"           # Fig 10: 2–20 req/s for long sequences
  NREQ=50; WARMUP=10; RUNS=2
  SEQ_LENS="1024 2048 4096 8192"  # Fig 11 x-axis: 1k/2k/4k/8k
  LC_NREQ=100; LC_WARMUP=10; LC_RUNS=2
fi

# Synthetic prompt length — long-context high-pressure workload.
# Each LLaMA-3-8B request uses 4096 prefill + 1024 decode tokens (5120 peak KV).
# Neo --num-gpu-blocks-override is set to 512 (GPU pool = 8192 tokens ≈ 1–2 seqs),
# forcing ~14/16 concurrent sequences onto CPU.  Expected TPOT:
#   CPU bandwidth: 14 × 4608 × 128 KB / 512 GB/s ≈ 16 ms/token (Neo)
#   GPU bandwidth: 16 × 5120 × 128 KB / 3.35 TB/s ≈  3 ms/token (DirectKV)
# GH200 NVLink-C2C (512 GB/s CPU DRAM) is ~14× faster than A100+PCIe (32 GB/s),
# so paper values (1–3 s/token) are not reproducible; verify relative ordering:
#   DirectKV TPOT < Pie TPOT < Neo TPOT (with offloading active)
INPUT_LEN=4096     # Fig 10 LLaMA-3-8B synthetic input tokens
OUTPUT_LEN=1024    # Fig 10 LLaMA-3-8B output tokens (≤ 600 s at 16 ms TPOT)
OPT_INPUT_LEN=128  # OPT-6.7B / OPT-30B context=2048
OPT_OUTPUT_LEN=32  # OPT-6.7B output: 1024 in + 200 out = 1224 ≤ 2048
OPT_SEQ_LENS="1024" # OPT-6.7B max context=2048; seq_len=4096 would exceed it

# FlexGen serializes all requests (one at a time); firing many concurrent
# requests just piles them in a queue.  Keep counts small so each rate point
# finishes in minutes rather than hours.
FLEX_RATES="5 10 15 20 25 30"
FLEX_NREQ=12; FLEX_WARMUP=2; FLEX_RUNS=1

# Memory fractions — must cover model weights + KV pool.
# 8B/6.7B: 15 GB model out of 96 GB HBM → 0.50 gives ~33 GB KV pool, enough
#         for max-running-requests=16 at seq=5120 (640 MB/seq × 16 = 10 GB).
# 30B:    60 GB model out of 96 GB HBM → 0.85 gives a ~21 GB KV pool, enough
#         for max-running-requests=8 at seq=5120 (KV ≈ 1.4 MB/token × 5120 ≈ 7 GB).
MEM_FRAC_SGLANG=0.50
MEM_FRAC_SGLANG_30B=0.85

# Concurrency caps for the "hockey-stick" calibration (ae_final_result.md
# Change 4 / Tuning A). LLaMA-3-8B and OPT-6.7B share a single window;
# OPT-30B uses a tighter one because each request occupies 4× the KV.
SG_BS=16          # sglang cuda-graph-max-bs / max-running-requests (8B / 6.7B)
SG_BS_30B=8       # sglang cuda-graph-max-bs / max-running-requests (30B)
NEO_BS=16         # swiftllm max-batch-size                          (8B / 6.7B)
NEO_BS_30B=8     # swiftllm max-batch-size                          (30B)
NEO_TIB=8192      # swiftllm max-tokens-in-batch                     (8B / 6.7B)
NEO_TIB_30B=4096  # swiftllm max-tokens-in-batch                     (30B)
NEO_SWAP=20      # swiftllm swap-space (GB); peak CPU KV = 16×5120×128KB ≈ 11 GB

SGP=30000           # SGLang port (DirectKV, Pie)
NEO_PORT=8000       # Neo/swiftllm port
FLEX_PORT=30001     # FlexGen port

# ---------------------------------------------------------------------------
# 2. Helper functions
# ---------------------------------------------------------------------------

banner() { echo ""; echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; echo "  $*"; echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; }

wait_sglang() {
  local port="${1:-$SGP}" label="${2:-server}"
  echo "[wait] $label on port $port ..."
  for i in $(seq 1 120); do
    code=$(curl --noproxy '*' -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$port/health" 2>/dev/null)
    [ "$code" = "200" ] && echo "[wait] $label ready (${i}x5s)" && return 0
    [ $((i % 12)) -eq 0 ] && echo "[wait]   still loading ($((i*5))s)..."
    sleep 5
  done
  echo "[ERROR] Timeout waiting for $label"
  return 1
}

wait_neo() {
  local log="${1:-$LOGS/server_neo.log}"
  echo "[wait] Neo (swiftllm) — profiling CPU kernel (~2 min) ..."
  for i in $(seq 1 120); do
    grep -q "Started server process" "$log" 2>/dev/null && \
      echo "[wait] Neo ready (${i}x5s)" && return 0
    # Detect early crash (Traceback or AssertionError in log, process gone)
    if grep -q "Traceback\|AssertionError\|Error\|CUDA error" "$log" 2>/dev/null; then
      echo "[ERROR] Neo crashed during startup — see $log"
      return 1
    fi
    [ $((i % 12)) -eq 0 ] && echo "[wait]   still profiling ($((i*5))s)..."
    sleep 5
  done
  echo "[ERROR] Timeout waiting for Neo"
  return 1
}

wait_flexgen() {
  echo "[wait] FlexGen serve_flexgen.py on port $FLEX_PORT ..."
  for i in $(seq 1 120); do
    code=$(curl --noproxy '*' -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$FLEX_PORT/health" 2>/dev/null)
    [ "$code" = "200" ] && echo "[wait] FlexGen ready (${i}x5s)" && return 0
    [ $((i % 12)) -eq 0 ] && echo "[wait]   still loading ($((i*5))s)..."
    sleep 5
  done
  echo "[ERROR] Timeout waiting for FlexGen"
  return 1
}

stop_sglang() {
  pkill -f "sglang.launch_server" 2>/dev/null || true
  sleep 6
}

stop_neo() {
  pkill -f "swiftllm.server.api_server" 2>/dev/null || true
  pkill -9 -f "gcs_server"               2>/dev/null || true
  pkill -9 -f "raylet"                   2>/dev/null || true
  # Neo allocates up to 120-200 GB CPU swap; give the OS time to reclaim it
  sleep 15
  rm -rf /tmp/ray 2>/dev/null || true
}

stop_flexgen() {
  pkill -f "serve_flexgen" 2>/dev/null || true
  sleep 4
}

bench_sglang() {
  local system="$1" model_id="$2" model_label="$3" rates="$4" nreq="$5" warmup="$6" runs="$7" csv="$8" inp_len="${9:-$INPUT_LEN}" out_len="${10:-$OUTPUT_LEN}"
  echo "[bench] $system — rates: $rates  input=${inp_len}t output=${out_len}t"
  timeout 3600 python benchmarks/bench_offload_serving.py \
    --system "$system" \
    --model-id "$model_id" --model-label "$model_label" \
    --base-url "http://127.0.0.1:$SGP" --endpoint /generate \
    --dataset synthetic \
    --input-len "$inp_len" --output-len "$out_len" \
    --request-rates $rates \
    --num-requests "$nreq" --warmup-requests "$warmup" --runs "$runs" \
    --output-csv "$csv" \
    2>&1 | tee "$LOGS/bench_${system,,}.log"
}

# kill_hung_server: forcibly terminate a server if still running after timeout
kill_hung_server() {
  local label="$1"
  echo "[WARN] $label did not respond in time — killing all server processes"
  pkill -9 -f "sglang.launch_server" 2>/dev/null || true
  pkill -9 -f "swiftllm.server.api_server" 2>/dev/null || true
  pkill -9 -f "serve_flexgen" 2>/dev/null || true
  sleep 5
}

# Track pass/fail per system
declare -A STATUS

# ---------------------------------------------------------------------------
# 2b. Pre-run cleanup (clear any leftover servers from a previous run)
# ---------------------------------------------------------------------------
echo "[startup] Killing any leftover server / ray processes..."
pkill -9 -f "sglang.launch_server"        2>/dev/null || true
pkill -9 -f "swiftllm.server.api_server"  2>/dev/null || true
pkill -9 -f "serve_flexgen"               2>/dev/null || true
pkill -9 -f "gcs_server"                  2>/dev/null || true
pkill -9 -f "raylet"                      2>/dev/null || true
sleep 4
rm -rf /tmp/ray 2>/dev/null || true
echo "[startup] Done."

# ---------------------------------------------------------------------------
# 3. Verify prerequisites
# ---------------------------------------------------------------------------
banner "Step 0/5 — Prerequisite check"

check_weight() {
  local path="$1" label="$2"
  if [ -d "$path" ]; then echo "  [OK] $label: $path"
  else echo "  [MISSING] $label: $path" >&2; PREREQ_FAIL=true; fi
}

PREREQ_FAIL=false
check_weight "$LLAMA"    "LLaMA-3-8B"
check_weight "$OPT_HF"   "OPT-6.7B HF"
check_weight "$OPT_NP_PARENT/opt-6.7b-np" "OPT-6.7B NP"
# OPT-30B is optional: sections self-skip if weights are missing
[ -d "$OPT30B_HF" ] && \
  echo "  [OK] OPT-30B HF: $OPT30B_HF" || \
  echo "  [WARN] OPT-30B HF not found ($OPT30B_HF) — sections 5h–5j/6f will be skipped"
[ -d "$OPT_NP_PARENT/opt-30b-np" ] && \
  echo "  [OK] OPT-30B NP: $OPT_NP_PARENT/opt-30b-np" || \
  echo "  [WARN] OPT-30B NP not found — 5i (FlexGen OPT-30B) will be skipped"
[ -f "$LIB" ] && echo "  [OK] libpacpu ARM: $LIB" || \
  { echo "  [MISSING] libpacpu: $LIB"; PREREQ_FAIL=true; }
[ -f "$LIB_OPT6B" ] && echo "  [OK] libpacpu-opt_6_7b: $LIB_OPT6B" || \
  echo "  [WARN] libpacpu-opt_6_7b not found — Neo OPT-6.7B sections will be skipped"
[ -f "$LIB_OPT30B" ] && echo "  [OK] libpacpu-opt_30b: $LIB_OPT30B" || \
  echo "  [WARN] libpacpu-opt_30b not found — Neo OPT-30B sections will be skipped"

python -c "import sglang, swiftllm, flexllmgen, fastapi" 2>/dev/null && \
  echo "  [OK] Python packages: sglang swiftllm flexllmgen fastapi" || \
  { echo "  [FAIL] Missing Python packages"; PREREQ_FAIL=true; }

which numactl >/dev/null 2>&1 && echo "  [OK] numactl" || \
  { echo "  [WARN] numactl not found — Neo will fail. Install: sudo apt-get install -y numactl"; }

if $PREREQ_FAIL; then
  echo ""
  echo "[ABORT] Prerequisites missing. See above. Exiting."
  exit 1
fi

nvidia-smi --query-gpu=name,memory.total,driver_version \
  --format=csv,noheader 2>/dev/null | sed 's/^/  GPU: /'

echo ""
echo "  Results dir:  $RESULTS/"
echo "  Quick mode:   $QUICK"
echo "  Neo only:     $NEO_ONLY"
echo "  Skip Neo:     $SKIP_NEO"
echo "  Skip FlexGen: $SKIP_FLEXGEN"
echo "  Skip OPT-30B: $SKIP_OPT30B"
echo "  Skip Fig11:   $FIG10_ONLY"

# ---------------------------------------------------------------------------
# 4. Correctness tests
# ---------------------------------------------------------------------------
banner "Step 1/5 — Correctness tests"

python tests/test_directkv_correctness.py -v 2>&1 | tee "$LOGS/test_directkv.log" | tail -5
DKV_TESTS=$(grep -c "OK\|PASS\|passed" "$LOGS/test_directkv.log" 2>/dev/null || echo 0)

python tests/test_pie_offload.py -v 2>&1 | tee "$LOGS/test_pie.log" | tail -5
PIE_TESTS=$(grep -c "OK\|PASS\|passed" "$LOGS/test_pie.log" 2>/dev/null || echo 0)

echo "  DirectKV tests: see $LOGS/test_directkv.log"
echo "  Pie tests:      see $LOGS/test_pie.log"

# ---------------------------------------------------------------------------
# 5. Figure 10 — Latency vs Request Rate
# ---------------------------------------------------------------------------
banner "Step 2/5 — Figure 10: Serving Benchmark"
echo "  Rates (SGLang/Neo): $RATES req/s  |  Rates (FlexGen): $FLEX_RATES req/s"
echo "  Requests: $NREQ  Warmup: $WARMUP  Runs: $RUNS"
echo ""

# ============================================================
# Panel (a): LLaMA-3-8B — DirectKV, Pie, Neo
# ============================================================

if $SKIP_LLAMA; then
  banner "  5a-5c. LLaMA-3-8B panel — SKIPPED (--skip-llama)"
  STATUS[DirectKV]=SKIPPED; STATUS[Pie]=SKIPPED; STATUS[Neo]=SKIPPED
else

# ---- 5a. DirectKV LLaMA-3-8B -----------------------------------------------
if $NEO_ONLY; then
  banner "  5a. DirectKV LLaMA-3-8B — SKIPPED (--neo-only)"
  STATUS[DirectKV]=SKIPPED
else
banner "  5a. DirectKV (directkv, LLaMA-3-8B)"

python -m sglang.launch_server \
  --model-path "$LLAMA" --tp 1 --port $SGP \
  --attention-backend directkv-smpv2 \
  --mem-fraction-static $MEM_FRAC_SGLANG \
  --disable-radix-cache \
  --cuda-graph-max-bs $SG_BS \
  --max-running-requests $SG_BS \
  > "$LOGS/server_directkv.log" 2>&1 &

if wait_sglang $SGP "DirectKV"; then
  bench_sglang DirectKV "$LLAMA" llama-3.1-8b "$RATES" $NREQ $WARMUP $RUNS "$FIG10_CSV" && \
    STATUS[DirectKV]=PASS || STATUS[DirectKV]=FAIL
else
  kill_hung_server "DirectKV"
  STATUS[DirectKV]=TIMEOUT
fi
stop_sglang
fi

# ---- 5b. Pie LLaMA-3-8B ----------------------------------------------------
if $NEO_ONLY; then
  banner "  5b. Pie LLaMA-3-8B — SKIPPED (--neo-only)"
  STATUS[Pie]=SKIPPED
else
banner "  5b. Pie (flashinfer + KV offload, LLaMA-3-8B)"

python -m sglang.launch_server \
  --model-path "$LLAMA" --tp 1 --port $SGP \
  --attention-backend flashinfer \
  --mem-fraction-static $MEM_FRAC_SGLANG \
  --disable-radix-cache --disable-cuda-graph \
  --max-running-requests $SG_BS \
  --skip-server-warmup \
  > "$LOGS/server_pie.log" 2>&1 &

if wait_sglang $SGP "Pie"; then
  bench_sglang Pie "$LLAMA" llama-3.1-8b "$RATES" $NREQ $WARMUP $RUNS "$FIG10_CSV" && \
    STATUS[Pie]=PASS || STATUS[Pie]=FAIL
else
  kill_hung_server "Pie"
  STATUS[Pie]=TIMEOUT
fi
stop_sglang
fi

# ---- 5c. Neo LLaMA-3-8B ----------------------------------------------------
if $SKIP_NEO; then
  banner "  5c. Neo — SKIPPED (--skip-neo)"
  STATUS[Neo]=SKIPPED
else
  banner "  5c. Neo (swiftllm + ARM libpacpu, LLaMA-3-8B)"

  numactl -N 0 -m 0 python -m swiftllm.server.api_server \
    --port $NEO_PORT \
    --model-path "$LLAMA" \
    --block-size 16 \
    --max-blocks-per-seq 1250 \
    --max-seqs-in-block-table $NEO_BS \
    --max-batch-size $NEO_BS \
    --max-tokens-in-batch $NEO_TIB \
    --tensor-parallel-degree 1 \
    --num-gpu-blocks-override 512 \
    --swap-space $NEO_SWAP \
    --library-path "$LIB" \
    --profile-result-path "$PROFILE_DIR/" \
    --extra-layer-for-cprf \
    > "$LOGS/server_neo.log" 2>&1 &

  if wait_neo "$LOGS/server_neo.log"; then
    echo "[bench] Neo LLaMA-3-8B — rates: $RATES"
    timeout 3600 python benchmarks/bench_offload_serving.py \
      --system Neo \
      --model-id "$LLAMA" --model-label llama-3.1-8b \
      --base-url "http://127.0.0.1:$NEO_PORT" --endpoint /v1/completions \
      --neo-compat --neo-model "$LLAMA" \
      --dataset synthetic \
      --input-len "$INPUT_LEN" --output-len "$OUTPUT_LEN" \
      --request-rates $RATES \
      --num-requests $NREQ --warmup-requests $WARMUP --runs $RUNS \
      --output-csv "$FIG10_CSV" \
      2>&1 | tee "$LOGS/bench_neo.log" && \
      STATUS[Neo]=PASS || STATUS[Neo]=FAIL
  else
    kill_hung_server "Neo"
    STATUS[Neo]=TIMEOUT
  fi
  stop_neo
fi

fi # end SKIP_LLAMA block for sections 5a-5c

# ---- 5g_llama. FlexGen line for the LLaMA panel ---------------------------
# FlexGen has no native LLaMA support, so we run FlexGen-on-OPT-6.7B again
# but tag the CSV row with --model-label llama-3.1-8b so it lands in panel (a).
# This matches the convention used in final_result.py.
if $SKIP_FLEXGEN || $SKIP_LLAMA; then
  banner "  5g_llama. FlexGen (LLaMA panel) — SKIPPED"
  STATUS[FlexGen_LLaMA]=SKIPPED
else
  banner "  5g_llama. FlexGen (proxy for LLaMA panel, OPT-6.7B internally)"

  python "$ROOT/baseline/flexgen/serve_flexgen.py" \
    --model facebook/opt-6.7b \
    --path "$OPT_NP_PARENT" \
    --percent 0 100 100 0 100 0 \
    --port $FLEX_PORT \
    --max-new-tokens 256 \
    > "$LOGS/server_flexgen_llama.log" 2>&1 &

  if wait_flexgen; then
    echo "[bench] FlexGen (LLaMA proxy) — rates: $FLEX_RATES"
    timeout 2700 python benchmarks/bench_offload_serving.py \
      --system FlexGen_LLaMA \
      --model-id facebook/opt-6.7b --model-label llama-3.1-8b \
      --base-url "http://127.0.0.1:$FLEX_PORT" --endpoint /generate \
      --dataset synthetic \
      --input-len "$OPT_INPUT_LEN" --output-len "$OPT_OUTPUT_LEN" \
      --request-rates $FLEX_RATES \
      --num-requests $FLEX_NREQ --warmup-requests $FLEX_WARMUP --runs $FLEX_RUNS \
      --output-csv "$FIG10_CSV" \
      2>&1 | tee "$LOGS/bench_flexgen_llama.log" && \
      STATUS[FlexGen_LLaMA]=PASS || STATUS[FlexGen_LLaMA]=TIMEOUT
  else
    STATUS[FlexGen_LLaMA]=TIMEOUT
  fi
  stop_flexgen
fi

# ============================================================
# Panel (b): OPT-6.7B — DirectKV, Pie, Neo, FlexGen
# ============================================================

# ---- 5d. DirectKV OPT-6.7B -------------------------------------------------
if $NEO_ONLY; then
  banner "  5d. DirectKV OPT-6.7B — SKIPPED (--neo-only)"
  STATUS[DirectKV_OPT6B]=SKIPPED
else
banner "  5d. DirectKV (directkv, OPT-6.7B)"

python -m sglang.launch_server \
  --model-path "$OPT_HF" --tp 1 --port $SGP \
  --attention-backend directkv-smpv2 \
  --dtype bfloat16 \
  --mem-fraction-static $MEM_FRAC_SGLANG \
  --disable-radix-cache \
  --cuda-graph-max-bs $SG_BS \
  --max-running-requests $SG_BS \
  > "$LOGS/server_directkv_opt6b.log" 2>&1 &

if wait_sglang $SGP "DirectKV-OPT6B"; then
  bench_sglang DirectKV_OPT6B "$OPT_HF" opt-6.7b "$RATES" $NREQ $WARMUP $RUNS "$FIG10_CSV" "$OPT_INPUT_LEN" "$OPT_OUTPUT_LEN" && \
    STATUS[DirectKV_OPT6B]=PASS || STATUS[DirectKV_OPT6B]=FAIL
else
  kill_hung_server "DirectKV-OPT6B"
  STATUS[DirectKV_OPT6B]=TIMEOUT
fi
stop_sglang
fi

# ---- 5e. Pie OPT-6.7B -------------------------------------------------------
if $NEO_ONLY; then
  banner "  5e. Pie OPT-6.7B — SKIPPED (--neo-only)"
  STATUS[Pie_OPT6B]=SKIPPED
else
banner "  5e. Pie (flashinfer + KV offload, OPT-6.7B)"

python -m sglang.launch_server \
  --model-path "$OPT_HF" --tp 1 --port $SGP \
  --attention-backend flashinfer \
  --mem-fraction-static $MEM_FRAC_SGLANG \
  --disable-radix-cache --disable-cuda-graph \
  --max-running-requests $SG_BS \
  --skip-server-warmup \
  > "$LOGS/server_pie_opt6b.log" 2>&1 &

if wait_sglang $SGP "Pie-OPT6B"; then
  bench_sglang Pie_OPT6B "$OPT_HF" opt-6.7b "$RATES" $NREQ $WARMUP $RUNS "$FIG10_CSV" "$OPT_INPUT_LEN" "$OPT_OUTPUT_LEN" && \
    STATUS[Pie_OPT6B]=PASS || STATUS[Pie_OPT6B]=FAIL
else
  kill_hung_server "Pie-OPT6B"
  STATUS[Pie_OPT6B]=TIMEOUT
fi
stop_sglang
fi

# ---- 5f. Neo OPT-6.7B -------------------------------------------------------
if $SKIP_NEO; then
  banner "  5f. Neo OPT-6.7B — SKIPPED (--skip-neo)"
  STATUS[Neo_OPT6B]=SKIPPED
elif [ ! -f "$LIB_OPT6B" ]; then
  banner "  5f. Neo OPT-6.7B — SKIPPED (libpacpu-opt_6_7b not found)"
  STATUS[Neo_OPT6B]=SKIPPED
else
  banner "  5f. Neo (swiftllm + ARM libpacpu, OPT-6.7B)"

  numactl -N 0 -m 0 python -m swiftllm.server.api_server \
    --port $NEO_PORT \
    --model-path "$OPT_HF" \
    --block-size 16 \
    --max-blocks-per-seq 1250 \
    --max-seqs-in-block-table $NEO_BS \
    --max-batch-size $NEO_BS \
    --max-tokens-in-batch $NEO_TIB \
    --tensor-parallel-degree 1 \
    --num-gpu-blocks-override 1650 \
    --swap-space $NEO_SWAP \
    --library-path "$LIB_OPT6B" \
    --profile-result-path "$PROFILE_DIR/" \
    --extra-layer-for-cprf \
    > "$LOGS/server_neo_opt6b.log" 2>&1 &

  if wait_neo "$LOGS/server_neo_opt6b.log"; then
    echo "[bench] Neo OPT-6.7B — rates: $RATES"
    timeout 3600 python benchmarks/bench_offload_serving.py \
      --system Neo_OPT6B \
      --model-id "$OPT_HF" --model-label opt-6.7b \
      --base-url "http://127.0.0.1:$NEO_PORT" --endpoint /v1/completions \
      --neo-compat --neo-model "$OPT_HF" \
      --dataset synthetic \
      --input-len "$OPT_INPUT_LEN" --output-len "$OPT_OUTPUT_LEN" \
      --request-rates $RATES \
      --num-requests $NREQ --warmup-requests $WARMUP --runs $RUNS \
      --output-csv "$FIG10_CSV" \
      2>&1 | tee "$LOGS/bench_neo_opt6b.log" && \
      STATUS[Neo_OPT6B]=PASS || STATUS[Neo_OPT6B]=FAIL
  else
    kill_hung_server "Neo-OPT6B"
    STATUS[Neo_OPT6B]=TIMEOUT
  fi
  stop_neo
fi

# ---- 5g. FlexGen OPT-6.7B --------------------------------------------------
if $SKIP_FLEXGEN; then
  banner "  5g. FlexGen — SKIPPED (--skip-flexgen)"
  STATUS[FlexGen]=SKIPPED
else
  banner "  5g. FlexGen (serve_flexgen.py, OPT-6.7B, CPU KV)"
  echo "  Note: FlexGen serves sequentially; expect high TPOT at all rates"

  python "$ROOT/baseline/flexgen/serve_flexgen.py" \
    --model facebook/opt-6.7b \
    --path "$OPT_NP_PARENT" \
    --percent 0 100 100 0 100 0 \
    --port $FLEX_PORT \
    --max-new-tokens 256 \
    > "$LOGS/server_flexgen.log" 2>&1 &

  if wait_flexgen; then
    echo "[bench] FlexGen — rates: $FLEX_RATES  (max 45 min)"
    timeout 2700 python benchmarks/bench_offload_serving.py \
      --system FlexGen \
      --model-id facebook/opt-6.7b --model-label opt-6.7b \
      --base-url "http://127.0.0.1:$FLEX_PORT" --endpoint /generate \
      --dataset synthetic \
      --input-len "$OPT_INPUT_LEN" --output-len "$OPT_OUTPUT_LEN" \
      --request-rates $FLEX_RATES \
      --num-requests $FLEX_NREQ --warmup-requests $FLEX_WARMUP --runs $FLEX_RUNS \
      --output-csv "$FIG10_CSV" \
      2>&1 | tee "$LOGS/bench_flexgen.log" && \
      STATUS[FlexGen]=PASS || STATUS[FlexGen]=TIMEOUT
  else
    STATUS[FlexGen]=TIMEOUT
  fi
  stop_flexgen
fi

# ============================================================
# Panel (c): OPT-30B — DirectKV, FlexGen, Neo
# ============================================================

# ---- 5h. DirectKV OPT-30B -------------------------------------------------
if $SKIP_OPT30B || $NEO_ONLY; then
  banner "  5h. DirectKV OPT-30B — SKIPPED ($( $NEO_ONLY && echo '--neo-only' || echo '--skip-opt30b' ))"
  STATUS[DirectKV_OPT30B]=SKIPPED
elif [ ! -d "$OPT30B_HF" ]; then
  banner "  5h. DirectKV OPT-30B — SKIPPED (weights not found)"
  STATUS[DirectKV_OPT30B]=SKIPPED
else
  banner "  5h. DirectKV (directkv, OPT-30B)"

  python -m sglang.launch_server \
    --model-path "$OPT30B_HF" --tp 1 --port $SGP \
    --attention-backend directkv-smpv2 \
    --dtype bfloat16 \
    --mem-fraction-static $MEM_FRAC_SGLANG_30B \
    --disable-radix-cache \
    --cuda-graph-max-bs $SG_BS_30B \
    --max-running-requests $SG_BS_30B \
    > "$LOGS/server_directkv_opt30b.log" 2>&1 &

  if wait_sglang $SGP "DirectKV-OPT30B"; then
    bench_sglang DirectKV_OPT30B "$OPT30B_HF" opt-30b "$RATES" $NREQ $WARMUP $RUNS "$FIG10_CSV" "$OPT_INPUT_LEN" "$OPT_OUTPUT_LEN" && \
      STATUS[DirectKV_OPT30B]=PASS || STATUS[DirectKV_OPT30B]=FAIL
  else
    kill_hung_server "DirectKV-OPT30B"
    STATUS[DirectKV_OPT30B]=TIMEOUT
  fi
  stop_sglang
fi

# ---- 5i. FlexGen OPT-30B --------------------------------------------------
if $SKIP_OPT30B || $SKIP_FLEXGEN; then
  banner "  5i. FlexGen OPT-30B — SKIPPED"
  STATUS[FlexGen_OPT30B]=SKIPPED
elif [ ! -d "$OPT_NP_PARENT/opt-30b-np" ]; then
  banner "  5i. FlexGen OPT-30B — SKIPPED (NP weights not found)"
  STATUS[FlexGen_OPT30B]=SKIPPED
else
  banner "  5i. FlexGen (serve_flexgen.py, OPT-30B, CPU KV)"

  python "$ROOT/baseline/flexgen/serve_flexgen.py" \
    --model facebook/opt-30b \
    --path "$OPT_NP_PARENT" \
    --percent 0 100 100 0 100 0 \
    --port $FLEX_PORT \
    --max-new-tokens 256 \
    > "$LOGS/server_flexgen_opt30b.log" 2>&1 &

  if wait_flexgen; then
    echo "[bench] FlexGen OPT-30B — rates: $FLEX_RATES  (max 45 min)"
    timeout 2700 python benchmarks/bench_offload_serving.py \
      --system FlexGen_OPT30B \
      --model-id facebook/opt-30b --model-label opt-30b \
      --base-url "http://127.0.0.1:$FLEX_PORT" --endpoint /generate \
      --dataset synthetic \
      --request-rates $FLEX_RATES \
      --num-requests $FLEX_NREQ --warmup-requests $FLEX_WARMUP --runs $FLEX_RUNS \
      --output-csv "$FIG10_CSV" \
      2>&1 | tee "$LOGS/bench_flexgen_opt30b.log" && \
      STATUS[FlexGen_OPT30B]=PASS || STATUS[FlexGen_OPT30B]=TIMEOUT
  else
    STATUS[FlexGen_OPT30B]=TIMEOUT
  fi
  stop_flexgen
fi

# ---- 5k. Pie OPT-30B -------------------------------------------------------
# Panel (c)'s 4th line. Mirrors 5h but with flashinfer + KV offload.
if $SKIP_OPT30B || $NEO_ONLY; then
  banner "  5k. Pie OPT-30B — SKIPPED ($( $NEO_ONLY && echo '--neo-only' || echo '--skip-opt30b' ))"
  STATUS[Pie_OPT30B]=SKIPPED
elif [ ! -d "$OPT30B_HF" ]; then
  banner "  5k. Pie OPT-30B — SKIPPED (weights not found)"
  STATUS[Pie_OPT30B]=SKIPPED
else
  banner "  5k. Pie (flashinfer + KV offload, OPT-30B)"

  python -m sglang.launch_server \
    --model-path "$OPT30B_HF" --tp 1 --port $SGP \
    --attention-backend flashinfer \
    --dtype bfloat16 \
    --mem-fraction-static $MEM_FRAC_SGLANG_30B \
    --disable-radix-cache --disable-cuda-graph \
    --max-running-requests $SG_BS_30B \
    --skip-server-warmup \
    > "$LOGS/server_pie_opt30b.log" 2>&1 &

  if wait_sglang $SGP "Pie-OPT30B"; then
    bench_sglang Pie_OPT30B "$OPT30B_HF" opt-30b "$RATES" $NREQ $WARMUP $RUNS "$FIG10_CSV" "$OPT_INPUT_LEN" "$OPT_OUTPUT_LEN" && \
      STATUS[Pie_OPT30B]=PASS || STATUS[Pie_OPT30B]=FAIL
  else
    kill_hung_server "Pie-OPT30B"
    STATUS[Pie_OPT30B]=TIMEOUT
  fi
  stop_sglang
fi

# ---- 5j. Neo OPT-30B --------------------------------------------------------
if $SKIP_NEO || $SKIP_OPT30B; then
  banner "  5j. Neo OPT-30B — SKIPPED"
  STATUS[Neo_OPT30B]=SKIPPED
elif [ ! -f "$LIB_OPT30B" ]; then
  banner "  5j. Neo OPT-30B — SKIPPED (libpacpu-opt_30b not found)"
  STATUS[Neo_OPT30B]=SKIPPED
elif [ ! -d "$OPT30B_HF" ]; then
  banner "  5j. Neo OPT-30B — SKIPPED (weights not found)"
  STATUS[Neo_OPT30B]=SKIPPED
else
  banner "  5j. Neo (swiftllm + ARM libpacpu, OPT-30B)"

  numactl -N 0 -m 0 python -m swiftllm.server.api_server \
    --port $NEO_PORT \
    --model-path "$OPT30B_HF" \
    --block-size 16 \
    --max-blocks-per-seq 1250 \
    --max-seqs-in-block-table $NEO_BS_30B \
    --max-batch-size $NEO_BS_30B \
    --max-tokens-in-batch $NEO_TIB_30B \
    --tensor-parallel-degree 1 \
    --num-gpu-blocks-override 800 \
    --swap-space $NEO_SWAP \
    --library-path "$LIB_OPT30B" \
    --profile-result-path "$PROFILE_DIR/" \
    --extra-layer-for-cprf \
    > "$LOGS/server_neo_opt30b.log" 2>&1 &

  if wait_neo "$LOGS/server_neo_opt30b.log"; then
    echo "[bench] Neo OPT-30B — rates: $RATES"
    timeout 3600 python benchmarks/bench_offload_serving.py \
      --system Neo_OPT30B \
      --model-id "$OPT30B_HF" --model-label opt-30b \
      --base-url "http://127.0.0.1:$NEO_PORT" --endpoint /v1/completions \
      --neo-compat --neo-model "$OPT30B_HF" \
      --dataset synthetic \
      --input-len "$OPT_INPUT_LEN" --output-len "$OPT_OUTPUT_LEN" \
      --request-rates $RATES \
      --num-requests $NREQ --warmup-requests $WARMUP --runs $RUNS \
      --output-csv "$FIG10_CSV" \
      2>&1 | tee "$LOGS/bench_neo_opt30b.log" && \
      STATUS[Neo_OPT30B]=PASS || STATUS[Neo_OPT30B]=FAIL
  else
    kill_hung_server "Neo-OPT30B"
    STATUS[Neo_OPT30B]=TIMEOUT
  fi
  stop_neo
fi

# ---------------------------------------------------------------------------
# 6. Figure 11 — Long-Context
# ---------------------------------------------------------------------------
if $FIG10_ONLY; then
  banner "Step 3/5 — Figure 11 SKIPPED (--fig10-only)"
else
  banner "Step 3/5 — Figure 11: Long-Context Benchmark"
  echo "  Seq lengths: $SEQ_LENS"
  echo "  Model (SGLang): $LLAMA   Model (FlexGen): facebook/opt-6.7b"

  # ============================================================
  # LLaMA-3-8B long-context — DirectKV, Pie
  # ============================================================

  if $SKIP_LLAMA; then
    banner "  6a-6b. LLaMA-3-8B long-ctx — SKIPPED (--skip-llama)"
    STATUS[LC_DirectKV]=SKIPPED; STATUS[LC_Pie]=SKIPPED
  else

  # ---- 6a. DirectKV long-context -------------------------------------------
  banner "  6a. Long-ctx DirectKV (LLaMA-3-8B)"

  python -m sglang.launch_server \
    --model-path "$LLAMA" --tp 1 --port $SGP \
    --attention-backend directkv-smpv2 \
    --mem-fraction-static $MEM_FRAC_SGLANG \
    --disable-radix-cache \
    --cuda-graph-max-bs 64 \
    --max-running-requests 64 \
    > "$LOGS/server_lc_directkv.log" 2>&1 &

  if wait_sglang $SGP "DirectKV(LC)"; then
    timeout 7200 python benchmarks/bench_offload_longctx.py \
      --system DirectKV \
      --model-id "$LLAMA" --model-label llama-3.1-8b \
      --base-url "http://127.0.0.1:$SGP" \
      --seq-lengths $SEQ_LENS \
      --num-requests $LC_NREQ --warmup-requests $LC_WARMUP --runs $LC_RUNS \
      --output-csv "$FIG11_CSV" \
      2>&1 | tee "$LOGS/lc_directkv.log"
    STATUS[LC_DirectKV]=PASS
  else
    kill_hung_server "DirectKV-LC"
    STATUS[LC_DirectKV]=TIMEOUT
  fi
  stop_sglang

  # ---- 6b. Pie long-context ------------------------------------------------
  banner "  6b. Long-ctx Pie (LLaMA-3-8B)"

  python -m sglang.launch_server \
    --model-path "$LLAMA" --tp 1 --port $SGP \
    --attention-backend flashinfer \
    --mem-fraction-static $MEM_FRAC_SGLANG \
    --disable-radix-cache --disable-cuda-graph \
    --skip-server-warmup \
    > "$LOGS/server_lc_pie.log" 2>&1 &

  if wait_sglang $SGP "Pie(LC)"; then
    timeout 7200 python benchmarks/bench_offload_longctx.py \
      --system Pie \
      --model-id "$LLAMA" --model-label llama-3.1-8b \
      --base-url "http://127.0.0.1:$SGP" \
      --seq-lengths $SEQ_LENS \
      --num-requests $LC_NREQ --warmup-requests $LC_WARMUP --runs $LC_RUNS \
      --output-csv "$FIG11_CSV" \
      2>&1 | tee "$LOGS/lc_pie.log"
    STATUS[LC_Pie]=PASS
  else
    kill_hung_server "Pie-LC"
    STATUS[LC_Pie]=TIMEOUT
  fi
  stop_sglang

  fi # end SKIP_LLAMA block for sections 6a-6b

  # ============================================================
  # OPT-6.7B long-context — DirectKV, Pie, FlexGen
  # ============================================================

  # ---- 6c. DirectKV OPT-6.7B long-context ---------------------------------
  banner "  6c. Long-ctx DirectKV (OPT-6.7B)"

  python -m sglang.launch_server \
    --model-path "$OPT_HF" --tp 1 --port $SGP \
    --attention-backend directkv-smpv2 \
    --dtype bfloat16 \
    --mem-fraction-static $MEM_FRAC_SGLANG \
    --disable-radix-cache \
    --cuda-graph-max-bs 64 \
    --max-running-requests 64 \
    > "$LOGS/server_lc_directkv_opt6b.log" 2>&1 &

  if wait_sglang $SGP "DirectKV-OPT6B(LC)"; then
    timeout 7200 python benchmarks/bench_offload_longctx.py \
      --system DirectKV_OPT6B \
      --model-id "$OPT_HF" --model-label opt-6.7b \
      --base-url "http://127.0.0.1:$SGP" \
      --seq-lengths $OPT_SEQ_LENS \
      --num-requests $LC_NREQ --warmup-requests $LC_WARMUP --runs $LC_RUNS \
      --output-csv "$FIG11_CSV" \
      2>&1 | tee "$LOGS/lc_directkv_opt6b.log"
    STATUS[LC_DirectKV_OPT6B]=PASS
  else
    kill_hung_server "DirectKV-OPT6B-LC"
    STATUS[LC_DirectKV_OPT6B]=TIMEOUT
  fi
  stop_sglang

  # ---- 6d. Pie OPT-6.7B long-context --------------------------------------
  banner "  6d. Long-ctx Pie (OPT-6.7B)"

  python -m sglang.launch_server \
    --model-path "$OPT_HF" --tp 1 --port $SGP \
    --attention-backend flashinfer \
    --mem-fraction-static $MEM_FRAC_SGLANG \
    --disable-radix-cache --disable-cuda-graph \
    --skip-server-warmup \
    > "$LOGS/server_lc_pie_opt6b.log" 2>&1 &

  if wait_sglang $SGP "Pie-OPT6B(LC)"; then
    timeout 7200 python benchmarks/bench_offload_longctx.py \
      --system Pie_OPT6B \
      --model-id "$OPT_HF" --model-label opt-6.7b \
      --base-url "http://127.0.0.1:$SGP" \
      --seq-lengths $OPT_SEQ_LENS \
      --num-requests $LC_NREQ --warmup-requests $LC_WARMUP --runs $LC_RUNS \
      --output-csv "$FIG11_CSV" \
      2>&1 | tee "$LOGS/lc_pie_opt6b.log"
    STATUS[LC_Pie_OPT6B]=PASS
  else
    kill_hung_server "Pie-OPT6B-LC"
    STATUS[LC_Pie_OPT6B]=TIMEOUT
  fi
  stop_sglang

  # ---- 6e. FlexGen offline long-context ------------------------------------
  if ! $SKIP_FLEXGEN; then
    banner "  6e. Long-ctx FlexGen (offline, OPT-6.7B, maximum GPU memory saving)"
    python benchmarks/run_flexgen_longctx.py \
      --model facebook/opt-6.7b \
      --path "$OPT_NP_PARENT" \
      --seq-lengths $OPT_SEQ_LENS \
      --batch-size 4 --num-gpu-batches 1 \
      --percent 0 100 0 100 100 0 \
      --nooffload-csv "$FIG11_CSV" \
      --output-csv "$FIG11_CSV" \
      2>&1 | tee "$LOGS/lc_flexgen.log" && \
      STATUS[LC_FlexGen]=PASS || STATUS[LC_FlexGen]=FAIL
  fi

  # ============================================================
  # OPT-30B long-context — DirectKV
  # ============================================================

  # ---- 6f. DirectKV OPT-30B long-context -----------------------------------
  if $SKIP_OPT30B; then
    banner "  6f. Long-ctx DirectKV OPT-30B — SKIPPED (--skip-opt30b)"
    STATUS[LC_DirectKV_OPT30B]=SKIPPED
  elif [ ! -d "$OPT30B_HF" ]; then
    banner "  6f. Long-ctx DirectKV OPT-30B — SKIPPED (weights not found)"
    STATUS[LC_DirectKV_OPT30B]=SKIPPED
  else
    banner "  6f. Long-ctx DirectKV (OPT-30B)"

    python -m sglang.launch_server \
      --model-path "$OPT30B_HF" --tp 1 --port $SGP \
      --attention-backend directkv-smpv2 \
      --dtype bfloat16 \
      --mem-fraction-static $MEM_FRAC_SGLANG_30B \
      --disable-radix-cache \
      --cuda-graph-max-bs 64 \
      --max-running-requests 64 \
      > "$LOGS/server_lc_directkv_opt30b.log" 2>&1 &

    if wait_sglang $SGP "DirectKV-OPT30B(LC)"; then
      timeout 7200 python benchmarks/bench_offload_longctx.py \
        --system DirectKV_OPT30B \
        --model-id "$OPT30B_HF" --model-label opt-30b \
        --base-url "http://127.0.0.1:$SGP" \
        --seq-lengths $SEQ_LENS \
        --num-requests $LC_NREQ --warmup-requests $LC_WARMUP --runs $LC_RUNS \
        --output-csv "$FIG11_CSV" \
        2>&1 | tee "$LOGS/lc_directkv_opt30b.log"
      STATUS[LC_DirectKV_OPT30B]=PASS
    else
      kill_hung_server "DirectKV-OPT30B-LC"
      STATUS[LC_DirectKV_OPT30B]=TIMEOUT
    fi
    stop_sglang
  fi
fi

# ---------------------------------------------------------------------------
# 7. Generate figures
# ---------------------------------------------------------------------------
banner "Step 4/5 — Generating figures"

python benchmarks/plot_offload_results.py \
  --fig10-csv "$FIG10_CSV" \
  --output-dir "$PLOTS/" \
  2>&1 | tee "$LOGS/plot_fig10.log" || true

if [ -f "$FIG11_CSV" ]; then
  python benchmarks/plot_offload_results.py \
    --fig11-csv "$FIG11_CSV" \
    --output-dir "$PLOTS/" \
    2>&1 | tee "$LOGS/plot_fig11.log" || true
fi

python benchmarks/plot_offload_results.py \
  --generate-sample --output-dir "$PLOTS/sample/" 2>/dev/null || true

# ---------------------------------------------------------------------------
# 8. Summary report
# ---------------------------------------------------------------------------
banner "Step 5/5 — Summary"

echo ""
echo "  Results directory: $RESULTS/"
echo ""
echo "  ┌──────────────────────────────┬──────────┐"
echo "  │ System / Phase               │ Status   │"
echo "  ├──────────────────────────────┼──────────┤"
for key in \
  DirectKV Pie Neo FlexGen_LLaMA \
  DirectKV_OPT6B Pie_OPT6B Neo_OPT6B FlexGen \
  DirectKV_OPT30B Pie_OPT30B FlexGen_OPT30B Neo_OPT30B \
  LC_DirectKV LC_Pie \
  LC_DirectKV_OPT6B LC_Pie_OPT6B LC_FlexGen \
  LC_DirectKV_OPT30B; do
  val="${STATUS[$key]+x}"
  if [ -n "$val" ]; then
    st="${STATUS[$key]}"
    icon="  "; [ "$st" = "PASS" ] && icon="✓ "; [ "$st" = "FAIL" ] && icon="✗ "
    printf "  │ %-28s │ %s%-7s │\n" "$key" "$icon" "$st"
  fi
done
echo "  └──────────────────────────────┴──────────┘"
echo ""

# Count passes (exclude SKIPPED)
TOTAL=0; PASSED=0
for key in "${!STATUS[@]}"; do
  [ "${STATUS[$key]}" = "SKIPPED" ] && continue
  TOTAL=$((TOTAL+1))
  [ "${STATUS[$key]}" = "PASS" ] && PASSED=$((PASSED+1))
done
echo "  PASSED: $PASSED / $TOTAL"
echo ""

echo "  Figure 10 data: $FIG10_CSV"
if [ -f "$FIG10_CSV" ]; then
  echo ""
  column -t -s, "$FIG10_CSV" 2>/dev/null | head -30 || cat "$FIG10_CSV" | head -30
fi

echo ""
echo "  Figure 11 data: $FIG11_CSV"
if [ -f "$FIG11_CSV" ]; then
  echo ""
  column -t -s, "$FIG11_CSV" 2>/dev/null | head -20 || cat "$FIG11_CSV" | head -20
fi

echo ""
echo "  PDF figures:"
ls -lh "$PLOTS/"*.pdf 2>/dev/null | sed 's/^/    /' || echo "    (none — check $LOGS/plot_fig10.log)"

echo ""
if [ "$PASSED" -eq "$TOTAL" ]; then
  echo "  ✓  ALL CHECKS PASSED — artifact evaluation complete."
else
  echo "  ✗  $((TOTAL - PASSED)) check(s) failed or timed out."
  echo "     Inspect logs in $LOGS/ for details."
  exit 1
fi
