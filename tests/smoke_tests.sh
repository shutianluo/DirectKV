#!/usr/bin/env bash
# =============================================================================
# smoke_tests.sh — Quick sanity check for all four AE systems
#
# Systems tested (no NoOffload):
#   DirectKV   → LLaMA-3-8B  (SGLang + directkv-smpv2 backend)
#   Pie        → LLaMA-3-8B  (SGLang + flashinfer backend)
#   Neo        → LLaMA-3-8B  (swiftllm + ARM libpacpu)
#   FlexGen    → OPT-6.7B-HF + OPT-6.7B-NP
#
# New vs ae_reviewer.sh:
#   - CUDA graph   ENABLED  (--disable-cuda-graph removed)
#   - Pie warmup   DISABLED (--skip-server-warmup; default 4096-req warmup takes >60 s)
#   - Radix cache  DISABLED  (--disable-radix-cache kept — synthetic workload)
#
# Usage
# -----
#   bash tests/smoke_tests.sh               # all four systems
#   bash tests/smoke_tests.sh --skip-neo    # skip Neo
#   bash tests/smoke_tests.sh --skip-flexgen
#   bash tests/smoke_tests.sh --keep-cuda-graph-off   # diagnostic
#
# Pass/fail: exits 0 only if every enabled system produces a valid response.
# Run tests/ae_reviewer.sh only after this script exits 0.
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# 0. Flags
# ---------------------------------------------------------------------------
SKIP_NEO=false
SKIP_FLEXGEN=false
KEEP_CUDA_GRAPH_OFF=false     # diagnostic: restore old --disable-cuda-graph

for arg in "$@"; do
  case $arg in
    --skip-neo)             SKIP_NEO=true ;;
    --skip-flexgen)         SKIP_FLEXGEN=true ;;
    --keep-cuda-graph-off)  KEEP_CUDA_GRAPH_OFF=true ;;
    *) echo "[WARN] Unknown flag: $arg" ;;
  esac
done

# ---------------------------------------------------------------------------
# 1. Paths
# ---------------------------------------------------------------------------
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export NO_PROXY=localhost,127.0.0.1
export no_proxy=localhost,127.0.0.1

TIMESTAMP="$(date +%Y-%m-%d_%H-%M)"
SMOKE_LOG="$ROOT/results/smoke_$TIMESTAMP"
mkdir -p "$SMOKE_LOG"

LLAMA="$ROOT/weights/Llama/Llama-3-8B"
OPT_HF="$ROOT/weights/opt/opt-6.7b-hf"
OPT_NP_PARENT="$ROOT/weights/opt"          # parent of opt-6.7b-np/

NEO_DIR="$ROOT/baseline/neo"
LIB="$NEO_DIR/pacpu/prebuilt/libpacpu-llama3_8b-tp1.so"
PROFILE_DIR="$NEO_DIR/profile_results"
mkdir -p "$PROFILE_DIR"

SGP=30000
NEO_PORT=8000
FLEX_PORT=30001

# CUDA graph flag: empty string = enabled (new default), set if --keep-cuda-graph-off
CUDA_GRAPH_FLAG=""
WARMUP_FLAG=""
if $KEEP_CUDA_GRAPH_OFF; then
  CUDA_GRAPH_FLAG="--disable-cuda-graph"
  WARMUP_FLAG="--skip-server-warmup"
  echo "[INFO] CUDA graph and warmup DISABLED (diagnostic mode)"
else
  echo "[INFO] CUDA graph and warmup ENABLED (new default)"
fi

declare -A STATUS

# ---------------------------------------------------------------------------
# 2. Helpers
# ---------------------------------------------------------------------------
banner() {
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  $*"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

# Wait for /health to return 200; bail immediately if the process dies.
# Usage: wait_sglang <port> <label> <pid>
wait_sglang() {
  local port="${1:-$SGP}" label="${2:-server}" pid="${3:-0}"
  echo "[wait] $label on port $port (CUDA graph capture may take 3-5 min)..."
  for i in $(seq 1 180); do
    # Detect early exit (OOM-kill, crash, etc.) before polling the port.
    if [ "$pid" -gt 0 ] && ! kill -0 "$pid" 2>/dev/null; then
      echo "[wait] $label process (PID $pid) died — check log for errors"
      return 1
    fi
    code=$(curl --noproxy '*' -s -o /dev/null -w "%{http_code}" \
           "http://127.0.0.1:$port/health" 2>/dev/null)
    [ "$code" = "200" ] && echo "[wait] $label ready (${i}x5s = $((i*5))s)" && return 0
    [ $((i % 12)) -eq 0 ] && echo "[wait]   still loading ($((i*5))s)..."
    sleep 5
  done
  echo "[ERROR] Timeout waiting for $label"
  return 1
}

wait_neo() {
  local pid="${1:-0}" log="$SMOKE_LOG/server_neo.log"
  echo "[wait] Neo (swiftllm) — CPU profiling takes ~2 min..."
  for i in $(seq 1 120); do
    if [ "$pid" -gt 0 ] && ! kill -0 "$pid" 2>/dev/null; then
      echo "[wait] Neo process (PID $pid) died — check $log"
      return 1
    fi
    grep -q "Started server process" "$log" 2>/dev/null && \
      echo "[wait] Neo ready (${i}x5s)" && return 0
    [ $((i % 12)) -eq 0 ] && echo "[wait]   still profiling ($((i*5))s)..."
    sleep 5
  done
  echo "[ERROR] Timeout waiting for Neo"
  return 1
}

wait_flexgen() {
  local pid="${1:-0}"
  echo "[wait] FlexGen on port $FLEX_PORT..."
  for i in $(seq 1 120); do
    if [ "$pid" -gt 0 ] && ! kill -0 "$pid" 2>/dev/null; then
      echo "[wait] FlexGen process (PID $pid) died — check $SMOKE_LOG/server_flexgen.log"
      return 1
    fi
    code=$(curl --noproxy '*' -s -o /dev/null -w "%{http_code}" \
           "http://127.0.0.1:$FLEX_PORT/health" 2>/dev/null)
    [ "$code" = "200" ] && echo "[wait] FlexGen ready (${i}x5s)" && return 0
    [ $((i % 12)) -eq 0 ] && echo "[wait]   still loading ($((i*5))s)..."
    sleep 5
  done
  echo "[ERROR] Timeout waiting for FlexGen"
  return 1
}

stop_sglang() {
  pkill    -f "sglang.launch_server"       2>/dev/null || true; sleep 6
  pkill -9 -f "sglang.launch_server"       2>/dev/null || true; sleep 2
}
stop_neo() {
  pkill    -f "swiftllm.server.api_server" 2>/dev/null || true; sleep 6
  pkill -9 -f "swiftllm.server.api_server" 2>/dev/null || true
  pkill -9 -f "gcs_server"                 2>/dev/null || true
  pkill -9 -f "raylet"                     2>/dev/null || true
  sleep 3
  rm -rf /tmp/ray 2>/dev/null || true
}
stop_flexgen() {
  pkill    -f "serve_flexgen"              2>/dev/null || true; sleep 4
  pkill -9 -f "serve_flexgen"              2>/dev/null || true; sleep 2
}

# Probe SGLang /generate (stream:false → plain JSON response)
# Pass response via env var to avoid Python 3.10 f-string parse errors when
# the JSON contains { } characters embedded in triple-quoted string literals.
probe_sglang() {
  local port="${1:-$SGP}" label="${2:-sglang}"
  echo "[probe] $label — POST /generate (stream:false) ..."
  local resp
  resp=$(curl --noproxy '*' -s --max-time 200 -X POST \
    "http://127.0.0.1:$port/generate" \
    -H "Content-Type: application/json" \
    -d '{"text":"The quick brown fox jumps","sampling_params":{"max_new_tokens":8,"temperature":0},"stream":false}')
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "[probe] FAIL — curl error $rc"
    return 1
  fi
  PROBE_RESP="$resp" python3 - <<PYEOF
import sys, json, os
resp = os.environ.get('PROBE_RESP', '')
try:
    d = json.loads(resp)
    t = d.get("text", "")
    if not t:
        print(f"[probe] FAIL — 'text' field empty or missing. Response: {d}")
        sys.exit(1)
    print(f"[probe] OK — generated: {repr(t[:80])}")
except json.JSONDecodeError as e:
    print(f"[probe] FAIL — not valid JSON: {e}")
    print(f"  raw: {repr(resp[:300])}")
    sys.exit(1)
PYEOF
}

# Probe Neo /v1/completions (OpenAI-compatible, non-streaming JSON)
probe_neo() {
  local model_path="$1"
  echo "[probe] Neo — POST /v1/completions ..."
  local resp
  resp=$(curl --noproxy '*' -s --max-time 60 -X POST \
    "http://127.0.0.1:$NEO_PORT/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$model_path\",\"prompt\":\"The quick brown fox jumps\",\"max_tokens\":8,\"temperature\":0,\"stream\":false}")
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "[probe] FAIL — curl error $rc"
    return 1
  fi
  PROBE_RESP="$resp" python3 - <<PYEOF
import sys, json, os
resp = os.environ.get('PROBE_RESP', '')
try:
    d = json.loads(resp)
    choices = d.get("choices", [])
    if not choices or not choices[0].get("text", ""):
        print(f"[probe] FAIL — no text in choices. Response: {d}")
        sys.exit(1)
    print(f"[probe] OK — generated: {repr(choices[0]['text'][:80])}")
except json.JSONDecodeError as e:
    print(f"[probe] FAIL — not valid JSON: {e}")
    print(f"  raw: {repr(resp[:300])}")
    sys.exit(1)
PYEOF
}

# Probe FlexGen /generate (always SSE — read until [DONE] and check last data line)
probe_flexgen() {
  local port="${1:-$FLEX_PORT}"
  echo "[probe] FlexGen — POST /generate (SSE stream) ..."
  local sse
  sse=$(curl --noproxy '*' -s -N --max-time 90 -X POST \
    "http://127.0.0.1:$port/generate" \
    -H "Content-Type: application/json" \
    -d '{"text":"The quick brown fox jumps","sampling_params":{"max_new_tokens":8,"temperature":0},"stream":true}')
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "[probe] FAIL — curl error $rc"
    return 1
  fi
  PROBE_SSE="$sse" python3 - <<PYEOF
import sys, json, os
sse = os.environ.get('PROBE_SSE', '')
lines = sse.splitlines()
found = False
last_text = ""
for line in lines:
    line = line.strip()
    if not line.startswith("data:"):
        continue
    payload = line[len("data:"):].strip()
    if payload == "[DONE]":
        break
    try:
        d = json.loads(payload)
        t = d.get("text", "")
        if t:
            found = True
            last_text = t
    except Exception:
        pass
if found:
    print(f"[probe] OK — generated: {repr(last_text[:80])}")
else:
    print(f"[probe] FAIL — no 'text' found in SSE stream")
    print(f"  raw (first 500 chars): {repr(sse[:500])}")
    sys.exit(1)
PYEOF
}

# ---------------------------------------------------------------------------
# 2b. Pre-run kill-all  (clear any leftover servers from a previous run)
# ---------------------------------------------------------------------------
banner "Pre-run cleanup"
echo "  Killing any leftover sglang / swiftllm / flexgen / ray processes..."
pkill -9 -f "sglang.launch_server"       2>/dev/null || true
pkill -9 -f "swiftllm.server.api_server" 2>/dev/null || true
pkill -9 -f "serve_flexgen"              2>/dev/null || true
pkill -9 -f "gcs_server"                 2>/dev/null || true
pkill -9 -f "raylet"                     2>/dev/null || true
sleep 4
rm -rf /tmp/ray 2>/dev/null || true
echo "  Done."

# ---------------------------------------------------------------------------
# 3. Prerequisite check
# ---------------------------------------------------------------------------
banner "Step 0 — Prerequisites"

PREREQ_FAIL=false
check_dir() {
  local path="$1" label="$2"
  if [ -d "$path" ]; then echo "  [OK]      $label: $path"
  else echo "  [MISSING] $label: $path" >&2; PREREQ_FAIL=true; fi
}

echo "  Models:"
check_dir "$LLAMA"                        "LLaMA-3-8B       (DirectKV / Pie / Neo)"
check_dir "$OPT_HF"                       "OPT-6.7B HF      (FlexGen tokenizer)"
check_dir "$OPT_NP_PARENT/opt-6.7b-np"   "OPT-6.7B NP      (FlexGen weights)"

echo ""
echo "  Neo artifacts:"
[ -f "$LIB" ] && echo "  [OK]      libpacpu: $LIB" || \
  { echo "  [MISSING] libpacpu: $LIB"; PREREQ_FAIL=true; }

echo ""
echo "  Python packages:"
python3 -c "import sglang, swiftllm, flexllmgen, fastapi" 2>/dev/null && \
  echo "  [OK]      sglang swiftllm flexllmgen fastapi" || \
  { echo "  [FAIL]    Missing packages (sglang / swiftllm / flexllmgen / fastapi)"; PREREQ_FAIL=true; }

echo ""
which numactl >/dev/null 2>&1 && echo "  [OK]      numactl" || \
  echo "  [WARN]    numactl not found — Neo will fail (sudo apt-get install -y numactl)"

echo ""
nvidia-smi --query-gpu=name,memory.total,driver_version \
  --format=csv,noheader 2>/dev/null | sed 's/^/  GPU: /'

if $PREREQ_FAIL; then
  echo ""
  echo "[ABORT] Prerequisites missing. Fix above and re-run."
  exit 1
fi

echo ""
echo "  Smoke log dir: $SMOKE_LOG/"
echo "  CUDA graph:    $( $KEEP_CUDA_GRAPH_OFF && echo DISABLED || echo ENABLED )"

# ---------------------------------------------------------------------------
# 4. DirectKV smoke test
# ---------------------------------------------------------------------------
banner "System 1/4 — DirectKV (LLaMA-3-8B, directkv-smpv2)"

python -m sglang.launch_server \
  --model-path "$LLAMA" --tp 1 --port $SGP \
  --attention-backend directkv-smpv2 \
  --mem-fraction-static 0.5 \
  --disable-radix-cache \
  --cuda-graph-max-bs 64 \
  --max-running-requests 64 \
  $CUDA_GRAPH_FLAG $WARMUP_FLAG \
  > "$SMOKE_LOG/server_directkv.log" 2>&1 &
SRV_PID=$!

if wait_sglang $SGP "DirectKV" $SRV_PID; then
  if probe_sglang $SGP "DirectKV"; then
    STATUS[DirectKV]=PASS
  else
    STATUS[DirectKV]=PROBE_FAIL
  fi
else
  STATUS[DirectKV]=TIMEOUT
fi
stop_sglang

# Check for CUDA graph capture errors in log
if grep -qi "cuda graph\|CUDAGraph\|graph capture" "$SMOKE_LOG/server_directkv.log" 2>/dev/null; then
  echo "[INFO] CUDA graph messages in DirectKV log — check $SMOKE_LOG/server_directkv.log"
fi
if grep -i "error\|exception\|traceback" "$SMOKE_LOG/server_directkv.log" 2>/dev/null | grep -qiv "ignore import error"; then
  echo "[WARN] Errors detected in DirectKV log:"
  grep -i "error\|exception\|traceback" "$SMOKE_LOG/server_directkv.log" | grep -iv "ignore import error" | head -5
fi

# ---------------------------------------------------------------------------
# 5. Pie smoke test
# ---------------------------------------------------------------------------
banner "System 2/4 — Pie (LLaMA-3-8B, flashinfer)"

python -m sglang.launch_server \
  --model-path "$LLAMA" --tp 1 --port $SGP \
  --attention-backend flashinfer \
  --mem-fraction-static 0.5 \
  --disable-radix-cache \
  --skip-server-warmup \
  $CUDA_GRAPH_FLAG \
  > "$SMOKE_LOG/server_pie.log" 2>&1 &
SRV_PID=$!

if wait_sglang $SGP "Pie" $SRV_PID; then
  echo "[info] Pie: first decode JIT-compiles triton kernels (~90 s on cold cache) — please wait"
  if probe_sglang $SGP "Pie"; then
    STATUS[Pie]=PASS
  else
    STATUS[Pie]=PROBE_FAIL
  fi
else
  STATUS[Pie]=TIMEOUT
fi
stop_sglang

if grep -i "error\|exception\|traceback" "$SMOKE_LOG/server_pie.log" 2>/dev/null | grep -qiv "ignore import error"; then
  echo "[WARN] Errors detected in Pie log:"
  grep -i "error\|exception\|traceback" "$SMOKE_LOG/server_pie.log" | grep -iv "ignore import error" | head -5
fi

# ---------------------------------------------------------------------------
# 6. Neo smoke test
# ---------------------------------------------------------------------------
if $SKIP_NEO; then
  banner "System 3/4 — Neo SKIPPED (--skip-neo)"
  STATUS[Neo]=SKIPPED
else
  banner "System 3/4 — Neo (LLaMA-3-8B, swiftllm + ARM libpacpu)"

  numactl -N 0 -m 0 python -m swiftllm.server.api_server \
    --port $NEO_PORT \
    --model-path "$LLAMA" \
    --block-size 16 \
    --max-blocks-per-seq 1250 \
    --max-seqs-in-block-table 512 \
    --max-batch-size 16 \
    --max-tokens-in-batch 8192 \
    --tensor-parallel-degree 1 \
    --num-gpu-blocks-override 1650 \
    --swap-space 20 \
    --library-path "$LIB" \
    --profile-result-path "$PROFILE_DIR/" \
    --extra-layer-for-cprf \
    > "$SMOKE_LOG/server_neo.log" 2>&1 &

  NEO_PID=$!
  if wait_neo $NEO_PID; then
    if probe_neo "$LLAMA"; then
      STATUS[Neo]=PASS
    else
      STATUS[Neo]=PROBE_FAIL
    fi
  else
    STATUS[Neo]=TIMEOUT
  fi
  stop_neo

  if grep -i "error\|exception\|traceback" "$SMOKE_LOG/server_neo.log" 2>/dev/null | grep -qiv "ignore import error"; then
    echo "[WARN] Errors detected in Neo log:"
    grep -i "error\|exception\|traceback" "$SMOKE_LOG/server_neo.log" | grep -iv "ignore import error" | head -5
  fi
fi

# ---------------------------------------------------------------------------
# 7. FlexGen smoke test (OPT-6.7B HF + NP)
# ---------------------------------------------------------------------------
if $SKIP_FLEXGEN; then
  banner "System 4/4 — FlexGen SKIPPED (--skip-flexgen)"
  STATUS[FlexGen]=SKIPPED
else
  banner "System 4/4 — FlexGen (OPT-6.7B HF tokenizer + NP weights)"
  echo "  HF path:  $OPT_HF"
  echo "  NP path:  $OPT_NP_PARENT/opt-6.7b-np"

  python "$ROOT/baseline/flexgen/serve_flexgen.py" \
    --model facebook/opt-6.7b \
    --path "$OPT_NP_PARENT" \
    --percent 0 100 100 0 100 0 \
    --port $FLEX_PORT \
    --max-new-tokens 256 \
    > "$SMOKE_LOG/server_flexgen.log" 2>&1 &

  FLEX_PID=$!
  if wait_flexgen $FLEX_PID; then
    if probe_flexgen $FLEX_PORT; then
      STATUS[FlexGen]=PASS
    else
      STATUS[FlexGen]=PROBE_FAIL
    fi
  else
    STATUS[FlexGen]=TIMEOUT
  fi
  stop_flexgen

  if grep -i "error\|exception\|traceback" "$SMOKE_LOG/server_flexgen.log" 2>/dev/null | grep -qiv "ignore import error"; then
    echo "[WARN] Errors detected in FlexGen log:"
    grep -i "error\|exception\|traceback" "$SMOKE_LOG/server_flexgen.log" | grep -iv "ignore import error" | head -5
  fi
fi

# ---------------------------------------------------------------------------
# 8. Summary
# ---------------------------------------------------------------------------
banner "Smoke Test Summary"

echo ""
echo "  ┌──────────────────────────────┬──────────────┐"
echo "  │ System                       │ Status       │"
echo "  ├──────────────────────────────┼──────────────┤"
for key in DirectKV Pie Neo FlexGen; do
  val="${STATUS[$key]+x}"
  if [ -n "$val" ]; then
    st="${STATUS[$key]}"
    case "$st" in
      PASS)       icon="✓ " ;;
      SKIPPED)    icon="  " ;;
      *)          icon="✗ " ;;
    esac
    printf "  │ %-28s │ %s%-11s │\n" "$key" "$icon" "$st"
  fi
done
echo "  └──────────────────────────────┴──────────────┘"
echo ""
echo "  Logs: $SMOKE_LOG/"

TOTAL=0; PASSED=0; FAILED=0
for key in "${!STATUS[@]}"; do
  st="${STATUS[$key]}"
  [ "$st" = "SKIPPED" ] && continue
  TOTAL=$((TOTAL+1))
  [ "$st" = "PASS"    ] && PASSED=$((PASSED+1))
  [ "$st" != "PASS"   ] && FAILED=$((FAILED+1))
done

echo ""
echo "  PASSED: $PASSED / $TOTAL"
echo ""

if [ "$FAILED" -gt 0 ]; then
  echo "  ✗  $FAILED system(s) failed."
  echo ""
  echo "  Diagnose first:"
  for key in DirectKV Pie Neo FlexGen; do
    st="${STATUS[$key]:-}"
    [ "$st" = "PASS" ] || [ "$st" = "SKIPPED" ] || [ -z "$st" ] && continue
    logname=$(echo "$key" | tr '[:upper:]' '[:lower:]')
    echo "    tail -50 $SMOKE_LOG/server_${logname}.log"
  done
  echo ""
  echo "  If DirectKV fails CUDA graph capture:"
  echo "    Re-run with: bash tests/smoke_tests.sh --keep-cuda-graph-off"
  echo "    Then update tests/ae_reviewer.sh to keep --disable-cuda-graph only"
  echo "    for DirectKV call sites (sections 5b and 6b)."
  exit 1
fi

echo "  ✓  All systems passed. Safe to run the full benchmark:"
echo ""
echo "    bash tests/ae_reviewer.sh --quick    # ~25 min sanity run"
echo "    bash tests/ae_reviewer.sh            # full ~2.5 h run"
