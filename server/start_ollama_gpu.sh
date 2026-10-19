#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU=""; PORT="auto"; PORT_START=11434; PORT_END=11999; MAX_AUTO_RETRIES=8
MODELS_DIR="${OLLAMA_MODELS:-$HOME/.cache/asf_ollama/models}"; BACKGROUND=0
NUM_PARALLEL="${ASF_OLLAMA_NUM_PARALLEL:-1}"; MAX_LOADED_MODELS="${ASF_OLLAMA_MAX_LOADED_MODELS:-2}"; KEEP_ALIVE="${ASF_OLLAMA_KEEP_ALIVE:-10m}"
PYTHON_BIN="$(command -v python3 || command -v python || true)"
[[ -n "$PYTHON_BIN" ]] || { echo "python3/python not found; activate mem_gov or install Python 3" >&2; exit 2; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU="$2"; shift 2;;
    --port) PORT="$2"; shift 2;;
    --port-start) PORT_START="$2"; shift 2;;
    --port-end) PORT_END="$2"; shift 2;;
    --max-auto-retries) MAX_AUTO_RETRIES="$2"; shift 2;;
    --models-dir) MODELS_DIR="$2"; shift 2;;
    --background) BACKGROUND=1; shift;;
    *) echo "Unknown option: $1" >&2; exit 2;;
  esac
done
command -v ollama >/dev/null 2>&1 || { echo "ollama not found; run ./server/install_ollama_user.sh or ask the admin" >&2; exit 2; }
[[ -n "$GPU" ]] || { echo "Specify one physical GPU, e.g. --gpu 4 on phisonAI100." >&2; exit 2; }
$PYTHON_BIN - "$PORT_START" "$PORT_END" "$MAX_AUTO_RETRIES" <<'PY'
import sys
s,e,r=map(int,sys.argv[1:]);assert 1<=s<=e<=65535 and r>=1
PY
mkdir -p "$MODELS_DIR" "$HOME/.cache/asf_ollama/logs" "$ROOT/server_runtime"
RUNTIME_ENV="$ROOT/server_runtime/ollama_runtime.env"; RUNTIME_JSON="$ROOT/server_runtime/ollama_runtime.json"

write_runtime() {
  local port="$1" pid="${2:-}" log="${3:-}" ready="${4:-false}"
  local url="http://127.0.0.1:${port}"
  cat > "$RUNTIME_ENV" <<EOT
export ASF_OLLAMA_PORT='$port'
export ASF_OLLAMA_GPU='$GPU'
export OLLAMA_HOST='$url'
export OLLAMA_BASE_URL='$url'
export OLLAMA_MODELS='$MODELS_DIR'
EOT
  $PYTHON_BIN - "$RUNTIME_JSON" "$port" "$GPU" "$url" "$MODELS_DIR" "$pid" "$log" "$ready" "$NUM_PARALLEL" "$MAX_LOADED_MODELS" "$KEEP_ALIVE" <<'PY'
import json,sys,time
p,port,gpu,url,models,pid,log,ready,num_parallel,max_loaded,keep_alive=sys.argv[1:]
x={'schema_version':'hf13-ollama-runtime-v1','created_unix':time.time(),'port':int(port),'gpu':gpu,'base_url':url,'models_dir':models,'ready':ready.lower()=='true','runtime_hardening':'hf13','num_parallel':int(num_parallel),'max_loaded_models':int(max_loaded),'keep_alive':keep_alive}
if pid:x['pid']=int(pid)
if log:x['log']=log
json.dump(x,open(p,'w'),indent=2);open(p,'a').write('\n')
PY
}

start_background() {
  local start="$PORT_START" attempt=1 port log pid url
  while (( attempt <= MAX_AUTO_RETRIES )); do
    if [[ "$PORT" == "auto" ]]; then
      port="$($PYTHON_BIN "$ROOT/server/find_free_port.py" --start "$start" --end "$PORT_END")"
    else
      port="$PORT"
      $PYTHON_BIN "$ROOT/server/find_free_port.py" --start "$port" --end "$port" >/dev/null 2>&1 || { echo "Port $port is occupied. Use --port auto (default)." >&2; return 2; }
    fi
    url="http://127.0.0.1:${port}"; log="$HOME/.cache/asf_ollama/logs/ollama_${port}_$(date +%Y%m%d_%H%M%S).log"
    echo "[ollama] attempt=$attempt selected port=$port GPU=$GPU"
    CUDA_VISIBLE_DEVICES="$GPU" OLLAMA_HOST="127.0.0.1:${port}" OLLAMA_MODELS="$MODELS_DIR" OLLAMA_KEEP_ALIVE="$KEEP_ALIVE" OLLAMA_NUM_PARALLEL="$NUM_PARALLEL" OLLAMA_MAX_LOADED_MODELS="$MAX_LOADED_MODELS" nohup ollama serve >"$log" 2>&1 & pid=$!
    for _ in $(seq 1 90); do
      if curl -fsS "$url/api/tags" >/dev/null 2>&1; then
        echo "$pid" > "$ROOT/server_runtime/ollama.pid"; write_runtime "$port" "$pid" "$log" true
        echo "[ollama] ready endpoint=$url PID=$pid log=$log"; return 0
      fi
      if ! kill -0 "$pid" 2>/dev/null; then
        if [[ "$PORT" == "auto" ]] && grep -Eqi 'address already in use|bind.*in use|listen.*in use' "$log"; then
          echo "[ollama] port $port was claimed during startup; retrying automatically." >&2
          start=$((port+1)); attempt=$((attempt+1)); break
        fi
        echo "[ollama] exited before readiness; inspect $log" >&2; return 2
      fi
      sleep 1
    done
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      echo "[ollama] did not become ready within 90s; inspect $log" >&2; return 2
    fi
  done
  echo "[ollama] could not reserve/start an Ollama port after $MAX_AUTO_RETRIES attempts." >&2; return 2
}

if [[ "$BACKGROUND" -eq 1 ]]; then
  start_background
  exit $?
fi

# Foreground/tmux mode: select first free port once, record it, then exec. A millisecond-scale
# TOCTOU race is inherently possible because Ollama itself must bind the socket; use --background
# for automatic bind-collision retry on heavily shared hosts.
if [[ "$PORT" == "auto" ]]; then PORT="$($PYTHON_BIN "$ROOT/server/find_free_port.py" --start "$PORT_START" --end "$PORT_END")"; fi
$PYTHON_BIN "$ROOT/server/find_free_port.py" --start "$PORT" --end "$PORT" >/dev/null 2>&1 || { echo "Port $PORT is occupied; omit --port to auto-select." >&2; exit 2; }
URL="http://127.0.0.1:${PORT}"; write_runtime "$PORT" "" "" false
export CUDA_VISIBLE_DEVICES="$GPU" OLLAMA_HOST="127.0.0.1:${PORT}" OLLAMA_BASE_URL="$URL" OLLAMA_MODELS="$MODELS_DIR" OLLAMA_KEEP_ALIVE="$KEEP_ALIVE" OLLAMA_NUM_PARALLEL="$NUM_PARALLEL" OLLAMA_MAX_LOADED_MODELS="$MAX_LOADED_MODELS"
echo "[ollama] selected free port=$PORT GPU=$GPU endpoint=$URL"
echo "[ollama] keep this shell/tmux open; other shells: source ./server/load_ollama_runtime.sh"
exec ollama serve
