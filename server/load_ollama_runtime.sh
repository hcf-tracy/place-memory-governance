#!/usr/bin/env bash
# Source this file. It loads the endpoint selected by start_ollama_gpu.sh.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_ENV="${ASF_OLLAMA_RUNTIME_ENV:-$ROOT/server_runtime/ollama_runtime.env}"
if [[ -f "$RUNTIME_ENV" ]]; then
  # shellcheck disable=SC1090
  source "$RUNTIME_ENV"
elif [[ -n "${OLLAMA_BASE_URL:-}" ]]; then
  : # explicit caller override is authoritative
elif [[ "${ASF_ALLOW_DEFAULT_OLLAMA_PORT:-0}" == "1" ]]; then
  export OLLAMA_BASE_URL="http://127.0.0.1:11434"
else
  echo "[ASF] No Ollama runtime endpoint found." >&2
  echo "Run ./server/start_ollama_gpu.sh --gpu <physical_gpu> first, or export OLLAMA_BASE_URL explicitly." >&2
  return 2 2>/dev/null || exit 2
fi
export OLLAMA_BASE_URL="${OLLAMA_BASE_URL%/}"
export OLLAMA_HOST="${OLLAMA_BASE_URL}"
