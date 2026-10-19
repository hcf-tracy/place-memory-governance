#!/usr/bin/env bash
# Source from HF13 run scripts. Managed start_ollama_gpu runtime wins; an explicit caller OLLAMA_BASE_URL is also accepted.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$ROOT/server_runtime/ollama_runtime.env" ]]; then
  # shellcheck disable=SC1090
  source "$ROOT/server/load_ollama_runtime.sh"
elif [[ -n "${OLLAMA_BASE_URL:-}" ]]; then
  export OLLAMA_BASE_URL="${OLLAMA_BASE_URL%/}"; export OLLAMA_HOST="$OLLAMA_BASE_URL"
else
  # Keep the certified default usable for externally managed Ollama on 11434.
  export OLLAMA_BASE_URL="http://127.0.0.1:11434"; export OLLAMA_HOST="$OLLAMA_BASE_URL"
fi
export PYTHONUNBUFFERED=1 MEM0_TELEMETRY=False

# Refined parser/validation defaults. Callers may override before sourcing.
export HF13_PARSER_SCHEMA_MODE="${HF13_PARSER_SCHEMA_MODE:-structured}"
export HF13_PARSER_CONTEXT_TURNS="${HF13_PARSER_CONTEXT_TURNS:-12}"
export HF13_PARSER_SELF_CHECK="${HF13_PARSER_SELF_CHECK:-1}"
export HF13_ROBUSTNESS_PARSER_MODE="${HF13_ROBUSTNESS_PARSER_MODE:-llm}"
export HF13_ROBUSTNESS_ANSWER_MODE="${HF13_ROBUSTNESS_ANSWER_MODE:-llm}"
export HF13_BERTSCORE_MODE="${HF13_BERTSCORE_MODE:-on}"
export HF13_BERTSCORE_MODEL="${HF13_BERTSCORE_MODEL:-roberta-large}"
