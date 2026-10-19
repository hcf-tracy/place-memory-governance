#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$ROOT"
source ./server/load_hf13_runtime.sh; source ./server/hf13_model.sh; source ./server/hf13_status.sh
M="$(hf13_normalize_model "${1:-7b}")"; TAG="${2:-$(date +%Y%m%d_%H%M%S)}"; CORE_TAG="${3:-$TAG}"; CFG="$(hf13_model_config "$M")"
CORE_BASE="${HF13_CORE_SOURCE_ROOT:-results/hf13_live/$CORE_TAG}"
REGISTRY="$CORE_BASE/$M/runtime_artifacts/runtime_artifact_registry.json"
[[ -f "$REGISTRY" ]] || { echo "parser robustness requires RuntimeArtifactRegistry from complete core: $REGISTRY" >&2; exit 8; }
BASE="results/hf13_live/$TAG/$M/parser_robustness"; mkdir -p "$BASE"; echo RUNNING > "$BASE/STATUS"; [[ -f "$BASE/started_at.txt" ]] || date -Is > "$BASE/started_at.txt"; printf '%s\n' "$CORE_BASE" > "$BASE/CORE_SOURCE_ROOT"
fail(){ code=$?; echo FAILED > "$BASE/STATUS"; date -Is > "$BASE/finished_at.txt"; exit "$code"; }; trap fail ERR INT TERM
PARSER_MODE="${HF13_ROBUSTNESS_PARSER_MODE:-llm}"; ANSWER_MODE="${HF13_ROBUSTNESS_ANSWER_MODE:-llm}"; BOOT="${HF13_ROBUSTNESS_BOOTSTRAP_REPS:-4000}"
python -m hf13 parser-robustness --parser-mode "$PARSER_MODE" --config "$CFG" --data-root data/longitudinal --artifact-registry "$REGISTRY" --answer-mode "$ANSWER_MODE" --progress-json "$BASE/progress.json" --checkpoint-dir "$BASE/checkpoints" --bootstrap-reps "$BOOT" --output "$BASE/result.json" >> "$BASE/run.log" 2>&1
python scripts/render_hf13_parser_robustness.py "$BASE/result.json" --output "$BASE/parser_robustness_report.md"
S="$(hf13_json_status "$BASE/result.json")"; echo "$S" > "$BASE/STATUS"; date -Is > "$BASE/finished_at.txt"; hf13_accept_status "$S"; trap - ERR INT TERM
echo "[HF13 PARSER ROBUSTNESS] $S model=$M core_source=$CORE_BASE cases=24"
