#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$ROOT"
source ./server/load_hf13_runtime.sh; source ./server/hf13_model.sh; source ./server/hf13_status.sh
M="$(hf13_normalize_model "${1:-7b}")"; TAG="${2:-$(date +%Y%m%d_%H%M%S)}"; CFG="$(hf13_model_config "$M")"; REGISTRY="results/hf13_live/$TAG/$M/runtime_artifacts/runtime_artifact_registry.json"; [[ -f "$REGISTRY" ]] || { echo "missing shared RuntimeArtifactRegistry: $REGISTRY" >&2; exit 6; }
BASE="results/hf13_live/$TAG/$M/closed_loop"; mkdir -p "$BASE"; echo RUNNING > "$BASE/STATUS"; date -Is > "$BASE/started_at.txt"
for C in native qasc css place; do mkdir -p "$BASE/$C"; echo RUNNING > "$BASE/$C/STATUS"; date -Is > "$BASE/$C/started_at.txt"; done
fail(){ code=$?; echo FAILED > "$BASE/STATUS"; date -Is > "$BASE/finished_at.txt"; for C in native qasc css place; do [[ "$(cat "$BASE/$C/STATUS" 2>/dev/null || true)" == RUNNING ]] && echo FAILED > "$BASE/$C/STATUS"; date -Is > "$BASE/$C/finished_at.txt"; done; exit "$code"; }; trap fail ERR INT TERM
python -m hf13 factorial-run --closed-loop --backend mem0 --config "$CFG" --data-root data/longitudinal --output-root "$BASE" --answer-mode llm --run-id "hf13_${TAG}_${M}_closed_factorial" --artifact-registry "$REGISTRY" --progress-json "$BASE/progress.json" --output "$BASE/factorial_summary.json" > "$BASE/factorial_run.log" 2>&1
sts=(); for C in native qasc css place; do S="$(hf13_json_status "$BASE/$C/result.json")"; echo "$S" > "$BASE/$C/STATUS"; date -Is > "$BASE/$C/finished_at.txt"; hf13_accept_status "$S"; sts+=("$S"); done
S="$(hf13_merge_status "${sts[@]}")"; hf13_accept_status "$S"; ./run_hf13_state_plane_gate.sh "$M" "$TAG" closed_loop; echo "$S" > "$BASE/STATUS"; date -Is > "$BASE/finished_at.txt"; hf13_accept_status "$S"; trap - ERR INT TERM
echo "[HF13 CLOSED LOOP FACTORIAL] $S model=$M write_once_read_branch"
