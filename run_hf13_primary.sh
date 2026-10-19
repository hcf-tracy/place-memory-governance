#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$ROOT"
source ./server/load_hf13_runtime.sh; source ./server/hf13_model.sh; source ./server/hf13_status.sh
M="$(hf13_normalize_model "${1:-7b}")"; TAG="${2:-$(date +%Y%m%d_%H%M%S)}"; CFG="$(hf13_model_config "$M")"
if [[ "${HF13_PRIMARY_SELFTEST:-0}" == 1 ]]; then python -m hf13 factorial-run --help >/dev/null; echo '[HF13 PRIMARY SELFTEST] PASS write-once/read-branch'; exit 0; fi
REGISTRY="results/hf13_live/$TAG/$M/runtime_artifacts/runtime_artifact_registry.json"; [[ -f "$REGISTRY" ]] || { echo "missing shared RuntimeArtifactRegistry: $REGISTRY" >&2; exit 6; }
D="results/hf13_live/$TAG/$M/primary"; mkdir -p "$D"; echo RUNNING > "$D/PRIMARY_STATUS"; date -Is > "$D/started_at.txt"
for C in native qasc css place; do mkdir -p "$D/$C"; echo RUNNING > "$D/$C/STATUS"; date -Is > "$D/$C/started_at.txt"; done
fail(){ code=$?; echo FAILED > "$D/PRIMARY_STATUS"; date -Is > "$D/finished_at.txt"; for C in native qasc css place; do [[ "$(cat "$D/$C/STATUS" 2>/dev/null || true)" == RUNNING ]] && echo FAILED > "$D/$C/STATUS"; date -Is > "$D/$C/finished_at.txt"; done; exit "$code"; }; trap fail ERR INT TERM
python -m hf13 factorial-run --backend mem0 --config "$CFG" --data-root data/longitudinal --output-root "$D" --answer-mode llm --run-id "hf13_${TAG}_${M}_primary_factorial" --artifact-registry "$REGISTRY" --progress-json "$D/progress.json" --output "$D/factorial_summary.json" > "$D/factorial_run.log" 2>&1
sts=(); for C in native qasc css place; do S="$(hf13_json_status "$D/$C/evaluation.json")"; echo "$S" > "$D/$C/STATUS"; date -Is > "$D/$C/finished_at.txt"; hf13_accept_status "$S"; sts+=("$S"); done
S="$(hf13_merge_status "${sts[@]}")"; hf13_accept_status "$S"; ./run_hf13_state_plane_gate.sh "$M" "$TAG" primary; echo "$S" > "$D/PRIMARY_STATUS"; date -Is > "$D/finished_at.txt"; hf13_accept_status "$S"; trap - ERR INT TERM
echo "[HF13 PRIMARY FACTORIAL] $S model=$M write_once_read_branch"
