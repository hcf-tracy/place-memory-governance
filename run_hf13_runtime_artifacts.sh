#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)";cd "$ROOT";source ./server/load_hf13_runtime.sh;source ./server/hf13_model.sh;source ./server/hf13_status.sh
M="$(hf13_normalize_model "${1:-7b}")";TAG="${2:-$(date +%Y%m%d_%H%M%S)}";CFG="$(hf13_model_config "$M")";D="results/hf13_live/$TAG/$M/runtime_artifacts";mkdir -p "$D";echo RUNNING > "$D/STATUS";date -Is > "$D/started_at.txt"
fail(){ code=$?;echo FAILED > "$D/STATUS";date -Is > "$D/finished_at.txt";exit "$code";};trap fail ERR INT TERM
python -m hf13 runtime-artifacts --config "$CFG" --data-root data/longitudinal --mode llm --output "$D/runtime_artifact_registry.json" >"$D/run.log" 2>&1
S="$(hf13_json_status "$D/runtime_artifact_registry.json")";echo "$S" > "$D/STATUS";date -Is > "$D/finished_at.txt";hf13_accept_status "$S";trap - ERR INT TERM
echo "[HF13 RUNTIME ARTIFACT REGISTRY] $S model=$M tag=$TAG"
