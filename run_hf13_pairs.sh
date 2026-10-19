#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$ROOT"
source ./server/load_hf13_runtime.sh; source ./server/hf13_model.sh; source ./server/hf13_status.sh
M="$(hf13_normalize_model "${1:-7b}")"; TAG="${2:-$(date +%Y%m%d_%H%M%S)}"; CFG="$(hf13_model_config "$M")"; BASE="results/hf13_live/$TAG/$M/pairs"; mkdir -p "$BASE"; echo RUNNING > "$BASE/STATUS"; date -Is > "$BASE/started_at.txt"
fail(){ code=$?; echo FAILED > "$BASE/STATUS"; date -Is > "$BASE/finished_at.txt"; exit "$code"; }; trap fail ERR INT TERM
sts=()
for C in native css qasc place; do
  D="$BASE/$C"; mkdir -p "$D"; echo RUNNING > "$D/STATUS"; date -Is > "$D/started_at.txt"; : > "$D/progress.jsonl"
  python -m hf13 pairs --condition "$C" --backend mem0 --config "$CFG" --data-root data/longitudinal --answer-mode llm --progress-jsonl "$D/progress.jsonl" --output "$D/result.json" > "$D/run.log" 2>&1
  S="$(hf13_json_status "$D/result.json")"; echo "$S" > "$D/STATUS"; date -Is > "$D/finished_at.txt"; hf13_accept_status "$S"; sts+=("$S")
done
S="$(hf13_merge_status "${sts[@]}")"; echo "$S" > "$BASE/STATUS"; date -Is > "$BASE/finished_at.txt"; hf13_accept_status "$S"; trap - ERR INT TERM
echo "[HF13 PAIRS] $S model=$M"
