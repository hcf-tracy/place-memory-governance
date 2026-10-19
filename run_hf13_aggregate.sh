#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$ROOT"
source ./server/hf13_model.sh; source ./server/hf13_status.sh
M="$(hf13_normalize_model "${1:-7b}")"; TAG="${2:-$(date +%Y%m%d_%H%M%S)}"; D="results/hf13_live/$TAG/$M/aggregate"; mkdir -p "$D"; echo RUNNING > "$D/STATUS"; date -Is > "$D/started_at.txt"
for G in primary closed_loop; do [[ "$(cat "results/hf13_live/$TAG/$M/state_plane_gate/${G}.STATUS" 2>/dev/null || true)" == PASS ]] || { echo "paper aggregation refused: state-plane gate $G is not PASS" >&2; echo STATE_PLANE_INVARIANCE_FAILED > "$D/STATUS"; exit 12; }; done
fail(){ code=$?; echo FAILED > "$D/STATUS"; date -Is > "$D/finished_at.txt"; exit "$code"; }; trap fail ERR INT TERM
python scripts/aggregate_hf13_model.py --model-root "results/hf13_live/$TAG/$M" --output "$D/statistics.json" > "$D/run.log" 2>&1
S="$(hf13_json_status "$D/statistics.json")"; echo "$S" > "$D/STATUS"; date -Is > "$D/finished_at.txt"; hf13_accept_status "$S"; trap - ERR INT TERM
echo "[HF13 AGGREGATE] $S model=$M"
