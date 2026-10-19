#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$ROOT"
source ./server/hf13_model.sh; source ./server/hf13_status.sh
M="$(hf13_normalize_model "${1:-7b}")"; TAG="${2:-$(date +%Y%m%d_%H%M%S)}"; BASE="results/hf13_live/$TAG/$M/core"; mkdir -p "$BASE"; echo RUNNING > "$BASE/CORE_STATUS"; date -Is > "$BASE/started_at.txt"
fail(){ code=$?; stage="$(cat "$BASE/CURRENT_STAGE" 2>/dev/null || echo UNKNOWN)"; echo FAILED > "$BASE/CORE_STATUS"; echo "$stage" > "$BASE/FAILED_STAGE"; date -Is > "$BASE/finished_at.txt"; exit "$code"; }; trap fail ERR INT TERM
stages=(runtime_artifacts primary pairs utility fault closed_loop aggregate); sts=()
for S in "${stages[@]}"; do
  echo "$S" > "$BASE/CURRENT_STAGE"
  case "$S" in
    runtime_artifacts) ./run_hf13_runtime_artifacts.sh "$M" "$TAG"; P="results/hf13_live/$TAG/$M/runtime_artifacts/STATUS";;
    primary) ./run_hf13_primary.sh "$M" "$TAG"; P="results/hf13_live/$TAG/$M/primary/PRIMARY_STATUS";;
    pairs) ./run_hf13_pairs.sh "$M" "$TAG"; P="results/hf13_live/$TAG/$M/pairs/STATUS";;
    utility) ./run_hf13_utility.sh "$M" "$TAG"; P="results/hf13_live/$TAG/$M/utility/STATUS";;
    fault) ./run_hf13_fault.sh "$M" "$TAG"; P="results/hf13_live/$TAG/$M/fault/STATUS";;
    closed_loop) ./run_hf13_closed_loop.sh "$M" "$TAG"; P="results/hf13_live/$TAG/$M/closed_loop/STATUS";;
    aggregate) ./run_hf13_aggregate.sh "$M" "$TAG"; P="results/hf13_live/$TAG/$M/aggregate/STATUS";;
  esac
  sts+=("$(cat "$P")")
done
FINAL="$(hf13_merge_status "${sts[@]}")"; echo "$FINAL" > "$BASE/CORE_STATUS"; echo COMPLETE > "$BASE/CURRENT_STAGE"; date -Is > "$BASE/finished_at.txt"; hf13_accept_status "$FINAL"; trap - ERR INT TERM
echo "[HF13 CORE] $FINAL model=$M tag=$TAG"
