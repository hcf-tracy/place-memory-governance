#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$ROOT"
source ./server/hf13_model.sh
M="$(hf13_normalize_model "${1:-7b}")"; TAG="${2:?tag required}"; SURFACE="${3:-primary}"; BASE="results/hf13_live/$TAG/$M/state_plane_gate"; mkdir -p "$BASE"; OUT="$BASE/${SURFACE}.json"; STATUS="$BASE/${SURFACE}.STATUS"; echo RUNNING > "$STATUS"
if python scripts/check_state_plane_gate.py --model-root "results/hf13_live/$TAG/$M" --surface "$SURFACE" --output "$OUT" > "$BASE/${SURFACE}.log" 2>&1; then echo PASS > "$STATUS"; echo "[HF13 STATE-PLANE GATE] PASS model=$M surface=$SURFACE"; else echo STATE_PLANE_INVARIANCE_FAILED > "$STATUS"; cat "$BASE/${SURFACE}.log" >&2; echo "[HF13 STATE-PLANE GATE] FAILED model=$M surface=$SURFACE" >&2; exit 12; fi
