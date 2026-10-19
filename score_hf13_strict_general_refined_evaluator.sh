#!/usr/bin/env bash
set -euo pipefail
CALLER_PWD="$PWD"; ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ $# -ge 1 ]] || { echo "Usage: $0 <campaign-root> [models]" >&2; exit 2; }
CAMP_IN="$1"; MODELS="${2:-7b,3b}"; [[ "$CAMP_IN" = /* ]] && CAND="$CAMP_IN" || CAND="$CALLER_PWD/$CAMP_IN"
[[ -d "$CAND" ]] || { echo "campaign directory not found: $CAND" >&2; exit 4; }; CAMP="$(cd "$CAND" && pwd)"
cd "$ROOT"; PY="${HF13_PYTHON_BIN:-}"; [[ -n "$PY" ]] || { command -v python >/dev/null 2>&1 && PY=python || PY=python3; }
BASE="$CAMP/evaluator_strict_general_refined"; rm -rf "$BASE"; mkdir -p "$BASE"; echo RUNNING > "$BASE/STATUS"
"$PY" scripts/score_hf13_strict_general_answer.py --campaign-root "$CAMP" --data-root data/longitudinal --scopes primary,closed_loop,utility,fault,parser_robustness --models "$MODELS" --source-manifest '' --bertscore off --output-dir "$BASE/answer"
"$PY" scripts/score_hf13_strict_general_metric.py --campaign-root "$CAMP" --data-root data/longitudinal --models "$MODELS" --bootstrap-reps "${HF13_STRICT_GENERAL_BOOTSTRAP_REPS:-4000}" --output-dir "$BASE/metric"
echo COMPLETE > "$BASE/STATUS"
echo "[STRICT GENERAL v3.2] COMPLETE: $BASE"
echo "[PAPER] $BASE/metric/PAPER_TABLE_VALUES_STRICT_GENERAL.json"
