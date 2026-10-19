#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
[[ $# -ge 1 ]] || { echo "Usage: $0 <completed-campaign-root> [models]" >&2; exit 2; }
./score_hf13_strict_general_refined_evaluator.sh "$1" "${2:-7b,3b}"
