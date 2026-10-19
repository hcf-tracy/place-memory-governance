#!/usr/bin/env bash
# Source this file: source server/activate_mem_gov.sh
ENV_DIR="${MEM_GOV_ENV:-$HOME/.venvs/mem_gov}"
if [[ ! -f "$ENV_DIR/bin/activate" ]]; then
  echo "mem_gov environment not found at $ENV_DIR; run ./server/setup_mem_gov.sh first" >&2
  return 2 2>/dev/null || exit 2
fi
# shellcheck disable=SC1090
source "$ENV_DIR/bin/activate"
export PYTHONUNBUFFERED=1
export MEM0_TELEMETRY=False
echo "[mem_gov] active: $(python -V) @ $(command -v python)"
