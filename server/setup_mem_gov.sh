#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${MEM_GOV_ENV:-$HOME/.venvs/mem_gov}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || { echo "[mem_gov] Python 3.11 is required." >&2; exit 2; }
mkdir -p "$(dirname "$ENV_DIR")"
if [[ -x "$ENV_DIR/bin/python" ]]; then
  pyver="$($ENV_DIR/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  [[ "$pyver" == "3.11" ]] || { echo "[mem_gov] existing $ENV_DIR uses Python $pyver; expected 3.11" >&2; exit 2; }
  echo "[mem_gov] reusing $ENV_DIR"
else
  "$PYTHON_BIN" -m venv "$ENV_DIR"
fi
# shellcheck disable=SC1090
source "$ENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "$ROOT/server/requirements-server.txt"
python "$ROOT/server/check_mem_gov_env.py" --output "$ROOT/server/MEM_GOV_ENV_CHECK.json"
echo "[mem_gov] ready. Activate with: source ./server/activate_mem_gov.sh"
