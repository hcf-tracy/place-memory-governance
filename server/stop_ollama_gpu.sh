#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIDFILE="$ROOT/server_runtime/ollama.pid"
if [[ ! -f "$PIDFILE" ]]; then echo "[ollama] no managed HF13 PID file"; exit 0; fi
pid="$(cat "$PIDFILE")"
if kill -0 "$pid" 2>/dev/null; then kill "$pid"; for _ in $(seq 1 20); do kill -0 "$pid" 2>/dev/null || break; sleep .5; done; fi
rm -f "$PIDFILE" "$ROOT/server_runtime/ollama_runtime.env"
echo "[ollama] stopped managed PID=$pid"
