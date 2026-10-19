#!/usr/bin/env bash
# Status helpers. Scientifically complete runs may contain evaluated-system errors
# or fully verified retries; only FAILED is always fatal. Set HF13_REQUIRE_CLEAN=1
# to demand a pristine COMPLETE_CLEAN campaign.
hf13_json_status(){ python - "$1" <<'PY'
import json,sys
p=sys.argv[1]
try:o=json.load(open(p));print(o.get('status') or (o.get('evaluation') or {}).get('status') or 'UNKNOWN')
except Exception:print('UNKNOWN')
PY
}
hf13_accept_status(){
  local s="$1"
  case "$s" in
    COMPLETE_CLEAN) return 0;;
    COMPLETE_WITH_MODEL_OPERATION_ERRORS)
      [[ "${HF13_REQUIRE_CLEAN:-0}" == "1" ]] && { echo "[HF13 STRICT] model-operation errors present; clean-only mode rejects this run" >&2; return 4; }
      echo "[HF13 MODEL] Mem0/Qwen operation errors were recorded as evaluated-system outcomes; campaign continues." >&2; return 0;;
    COMPLETE_WITH_RECOVERED_ERRORS)
      [[ "${HF13_REQUIRE_CLEAN:-0}" == "1" ]] && { echo "[HF13 STRICT] verified recovery present; clean-only mode rejects this run" >&2; return 4; }
      echo "[HF13 RECOVERED] A read-only/verified operation required retry or bounded repair; recovery is recorded and campaign continues." >&2; return 0;;
    *) echo "[HF13 STRICT] unacceptable/fatal status: $s" >&2; return 5;;
  esac
}
hf13_complete_for_parser_barrier(){
  case "$1" in COMPLETE_CLEAN|COMPLETE_WITH_MODEL_OPERATION_ERRORS|COMPLETE_WITH_RECOVERED_ERRORS) return 0;; *) return 1;; esac
}
hf13_merge_status(){
  local worst=COMPLETE_CLEAN s
  for s in "$@"; do
    [[ "$s" == FAILED ]] && { echo FAILED; return; }
    [[ "$s" == COMPLETE_WITH_RECOVERED_ERRORS ]] && worst=COMPLETE_WITH_RECOVERED_ERRORS
    if [[ "$s" == COMPLETE_WITH_MODEL_OPERATION_ERRORS && "$worst" == COMPLETE_CLEAN ]]; then worst=COMPLETE_WITH_MODEL_OPERATION_ERRORS; fi
  done
  echo "$worst"
}
