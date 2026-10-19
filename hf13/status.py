from __future__ import annotations

COMPLETE_CLEAN='COMPLETE_CLEAN'
COMPLETE_MODEL_ERRORS='COMPLETE_WITH_MODEL_OPERATION_ERRORS'
COMPLETE_RECOVERED='COMPLETE_WITH_RECOVERED_ERRORS'
FAILED='FAILED'

def from_diagnostics(diags):
    vals=[str((d or {}).get('status') or 'clean') for d in (diags or [])]
    if any(v=='failed' for v in vals):return FAILED
    if any(v=='recovered' for v in vals):return COMPLETE_RECOVERED
    if any(v=='model_operation_error' for v in vals):return COMPLETE_MODEL_ERRORS
    return COMPLETE_CLEAN

def merge_statuses(statuses):
    vals=[str(x or '') for x in statuses]
    if FAILED in vals:return FAILED
    if COMPLETE_RECOVERED in vals:return COMPLETE_RECOVERED
    if COMPLETE_MODEL_ERRORS in vals:return COMPLETE_MODEL_ERRORS
    return COMPLETE_CLEAN

def scientifically_complete(status):
    return status in {COMPLETE_CLEAN,COMPLETE_MODEL_ERRORS}
