from __future__ import annotations
import copy
from typing import Any

RUNTIME_GOVERNANCE_KEYS=(
    'update','current_use','future_use','retention','data_access',
    'action_authority','applicability','authority_preference',
    'dependency_recompute_of','completion','collection_mutation'
)
UPDATE_VALUES={'assert','replace','remove_current','suspend','restore','clarify','advance_time','mutate_collection'}
TRI_VALUES={'allow','deny','unchanged'}
RETENTION_VALUES={'retain','history','preserve_future','do_not_retain'}
ACCESS_VALUES={'allowed','restricted'}
ACTION_VALUES={'authorized','not_authorized','unspecified'}
APPLICABILITY_VALUES={'current','historical','hypothetical'}
COMPLETION_VALUES={None,'task','condition','trip'}
COLLECTION_VALUES={None,'remove_selected_item'}

def default_governance()->dict[str,Any]:
    return {
        'update':'assert',
        'current_use':'unchanged',
        'future_use':'unchanged',
        'retention':'retain',
        'data_access':'allowed',
        'action_authority':'unspecified',
        'applicability':'current',
        'authority_preference':None,
        'dependency_recompute_of':None,
        'completion':None,
        'collection_mutation':None,
    }

def canonicalize_legacy_intent(intent:dict[str,Any]|None, event:dict[str,Any]|None=None)->dict[str,Any]:
    """Collapse verbose authoring intent into a small, general runtime contract.

    The verbose authoring keys are never required by CSS at runtime. This function
    is used only while building/diagnosing the benchmark and by deterministic
    parser authoring helpers.
    """
    i=copy.deepcopy(intent or {});g=default_governance();event=event or {}
    # Generic update semantics.
    if i.get('advance_effective_time'):g['update']='advance_time'
    elif i.get('requires_clarification_before_state_change'):g['update']='clarify'
    elif i.get('allow_current_planning_use') is False and i.get('preserve_for_future_use'):g['update']='suspend'
    elif i.get('allow_previously_paused_candidate_for_current_use'):g['update']='restore'
    elif i.get('remove_selected_item_from_existing_shortlist'):g['update']='mutate_collection';g['collection_mutation']='remove_selected_item'
    elif i.get('remove_from_current_planning') or i.get('remove_from_current_candidate_use'):g['update']='remove_current'
    elif any(i.get(k) for k in ('replace_current_value','set_current_limit','replace_tentative_date_state','condition_completed','replace_branch_component_only')):g['update']='replace'

    # Current/future applicability and preservation.
    if i.get('allow_current_planning_use') is True:g['current_use']='allow'
    if i.get('allow_current_planning_use') is False or i.get('do_not_apply_to_active_plan'):g['current_use']='deny'
    if i.get('allow_upcoming_planning_use') is True or i.get('allowed_for_future_planning') is True:g['future_use']='allow'
    if i.get('allow_upcoming_planning_use') is False or i.get('allowed_for_future_planning') is False:g['future_use']='deny'

    if i.get('future_retention_allowed') is False or i.get('retain_after_task_complete') is False or i.get('exclude_from_archived_trip_history'):
        g['retention']='do_not_retain'
    elif any(i.get(k) for k in ('preserve_history','preserve_historical_context','preserve_as_historical_travel_memory','preserve_trip_history_only')):
        g['retention']='history'
    elif i.get('preserve_for_future_use'):
        g['retention']='preserve_future'

    if i.get('future_retention_allowed') is False or i.get('allowed_for_future_planning') is False or i.get('exclude_from_shareable_summary') or i.get('exclude_private_user_information'):
        g['data_access']='restricted'

    if i.get('booking_authorized') is True or i.get('allow_irreversible_action') is True:
        g['action_authority']='authorized'
    elif i.get('booking_authorized') is False or i.get('require_new_confirmation_before_irreversible_booking') or i.get('require_explicit_confirmation'):
        g['action_authority']='not_authorized'

    ep=str((event.get('epistemic') or {}).get('status') or '')
    if i.get('hypothetical_only') or i.get('do_not_create_active_trip') or ep=='hypothetical':
        g['applicability']='hypothetical'
    elif i.get('do_not_apply_to_active_plan') or (i.get('preserve_as_historical_travel_memory') and i.get('allow_upcoming_planning_use') is False):
        g['applicability']='historical'

    if i.get('prefer_latest_source_for_current_value'):
        g['authority_preference']=str(i.get('preferred_source') or (event.get('assertion') or {}).get('value') or '') or None
    if i.get('recompute_dependents_of'):
        g['dependency_recompute_of']=str(i.get('recompute_dependents_of'))

    if i.get('task_completed'):g['completion']='task'
    elif i.get('condition_completed'):g['completion']='condition'
    elif i.get('trip_completed'):g['completion']='trip'
    return g

def canonicalize_event(event:dict[str,Any])->tuple[dict[str,Any],dict[str,Any]]:
    """Return (runtime_event, descriptive_annotation)."""
    src=copy.deepcopy(event);runtime={k:copy.deepcopy(v) for k,v in src.items() if k not in {'intent','surface_forms','runtime_binding'}}
    runtime['governance']=canonicalize_legacy_intent(src.get('intent') or {},src)
    desc={
        'event_id':src.get('event_id'),'turn_id':src.get('turn_id'),
        'authoring_intent':copy.deepcopy(src.get('intent') or {}),
        'surface_forms':copy.deepcopy(src.get('surface_forms') or []),
        'runtime_binding_authoring_hint':src.get('runtime_binding'),
        'visible_to_runtime':False,
    }
    return runtime,desc

def validate_runtime_governance(g:dict[str,Any]):
    if not isinstance(g,dict):raise ValueError('governance must be object')
    if set(g)!=set(RUNTIME_GOVERNANCE_KEYS):raise ValueError(f'governance keys must be fixed: {RUNTIME_GOVERNANCE_KEYS}')
    if g['update'] not in UPDATE_VALUES:raise ValueError('bad governance.update')
    if g['current_use'] not in TRI_VALUES or g['future_use'] not in TRI_VALUES:raise ValueError('bad governance use value')
    if g['retention'] not in RETENTION_VALUES:raise ValueError('bad governance.retention')
    if g['data_access'] not in ACCESS_VALUES:raise ValueError('bad governance.data_access')
    if g['action_authority'] not in ACTION_VALUES:raise ValueError('bad governance.action_authority')
    if g['applicability'] not in APPLICABILITY_VALUES:raise ValueError('bad governance.applicability')
    if g['completion'] not in COMPLETION_VALUES:raise ValueError('bad governance.completion')
    if g['collection_mutation'] not in COLLECTION_VALUES:raise ValueError('bad governance.collection_mutation')
    if g['authority_preference'] is not None and not isinstance(g['authority_preference'],str):raise ValueError('bad governance.authority_preference')
    if g['dependency_recompute_of'] is not None and not isinstance(g['dependency_recompute_of'],str):raise ValueError('bad governance.dependency_recompute_of')
    return None
