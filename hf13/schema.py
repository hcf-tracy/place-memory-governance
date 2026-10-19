from __future__ import annotations
from typing import Any
from .governance_schema import validate_runtime_governance

FORBIDDEN_RAW_FIELDS={'authoring_label','gold_category','governance_family','expected_transition','checkpoint_label','gold','required_answer','forbidden_answer','lifecycle'}
FORBIDDEN_EVENT_KEYS={'parsed_operation','operation','transition','resulting_state','expected_state','gold_state','required_answer','forbidden_answer','lifecycle','intent'}
FORBIDDEN_QUERY_KEYS={'required_answer','forbidden_answer','eligible_facts','ineligible_facts','gold_query','expected_answer'}

def _walk_keys(x:Any):
    if isinstance(x,dict):
        for k,v in x.items():yield str(k);yield from _walk_keys(v)
    elif isinstance(x,list):
        for v in x:yield from _walk_keys(v)

def validate_raw_row(x:dict[str,Any]):
    allowed={'conversation_id','turn_id','turn_index','timestamp','utterance'}
    extra=set(x)-allowed
    if extra or set(x)&FORBIDDEN_RAW_FIELDS:raise ValueError(f'raw trajectory leakage: {sorted(extra or (set(x)&FORBIDDEN_RAW_FIELDS))}')
    if not x.get('utterance'):raise ValueError('empty utterance')

def _require_object(name:str, value:Any, *, optional:bool=False):
    if value is None and optional:return {}
    if not isinstance(value,dict):raise ValueError(f'{name} must be object')
    return value

def _require_selector_list(value:Any):
    if value is None:return []
    if not isinstance(value,list):raise ValueError('query selectors must be a list')
    for i,sel in enumerate(value):
        if not isinstance(sel,dict):raise ValueError(f'query selector[{i}] must be object')
        if set(sel)-{'entity','predicate'}:raise ValueError(f'query selector[{i}] has noncanonical keys')
        for k,v in sel.items():
            if not isinstance(v,str) or not v:raise ValueError(f'query selector[{i}].{k} must be nonempty string')
    return value

def validate_trusted_event(e:dict[str,Any]):
    allowed={'event_id','turn_id','target','assertion','governance','scope','provenance','epistemic','temporal','depends_on'}
    extra=set(e)-allowed
    if extra:raise ValueError(f'noncanonical trusted event keys: {sorted(extra)}')
    bad=set(_walk_keys(e))&FORBIDDEN_EVENT_KEYS
    if bad:raise ValueError(f'trusted event contains downstream answer/operation: {sorted(bad)}')
    if 'event_id' not in e or 'turn_id' not in e or 'target' not in e:raise ValueError('event missing identity/target')
    target=_require_object('event target',e.get('target'))
    if set(target)!={'entity','predicate'}:raise ValueError('event target must be entity+predicate')
    if not all(isinstance(target[k],str) and target[k] for k in ('entity','predicate')):raise ValueError('event target values must be nonempty strings')
    assertion=_require_object('event assertion',e.get('assertion'))
    if set(assertion)!={'value'}:raise ValueError('event assertion must contain exactly value')
    _require_object('event scope',e.get('scope') or {})
    provenance=_require_object('event provenance',e.get('provenance') or {})
    if provenance.get('source_type') is not None and not isinstance(provenance.get('source_type'),str):raise ValueError('event provenance.source_type must be string')
    epistemic=_require_object('event epistemic',e.get('epistemic') or {})
    if epistemic.get('status') is not None and not isinstance(epistemic.get('status'),str):raise ValueError('event epistemic.status must be string')
    temporal=_require_object('event temporal',e.get('temporal'),optional=True)
    deps=e.get('depends_on')
    if deps is not None:
        if not isinstance(deps,list):raise ValueError('event depends_on must be a list')
        for i,d in enumerate(deps):
            if not isinstance(d,dict):raise ValueError(f'event depends_on[{i}] must be object')
            if not isinstance(d.get('entity'),str) or not isinstance(d.get('predicate'),str):raise ValueError(f'event depends_on[{i}] requires string entity/predicate')
            if d.get('source_sequence') is not None and not isinstance(d.get('source_sequence'),int):raise ValueError(f'event depends_on[{i}].source_sequence must be integer')
    validate_runtime_governance(e.get('governance'))
    # Runtime binding remains symbolic inside assertion value only.
    v=assertion.get('value')
    if isinstance(v,dict) and 'binding_id' in v and set(v)!={'binding_id'}:raise ValueError('binding placeholder must contain binding_id only')

def validate_trusted_query(q:dict[str,Any]):
    allowed={'query_id','turn_id','target','query_intent','scope','as_of','family'}
    extra=set(q)-allowed
    if extra:raise ValueError(f'noncanonical trusted query keys: {sorted(extra)}')
    bad=set(_walk_keys(q))&FORBIDDEN_QUERY_KEYS
    if bad:raise ValueError(f'trusted query leaks evaluator answer: {sorted(bad)}')
    if not q.get('query_id') or not q.get('turn_id'):raise ValueError('query missing id')
    target=_require_object('query target',q.get('target',{}))
    if set(target)-{'entity','predicate'}:raise ValueError('query target has noncanonical keys')
    for k,v in target.items():
        if not isinstance(v,str) or not v:raise ValueError(f'query target.{k} must be nonempty string')
    qi=_require_object('query query_intent',q.get('query_intent',{}))
    _require_object('query scope',q.get('scope',{}))
    _require_selector_list(qi.get('selectors'))
    if q.get('as_of') is not None and not isinstance(q.get('as_of'),str):raise ValueError('query as_of must be string or null')

def validate_native_metadata(metadata:dict[str,Any]|None):
    md=metadata or {}
    # Native is a real Mem0 baseline: benchmark governance semantics are forbidden.
    banned={'trusted_event','trusted_query','target','governance','scope','epistemic','permission','retention','authority','lifecycle','expected_state','gold','required_answer','forbidden_answer'}
    if set(md)&banned:raise ValueError(f'Native metadata leakage: {sorted(set(md)&banned)}')
    if md:raise ValueError('Native benchmark writes must use empty metadata={}')
