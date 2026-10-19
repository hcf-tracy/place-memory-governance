from __future__ import annotations
"""Generic authorized-memory selection used by the runtime evaluator.

This function was separated from unpublished benchmark-authoring data for
code-only release. Its body is identical to the experimental implementation.
"""
import copy
from collections import defaultdict
from .css import canon

def _scope_match(fscope,qscope):
    fs=fscope or {};qs=qscope or {}
    if qs.get('trips') and fs.get('trip') and fs['trip'] not in qs['trips']:return False
    for k,v in qs.items():
        if k=='trips':continue
        if k in fs and fs[k]!=v:return False
    return True

def oracle_authorize(facts:list[dict],q:dict)->list[dict]:
    target=q.get('target') or {};qi=q.get('query_intent') or {};view=qi.get('temporal_view','current');selectors=qi.get('selectors') or []
    def selected(f):
        if selectors:
            ok=False
            for sel in selectors:
                if sel.get('entity') and f.get('entity')!=sel['entity']:continue
                if sel.get('predicate') and f.get('predicate')!=sel['predicate']:continue
                ok=True;break
            if not ok:return False
        if target.get('entity') and f.get('entity')!=target['entity']:return False
        if target.get('predicate') and f.get('predicate')!=target['predicate']:return False
        return _scope_match(f.get('scope'),q.get('scope'))
    xs=[]
    for f in facts:
        if not selected(f):continue
        if view=='historical':
            if f.get('lifecycle') not in {'active','superseded','archived','expired','suspended','inactive'}:continue
        else:
            if f.get('lifecycle')!='active' or f.get('control_status')!='controlling':continue
            if f.get('epistemic') in {'ambiguous','hypothetical','stale'}:continue
            if f.get('retention')=='do_not_retain':continue
        if qi.get('request_sensitive_value') and f.get('permission')=='restricted':continue
        xs.append(copy.deepcopy(f))
    completed={str((f.get('scope') or {}).get('trip')) for f in facts if f.get('predicate')=='completion' and f.get('value') is True and f.get('lifecycle')=='active'}
    if view=='upcoming' or qi.get('exclude_completed_trip'):xs=[f for f in xs if str((f.get('scope') or {}).get('trip')) not in completed]
    rank={'explicitly_preferred_external':3,'user':2,'assistant':1,'unspecified':0};groups=defaultdict(list)
    for f in xs:groups[(f.get('entity'),f.get('predicate'),canon(f.get('scope') or {}))].append(f)
    out=[]
    for group in groups.values():
        if view=='historical':out.extend(group);continue
        ma=max(rank.get(str(x.get('authority')),0) for x in group);g=[x for x in group if rank.get(str(x.get('authority')),0)==ma]
        mv=max((int(x.get('version') or 1),int(x.get('sequence') or 0)) for x in g);out.extend([x for x in g if (int(x.get('version') or 1),int(x.get('sequence') or 0))==mv])
    return out
