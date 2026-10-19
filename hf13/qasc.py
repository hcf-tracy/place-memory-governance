from __future__ import annotations
import copy,json,re
from datetime import datetime
from typing import Any
from .css import canon

def _time(x):
    if not x:return None
    try:return datetime.fromisoformat(str(x).replace('Z','+00:00'))
    except Exception:return None

def _tokens(x):return set(re.findall(r'[a-z0-9]+',str(x).casefold()))
def _literal_values(v):
    if isinstance(v,dict):return [str(x) for x in v.values() if not isinstance(x,(dict,list))]
    if isinstance(v,list):return [str(x) for x in v if not isinstance(x,(dict,list))]
    return [str(v)]
def _value_evidence(text,value,surface_forms=None):
    t=str(text).casefold()
    vals=[x.casefold() for x in _literal_values(value) if x not in ('True','False','None') and len(x.strip())>=2]
    # scalar replacement values require literal evidence; slot words alone never suffice
    if not isinstance(value,(dict,list)) and vals:return any(v in t for v in vals)
    surfaces=[str(x).casefold() for x in (surface_forms or []) if len(str(x).strip())>=3]
    if vals and any(v in t for v in vals):return True
    return bool(surfaces and sum(1 for s in surfaces if s in t)>=min(2,len(surfaces)))

def _scope_match(fscope,qscope):
    if not qscope:return True
    fs=fscope or {}
    for k,v in qscope.items():
        if k in {'trips'}:continue
        if k in fs and fs[k]!=v:return False
    if qscope.get('trips') and fs.get('trip') and fs.get('trip') not in qscope['trips']:return False
    return True

def _eligible_base(f,query,now):
    qi=query.get('query_intent') or {};view=qi.get('temporal_view','current')
    lifecycle=str(f.get('lifecycle') or 'active');ep=str(f.get('epistemic') or 'confirmed')
    if view=='historical':
        if lifecycle not in {'active','superseded','archived','expired','suspended','inactive'}:return False
    else:
        if lifecycle!='active' or f.get('control_status','controlling')!='controlling':return False
    if ep in {'ambiguous','hypothetical','stale'} and view!='historical':return False
    if f.get('permission')=='restricted' and (qi.get('request_sensitive_value') or qi.get('check_archival_retention')):return False
    if f.get('retention')=='do_not_retain':return False
    if f.get('action_authority')=='not_authorized' and qi.get('ask_whether_irreversible_action_authorized'):return True
    vu=_time(f.get('valid_until'))
    if now and vu:
        try:
            if now.astimezone(vu.tzinfo)>vu and view!='historical':return False
        except Exception:pass
    return _scope_match(f.get('scope') or {},query.get('scope') or {})

def authorize(facts:list[dict[str,Any]],query:dict[str,Any],*,as_of=None,physical_record_ids=None)->list[dict[str,Any]]:
    """Return Query-Authorized Support over physically represented facts only.

    This function is a read-plane selector/closure. It never creates missing facts,
    replays CSS transitions, or mutates storage.
    """
    target=query.get('target') or {};qi=query.get('query_intent') or {};selectors=qi.get('selectors') or [];now=_time(as_of or query.get('as_of'))
    cand=[]
    for f in facts:
        # Physical-gating is mandatory for runtime facts.
        pids=[str(x) for x in (f.get('physical_ids') or []) if str(x)]
        if not pids:continue
        if physical_record_ids is not None and not set(pids).issubset(set(map(str,physical_record_ids))):continue
        if selectors:
            ok=False
            for sel in selectors:
                if sel.get('entity') and f.get('entity')!=sel['entity']:continue
                if sel.get('predicate') and f.get('predicate')!=sel['predicate']:continue
                ok=True;break
            if not ok:continue
        if target.get('entity') and f.get('entity')!=target['entity']:continue
        if target.get('predicate') and f.get('predicate')!=target['predicate']:continue
        if not _eligible_base(f,query,now):continue
        cand.append(copy.deepcopy(f))
    completed={str((f.get('scope') or {}).get('trip')) for f in facts if f.get('predicate')=='completion' and f.get('value') is True and f.get('lifecycle')=='active'}
    if qi.get('temporal_view')=='upcoming' or qi.get('exclude_completed_trip'):
        cand=[f for f in cand if str((f.get('scope') or {}).get('trip')) not in completed]
    rank={'explicitly_preferred_external':3,'user':2,'assistant':1,'unspecified':0};grouped={}
    for f in cand:
        key=(f.get('entity'),f.get('predicate'),canon(f.get('scope') or {}));grouped.setdefault(key,[]).append(f)
    out=[]
    for xs in grouped.values():
        if qi.get('temporal_view')=='historical':out.extend(xs);continue
        maxauth=max(rank.get(str(x.get('authority')),0) for x in xs);xs=[x for x in xs if rank.get(str(x.get('authority')),0)==maxauth]
        mv=max((int(x.get('version') or 1),int(x.get('sequence') or 0)) for x in xs);out.extend([x for x in xs if (int(x.get('version') or 1),int(x.get('sequence') or 0))==mv])
    latest={}
    for f in facts:
        if f.get('lifecycle')=='active' and f.get('control_status','controlling')=='controlling':
            k=(f.get('entity'),f.get('predicate'));latest[k]=max(latest.get(k,0),int(f.get('sequence') or 0))
    valid=[]
    for f in out:
        bad=False
        for d in f.get('depends_on') or []:
            dk=(d.get('entity'),d.get('predicate'));at=int(d.get('source_sequence') or 0)
            if at and latest.get(dk,0)>at:bad=True
        if not bad:valid.append(f)
    # Optional bounded semantic closure for provenance/source-preference justification.
    if qi.get('request_provenance') or qi.get('request_provenance_reason'):
        entities={f.get('entity') for f in valid};seen={tuple(f.get('physical_ids') or []) for f in valid}
        for f in facts:
            if f.get('entity') not in entities or f.get('predicate') not in {'source_preference','provenance','authority_preference'}:continue
            if not _eligible_base(f,query,now):continue
            pids=[str(x) for x in (f.get('physical_ids') or []) if str(x)]
            if not pids:continue
            if physical_record_ids is not None and not set(pids).issubset(set(map(str,physical_record_ids))):continue
            key=tuple(pids)
            if key not in seen:valid.append(copy.deepcopy(f));seen.add(key)
    if physical_record_ids is not None:
        avail=set(map(str,physical_record_ids))
        for f in valid:
            assert set(map(str,f.get('physical_ids') or [])).issubset(avail), 'QASC authorized support escaped physical state'
    return valid

