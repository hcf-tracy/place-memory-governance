from __future__ import annotations
import copy,re
from typing import Any
from .backend import record_text, ASSISTANT_PREFIX

STOP={'the','a','an','is','are','was','were','to','of','for','and','or','in','on','my','our','what','which','who','when','where','how','current','now','should','use','memory','remembered','fact','target'}
SENSITIVE_HINTS={'passport','card','credential','token','private','sensitive','access','code','number'}

def _rid(r):return str(r.get('id') or r.get('memory_id') or r.get('raw_id') or r.get('physical_id') or '')
def _tokens(x):return [t for t in re.findall(r'[a-z0-9]+',str(x).casefold()) if t not in STOP and len(t)>1]
def _target_terms(query:dict[str,Any]):
    target=query.get('target') or {};terms=[]
    for x in (target.get('entity'),target.get('predicate')):
        if x:terms.extend(_tokens(str(x).replace('_',' ').replace('.',' ')))
    for sel in ((query.get('query_intent') or {}).get('selectors') or []):
        for x in (sel.get('entity'),sel.get('predicate')):
            if x:terms.extend(_tokens(str(x).replace('_',' ').replace('.',' ')))
    return list(dict.fromkeys(terms))

def project_physical_records(records:list[dict[str,Any]])->list[dict[str,Any]]:
    """General physical semantic readback independent of trusted write events.

    The output is intentionally record-level rather than benchmark-slot-level. It
    is derived exclusively from bytes/metadata that physically exist in storage.
    """
    out=[]
    for seq,r in enumerate(records,1):
        text=record_text(r);rid=_rid(r)
        if not rid:continue
        md=copy.deepcopy(r.get('metadata') or {})
        role='assistant_non_authoritative' if text.startswith(ASSISTANT_PREFIX) or md.get('hf13_role')=='assistant' else 'user_or_native'
        clean=text[len(ASSISTANT_PREFIX):] if text.startswith(ASSISTANT_PREFIX) else text
        toks=set(_tokens(clean))
        out.append({'physical_id':rid,'physical_ids':[rid],'text':clean,'sequence':seq,'role':role,'tokens':sorted(toks),'sensitive_hint':bool(toks&SENSITIVE_HINTS)})
    return out

def authorize_physical_readback(readback:list[dict[str,Any]],trusted_query:dict[str,Any],*,raw_query:str='',limit:int|None=None)->list[dict[str,Any]]:
    """QASC-only runtime selector over physical readback + trusted query.

    No trusted write event, hidden gold, or CSS transition semantics are accepted.
    The policy is deliberately conservative: when physical records do not contain
    enough query-target evidence, QASC abstains rather than reconstructing a fact.
    """
    if not trusted_query:return []
    qi=trusted_query.get('query_intent') or {};view=qi.get('temporal_view','current')
    # Sensitive values require an explicit physically represented permission model;
    # Native records provide none, so the safe read-plane behavior is abstention.
    if qi.get('request_sensitive_value') or qi.get('check_archival_retention'):return []
    target_terms=set(_target_terms(trusted_query));raw_terms=set(_tokens(raw_query));scored=[]
    for r in readback:
        toks=set(r.get('tokens') or [])
        target_overlap=len(toks&target_terms);raw_overlap=len(toks&raw_terms)
        # Require actual target evidence whenever the trusted query names a target.
        if target_terms and target_overlap==0:continue
        if r.get('role')=='assistant_non_authoritative' and not qi.get('request_provenance') and not qi.get('request_provenance_reason'):role_penalty=3
        else:role_penalty=0
        score=4*target_overlap+raw_overlap-role_penalty
        if score<=0:continue
        scored.append((score,int(r.get('sequence') or 0),r))
    if not scored:return []
    scored.sort(key=lambda z:(z[0],z[1]),reverse=True)
    if view=='historical':
        chosen=[x[2] for x in scored[:max(2,min(4,len(scored)))]]
    else:
        # Current/upcoming semantics: select the latest among the strongest physical
        # candidates. This is record recency, not replayed lifecycle information.
        best=max(x[0] for x in scored);strong=[x for x in scored if x[0]>=best-1];latest=max(x[1] for x in strong);chosen=[x[2] for x in strong if x[1]==latest]
        if qi.get('request_provenance') or qi.get('request_provenance_reason'):
            # A second strong physical record may justify provenance, but never a
            # synthesized source preference.
            for _,_,r in scored:
                if r not in chosen and len(chosen)<2:chosen.append(r)
    if limit not in (None,0):chosen=chosen[:int(limit)]
    # Return fresh copies and assert physical provenance for every support item.
    out=[copy.deepcopy(x) for x in chosen]
    avail={str(x.get('physical_id')) for x in readback}
    for x in out:
        assert str(x.get('physical_id')) in avail and set(map(str,x.get('physical_ids') or []))<=avail
    return out
