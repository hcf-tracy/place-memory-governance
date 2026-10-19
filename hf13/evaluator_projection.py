"""Evaluator-only physical-to-semantic projection for RQ1/state scoring.

This module may read trusted user-level events because it is evaluator-only. Runtime
QASC never imports it and never receives its output as support.
"""
from __future__ import annotations
import copy,re
from pathlib import Path
from .io import read_jsonl
from .backend import record_text,ASSISTANT_PREFIX
from .css import fact_from_event_local,canon

def _tokens(x):return set(re.findall(r'[a-z0-9]+',str(x).casefold()))
def _literal_values(v):
    if isinstance(v,dict):return [str(x) for x in v.values() if not isinstance(x,(dict,list))]
    if isinstance(v,list):return [str(x) for x in v if not isinstance(x,(dict,list))]
    return [str(v)]
def _value_evidence(text,value,surface_forms=None):
    t=str(text).casefold();vals=[x.casefold() for x in _literal_values(value) if x not in ('True','False','None') and len(x.strip())>=2]
    if not isinstance(value,(dict,list)) and vals:return any(v in t for v in vals)
    surfaces=[str(x).casefold() for x in (surface_forms or []) if len(str(x).strip())>=3]
    if vals and any(v in t for v in vals):return True
    return bool(surfaces and sum(1 for s in surfaces if s in t)>=min(2,len(surfaces)))

class EvaluatorPhysicalProjector:
    """Physical-gated semantic alignment used only after runtime output exists."""
    def __init__(self,data_root,artifact_registry=None):
        root=Path(data_root);self.turns=read_jsonl(root/'trajectory_raw.jsonl');self.events=read_jsonl(root/'trusted_events.jsonl');self.order={t['turn_id']:i for i,t in enumerate(self.turns)};self.registry=artifact_registry
        # Surface forms moved out of runtime event schema; evaluator may access the
        # hidden descriptive authoring annotations for alignment/scoring only.
        desc_path=root/'hidden/descriptive_event_annotations.jsonl';self.desc={x['event_id']:x for x in read_jsonl(desc_path)} if desc_path.exists() else {}
    def _resolved_event(self,e):
        x=copy.deepcopy(e);v=(x.get('assertion') or {}).get('value')
        if isinstance(v,dict) and set(v)=={'binding_id'}:
            if self.registry is None:return x
            try:x['assertion']['value']=self.registry.resolve_id(v['binding_id'])
            except Exception:return x
        return x
    def project(self,records,*,through_turn_id=None,include_through=True):
        lim=self.order.get(through_turn_id,10**9) if through_turn_id else 10**9
        usable=[self._resolved_event(e) for e in self.events if self.order.get(e['turn_id'],10**9)<lim or (include_through and self.order.get(e['turn_id'],10**9)==lim)]
        physical=[r for r in records if not record_text(r).startswith(ASSISTANT_PREFIX)];facts=[];used=set();matches=0
        for seq,e in enumerate(usable,1):
            ass=e.get('assertion') or {}
            if 'value' not in ass:continue
            value=ass.get('value')
            if isinstance(value,dict) and value.get('binding_id'):continue
            surfaces=(self.desc.get(e['event_id']) or {}).get('surface_forms') or []
            candidates=[]
            for ri,r in enumerate(physical):
                if ri in used:continue
                tx=record_text(r)
                if _value_evidence(tx,value,surfaces):
                    overlap=len(_tokens(tx)&_tokens(' '.join(map(str,surfaces))));candidates.append((overlap,-ri,ri,r))
            if not candidates:continue
            _,_,ri,r=max(candidates);used.add(ri);matches+=1
            f=fact_from_event_local(e,sequence=seq,value=value,fact_id=f"eval-native:{e['event_id']}:{seq}")
            if f is None:continue
            f.version=1+sum(1 for x in facts if x['entity']==f.entity and x['predicate']==f.predicate and canon(x.get('scope') or {})==canon(f.scope))
            x=f.to_dict();x['physical_ids']=[str(r.get('id') or r.get('memory_id') or r.get('raw_id') or r.get('physical_id') or '')];facts.append(x)
        return {'facts':facts,'matched_event_count':matches,'physical_record_count':len(physical),'event_values_synthesized':False,'evaluator_only':True}
