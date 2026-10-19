"""Evaluator-only independent longitudinal oracle.

This intentionally does not call CSSKernel. It maps canonical user-level governance semantics to hidden
state using a separate implementation so CSS correctness is not self-scored.
"""
from __future__ import annotations
import copy
from datetime import datetime
from typing import Any
from .css import fact_from_event_local, canon

def _scope(s):return canon(s or {})
def _time(x):
    if not x:return None
    try:return datetime.fromisoformat(str(x).replace('Z','+00:00'))
    except Exception:return None

class OracleEngine:
    def __init__(self):self.facts=[];self.seq=0;self.clock=None;self.purged=[]
    def snapshot(self):return copy.deepcopy(self.facts)
    def purged_snapshot(self):return copy.deepcopy(self.purged)
    def _same(self,e,f):return f['entity']==e['target']['entity'] and f['predicate']==e['target']['predicate'] and _scope(f.get('scope'))==_scope(e.get('scope'))
    def _expire(self,t):
        now=_time(t)
        if now:self.clock=now
        if not self.clock:return
        for f in self.facts:
            vu=_time(f.get('valid_until'))
            if f.get('lifecycle')=='active' and vu:
                try:expired=self.clock.astimezone(vu.tzinfo)>vu
                except Exception:expired=False
                if expired:f.update(lifecycle='expired',applicability='historical',control_status='noncontrolling')
    def advance_time(self,t):self._expire(t)
    def apply(self,e:dict[str,Any],as_of=None):
        self._expire(as_of);g=e.get('governance') or {};t=e['target'];ass=e.get('assertion') or {}
        if g.get('update')=='advance_time':
            self.clock=_time(ass.get('value')) or self.clock;self._expire(ass.get('value'));return
        if g.get('update')=='clarify':return
        if g.get('authority_preference'):
            preferred=str(g.get('authority_preference') or (e.get('assertion') or {}).get('value') or '')
            for f in self.facts:
                if f.get('entity')!=t['entity'] or _scope(f.get('scope'))!=_scope(e.get('scope')):continue
                if f.get('lifecycle')!='active' or f.get('control_status')!='controlling':continue
                if str((f.get('provenance') or {}).get('source_type') or '')==preferred:f['authority']='explicitly_preferred_external'
        deferred_retention_intent=(t.get('predicate')=='retention_intent' and g.get('retention')=='do_not_retain' and g.get('completion')!='task')
        purge_entities=set()
        if g.get('completion')=='task' and g.get('retention')=='do_not_retain':
            escope=_scope(e.get('scope') or {})
            for f in self.facts:
                if f.get('predicate')=='retention_intent' and f.get('retention')=='do_not_retain' and _scope(f.get('scope'))==escope:
                    purge_entities.add(f.get('entity'))
        if t.get('entity') and g.get('retention')=='do_not_retain' and not deferred_retention_intent and t.get('entity')!='flight_booking_form':
            purge_entities.add(t['entity'])
        if purge_entities:
            doomed=[f for f in self.facts if f.get('entity') in purge_entities]
            if doomed:
                for f in doomed:
                    z=copy.deepcopy(f);z['_purged_by_turn_id']=e.get('turn_id');z['_purged_by_event_id']=e.get('event_id');self.purged.append(z)
                self.facts=[f for f in self.facts if f.get('entity') not in purge_entities]
            if t.get('entity') in purge_entities:return
        current=[f for f in self.facts if self._same(e,f) and f.get('lifecycle')=='active' and f.get('control_status')=='controlling']
        if g.get('update')=='suspend':
            for f in current:f.update(lifecycle='suspended',applicability='historical',control_status='noncontrolling')
            if t['predicate']=='hotel_candidate':return
        if g.get('update')=='restore':
            cand=[f for f in self.facts if self._same(e,f) and f.get('lifecycle') in {'suspended','archived','inactive'}]
            if cand:max(cand,key=lambda z:z.get('sequence',0)).update(lifecycle='active',applicability='current',control_status='controlling')
            return
        if g.get('update')=='remove_current':
            for f in current:f.update(lifecycle='archived' if g.get('retention') in {'history','preserve_future'} else 'inactive',applicability='historical',control_status='noncontrolling')
        if g.get('collection_mutation')=='remove_selected_item':
            rel=[f for f in self.facts if f['entity']==t['entity'] and 'restaurant_shortlist' in f['predicate'] and f.get('lifecycle')=='active' and isinstance(f.get('value'),list)]
            if rel:
                src=max(rel,key=lambda z:z.get('sequence',0));new=list(src['value'][1:]);src.update(lifecycle='superseded',applicability='historical',control_status='noncontrolling');self._add(e,new);return
        if 'value' not in ass:return
        value=copy.deepcopy(ass['value'])
        if value=='remove_first_saved' and g.get('collection_mutation')=='remove_selected_item':return
        same=[f for f in current if canon(f.get('value'))==canon(value)]
        if same:
            # Reaffirmation changes source-local epistemic/provenance but not identity/version.
            src=fact_from_event_local(e,sequence=self.seq or 1,value=value)
            f=max(same,key=lambda z:z.get('sequence',0));
            for k in ('epistemic','authority','action_authority','provenance','valid_until'):f[k]=copy.deepcopy(getattr(src,k))
            return
        replace=bool(g.get('update')=='replace' or (current and not isinstance(value,(list,dict))))
        if replace:
            for f in current:f.update(lifecycle='superseded',applicability='historical',control_status='noncontrolling')
        self._add(e,value)
        if g.get('dependency_recompute_of'):
            key=str(g['dependency_recompute_of'])
            for f in self.facts:
                if f.get('source_turn_id')==e.get('turn_id'):continue
                if any(f"{d.get('entity')}.{d.get('predicate')}"==key for d in f.get('depends_on') or []) and f.get('lifecycle')=='active':f.update(epistemic='stale',control_status='noncontrolling')
    def _add(self,e,value):
        self.seq+=1;obj=fact_from_event_local(e,sequence=self.seq,value=value,fact_id=f"gold:{e['event_id']}:{self.seq}").to_dict()
        prior=[x for x in self.facts if x['entity']==obj['entity'] and x['predicate']==obj['predicate'] and _scope(x.get('scope'))==_scope(obj.get('scope'))]
        obj['version']=1+max([int(x.get('version',1)) for x in prior],default=0)
        g=e.get('governance') or {}
        if g.get('applicability')=='historical' and g.get('future_use')=='deny':obj.update(lifecycle='archived',applicability='historical',control_status='noncontrolling')
        if obj['predicate'] in {'former_branch','followup_retention'}:obj.update(lifecycle='archived',applicability='historical',control_status='noncontrolling')
        self.facts.append(obj)
