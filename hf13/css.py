from __future__ import annotations
import copy, json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

SEMANTIC_FIELDS=(
    'entity','predicate','value','scope','lifecycle','epistemic','authority','permission',
    'retention','applicability','control_status','action_authority','depends_on','valid_until','version'
)

def canon(v:Any)->str:
    return json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':'),default=str)

def _norm_semantic(v:Any):
    if isinstance(v,dict):
        if 'artifact_id' in v:
            return {'artifact_id':v.get('artifact_id'),'source_turn_id':v.get('source_turn_id'),'ordinal':v.get('ordinal')}
        return {k:_norm_semantic(x) for k,x in v.items() if k!='text'}
    if isinstance(v,list):return [_norm_semantic(x) for x in v]
    return v

def fact_signature(f:dict[str,Any])->tuple:
    return tuple(canon(_norm_semantic(f.get(k))) for k in SEMANTIC_FIELDS)

def state_signature(facts:list[dict[str,Any]])->list[tuple]:
    return sorted(fact_signature(f) for f in facts)

def _scope_key(s): return canon(s or {})
def _parse_time(x):
    if not x:return None
    try:return datetime.fromisoformat(str(x).replace('Z','+00:00'))
    except Exception:return None

def _derive_authority(event:dict)->str:
    # Provenance identity is not governance precedence. A source being an airline,
    # hotel, itinerary, etc. never silently makes it the controlling winner.
    src=str((event.get('provenance') or {}).get('source_type') or 'user_report').casefold()
    if 'assistant' in src:return 'assistant'
    return 'user'

def _derive_action_authority(gov:dict)->str:
    return str(gov.get('action_authority') or 'unspecified')

def _derive_permission(gov:dict)->str:
    return 'restricted' if gov.get('data_access')=='restricted' else 'allowed'

def _derive_retention(gov:dict)->str:
    x=str(gov.get('retention') or 'retain')
    return 'do_not_retain' if x=='do_not_retain' else ('history' if x=='history' else 'retain')

def _derive_local_lifecycle(event:dict)->tuple[str,str,str]:
    gov=event.get('governance') or {}; ep=str((event.get('epistemic') or {}).get('status') or 'confirmed')
    lifecycle='active'; applicability='current'; control='controlling'
    if gov.get('applicability')=='hypothetical' or ep=='hypothetical':
        lifecycle='hypothetical'; applicability='hypothetical'; control='noncontrolling'
    elif gov.get('applicability')=='historical':
        lifecycle='archived'; applicability='historical'; control='noncontrolling'
    elif gov.get('update')=='suspend':
        lifecycle='archived'; applicability='historical'; control='noncontrolling'
    return lifecycle,applicability,control

@dataclass
class Fact:
    fact_id:str; entity:str; predicate:str; value:Any; scope:dict[str,Any]=field(default_factory=dict)
    lifecycle:str='active'; epistemic:str='confirmed'; authority:str='user'; permission:str='allowed'
    retention:str='retain'; applicability:str='current'; control_status:str='controlling'; action_authority:str='unspecified'
    provenance:dict[str,Any]=field(default_factory=dict); sequence:int=0; version:int=1; source_turn_id:str=''
    valid_until:str|None=None; depends_on:list[dict[str,Any]]=field(default_factory=list); physical_ids:list[str]=field(default_factory=list)
    def to_dict(self):return copy.deepcopy(self.__dict__)
    @classmethod
    def from_dict(cls,d):return cls(**{k:copy.deepcopy(v) for k,v in d.items() if k in cls.__dataclass_fields__})


def fact_from_event_local(event:dict[str,Any],*,sequence:int=0,value:Any=None,fact_id:str|None=None)->Fact|None:
    ass=event.get('assertion') or {}
    if value is None and 'value' not in ass:return None
    if value is None:value=copy.deepcopy(ass.get('value'))
    gov=event.get('governance') or {}; lifecycle,applicability,control=_derive_local_lifecycle(event)
    return Fact(
        fact_id=fact_id or f"physical:{event['event_id']}:{sequence or 1}",entity=event['target']['entity'],predicate=event['target']['predicate'],
        value=copy.deepcopy(value),scope=copy.deepcopy(event.get('scope') or {}),lifecycle=lifecycle,
        epistemic=str((event.get('epistemic') or {}).get('status') or 'confirmed'),authority=_derive_authority(event),
        permission=_derive_permission(gov),retention=_derive_retention(gov),applicability=applicability,control_status=control,
        action_authority=_derive_action_authority(gov),provenance=copy.deepcopy(event.get('provenance') or {}),sequence=sequence,
        version=1,source_turn_id=event['turn_id'],valid_until=(event.get('temporal') or {}).get('valid_until'),depends_on=copy.deepcopy(event.get('depends_on') or [])
    )

class CSSKernel:
    """Deterministic state-formation kernel. Input is user-level meaning; output transitions are internal."""
    def __init__(self):self.facts:list[Fact]=[];self.sequence=0;self.clock:datetime|None=None;self.transitions=[]
    def snapshot(self):return [f.to_dict() for f in self.facts]
    def _same_slot(self,event,f):return f.entity==event['target']['entity'] and f.predicate==event['target']['predicate'] and _scope_key(f.scope)==_scope_key(event.get('scope') or {})
    def _current(self,event):return [f for f in self.facts if self._same_slot(event,f) and f.lifecycle=='active' and f.control_status=='controlling']
    def _versions(self,event):return [f for f in self.facts if self._same_slot(event,f)]
    def _expire(self,as_of):
        now=_parse_time(as_of)
        if now:self.clock=now
        if not self.clock:return []
        ops=[]
        for f in self.facts:
            vu=_parse_time(f.valid_until)
            if f.lifecycle=='active' and vu:
                try:expired=self.clock.astimezone(vu.tzinfo)>vu
                except Exception:expired=False
                if expired:
                    f.lifecycle='expired';f.applicability='historical';f.control_status='noncontrolling';ops.append({'op':'EXPIRE','fact_id':f.fact_id})
        return ops
    def advance_time(self,as_of):return self._expire(as_of)
    def apply(self,event:dict[str,Any],*,as_of:str|None=None):
        ops=self._expire(as_of); gov=event.get('governance') or {}; target=event['target']
        if gov.get('update')=='advance_time':
            val=(event.get('assertion') or {}).get('value'); self.clock=_parse_time(val) or self.clock; ops+=self._expire(val); return self._record(event,ops)
        if gov.get('update')=='clarify':
            ops.append({'op':'REQUEST_CLARIFICATION','target':copy.deepcopy(target)});return self._record(event,ops)
        if gov.get('authority_preference'):
            # User-level source preference is input semantics; CSS derives operational
            # precedence only for physically represented facts whose provenance matches.
            preferred=str(gov.get('authority_preference') or (event.get('assertion') or {}).get('value') or '')
            affected=[]
            for f in self.facts:
                if f.entity!=target['entity'] or _scope_key(f.scope)!=_scope_key(event.get('scope') or {}):continue
                if f.lifecycle!='active' or f.control_status!='controlling':continue
                if str((f.provenance or {}).get('source_type') or '')==preferred:
                    f.authority='explicitly_preferred_external';affected.append(f.fact_id)
            ops.append({'op':'SET_SOURCE_PREFERENCE','preferred_source':preferred,'affected_fact_ids':affected})
        # A retention-intent assertion is a policy to apply at its trigger, not
        # the trigger itself. Example: "do not retain after the form" remains live
        # until the corresponding task-completion event arrives. CSS derives PURGE
        # only from an actual trigger or an immediate do-not-retain command.
        deferred_retention_intent=(target.get('predicate')=='retention_intent' and gov.get('retention')=='do_not_retain' and gov.get('completion')!='task')
        purge_entities=set()
        if gov.get('completion')=='task' and gov.get('retention')=='do_not_retain':
            escope=_scope_key(event.get('scope') or {})
            for f in self.facts:
                if f.predicate=='retention_intent' and f.retention=='do_not_retain' and _scope_key(f.scope)==escope:
                    purge_entities.add(f.entity)
        if target.get('entity') and gov.get('retention')=='do_not_retain' and not deferred_retention_intent and target.get('entity')!='flight_booking_form':
            purge_entities.add(target['entity'])
        if purge_entities:
            doomed=[f for f in self.facts if f.entity in purge_entities]
            if doomed:
                self.facts=[f for f in self.facts if f.entity not in purge_entities]
                ops += [{'op':'PURGE','fact_id':f.fact_id} for f in doomed]
            if target.get('entity') in purge_entities:return self._record(event,ops)
        current=self._current(event)
        if gov.get('update')=='suspend':
            for f in current:f.lifecycle='suspended';f.applicability='historical';f.control_status='noncontrolling';ops.append({'op':'SUSPEND','fact_id':f.fact_id})
            if target['predicate']=='hotel_candidate':return self._record(event,ops)
        if gov.get('update')=='restore':
            cand=[f for f in self._versions(event) if f.lifecycle in {'suspended','archived','inactive'}]
            if cand:
                f=max(cand,key=lambda z:z.sequence);f.lifecycle='active';f.applicability='current';f.control_status='controlling';ops.append({'op':'RESTORE','fact_id':f.fact_id});return self._record(event,ops)
        if gov.get('update')=='remove_current':
            for f in current:
                f.lifecycle='archived' if gov.get('retention') in {'history','preserve_future'} else 'inactive';f.applicability='historical';f.control_status='noncontrolling';ops.append({'op':'ARCHIVE' if f.lifecycle=='archived' else 'DEACTIVATE','fact_id':f.fact_id})
        # Set-valued shortlist mutation is a real transition, not an assertion replacement.
        if gov.get('collection_mutation')=='remove_selected_item':
            related=[f for f in self.facts if f.entity==target['entity'] and 'restaurant_shortlist' in f.predicate and f.lifecycle=='active' and isinstance(f.value,list)]
            if related:
                src=max(related,key=lambda z:z.sequence);new=list(src.value[1:]) if src.value else []
                src.lifecycle='superseded';src.control_status='noncontrolling';src.applicability='historical';ops.append({'op':'SUPERSEDE','fact_id':src.fact_id})
                return self._add_fact(event,new,ops)
        ass=event.get('assertion') or {}
        if 'value' not in ass:return self._record(event,ops)
        value=copy.deepcopy(ass['value'])
        # structural markers are not themselves resulting semantic values
        if value=='remove_first_saved' and gov.get('collection_mutation')=='remove_selected_item':return self._record(event,ops)
        same=[f for f in current if canon(f.value)==canon(value)]
        if same:
            f=max(same,key=lambda z:z.sequence);f.provenance=copy.deepcopy(event.get('provenance') or f.provenance);f.epistemic=str((event.get('epistemic') or {}).get('status') or f.epistemic);f.authority=_derive_authority(event);f.action_authority=_derive_action_authority(gov)
            if (event.get('temporal') or {}).get('valid_until'):f.valid_until=(event.get('temporal') or {}).get('valid_until')
            ops.append({'op':'REAFFIRM','fact_id':f.fact_id});return self._record(event,ops)
        replace=bool(gov.get('update')=='replace' or (current and not isinstance(value,(list,dict))))
        if replace:
            for f in current:
                f.lifecycle='superseded';f.applicability='historical';f.control_status='noncontrolling';ops.append({'op':'SUPERSEDE','fact_id':f.fact_id})
        return self._add_fact(event,value,ops)
    def _add_fact(self,event,value,ops):
        self.sequence+=1; f=fact_from_event_local(event,sequence=self.sequence,value=value,fact_id=f"hf13:{event['event_id']}:{self.sequence}")
        assert f is not None
        prior=[x for x in self.facts if x.entity==f.entity and x.predicate==f.predicate and _scope_key(x.scope)==_scope_key(f.scope)]
        f.version=1+max([x.version for x in prior],default=0)
        gov=event.get('governance') or {}
        if gov.get('applicability')=='historical' and gov.get('future_use')=='deny':
            f.lifecycle='archived';f.applicability='historical';f.control_status='noncontrolling'
        if f.predicate in {'former_branch','followup_retention'}:
            f.lifecycle='archived';f.applicability='historical';f.control_status='noncontrolling'
        self.facts.append(f);ops.append({'op':'ADD' if f.lifecycle=='active' else f.lifecycle.upper(),'fact_id':f.fact_id})
        if gov.get('dependency_recompute_of'):
            key=str(gov['dependency_recompute_of'])
            for x in self.facts:
                if x is f:continue
                if any(f"{d.get('entity')}.{d.get('predicate')}"==key for d in x.depends_on) and x.lifecycle=='active':
                    x.epistemic='stale';x.control_status='noncontrolling';ops.append({'op':'INVALIDATE_DEPENDENT','fact_id':x.fact_id})
        return self._record(event,ops)
    def _record(self,event,ops):
        row={'event_id':event['event_id'],'turn_id':event['turn_id'],'internal_transitions':copy.deepcopy(ops)};self.transitions.append(row);return row
