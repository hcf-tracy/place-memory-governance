from __future__ import annotations
import copy, re, unicodedata, calendar
from pathlib import Path
from .io import read_jsonl
from .backend import record_text,ASSISTANT_PREFIX
from .css import fact_from_event_local,canon
from .artifact_registry import RuntimeArtifactRegistry

VERSION='hf13-semantic-measurement-v2.6'

def _reg(x):
    if x is None:return None
    if isinstance(x,RuntimeArtifactRegistry):return x
    if isinstance(x,(str,Path)):return RuntimeArtifactRegistry.load(x)
    if isinstance(x,dict):return RuntimeArtifactRegistry(x)
    raise TypeError(type(x))

def norm(x):
    s=unicodedata.normalize('NFKC',str('' if x is None else x)).casefold()
    s=s.replace('’',"'").replace('`',"'").replace('_',' ')
    s=s.replace('a.m.','am').replace('p.m.','pm').replace('a.m','am').replace('p.m','pm')
    s=s.replace('nt$','nt ').replace('ntd','nt ')
    s=re.sub(r'(?<=\d),(?=\d{3}\b)','',s)
    s=re.sub(r'(?<=[a-z0-9])-+(?=[a-z0-9])',' ',s)
    s=re.sub(r'[^\w:$%\'+]+',' ',s)
    return ' '.join(s.split())

def _phrase(textn,ph):
    p=norm(ph)
    if not p:return False
    return bool(re.search(rf'(?<!\w){re.escape(p)}(?!\w)',textn))

def _variants(v):
    raw=str(v).strip();low=raw.casefold();out=[raw]
    m=re.fullmatch(r'(20\d{2})-(\d{2})-(\d{2})',low)
    if m:
        y,mo,d=map(int,m.groups());mon=calendar.month_name[mo];abbr=calendar.month_abbr[mo]
        out += [f'{mon} {d}',f'{abbr} {d}',f'{mon} {d} {y}',f'{abbr} {d} {y}',f'{mo}/{d}',f'{mo:02d}/{d:02d}',f'{d} {mon}',f'{d} {abbr}']
    m=re.fullmatch(r'([01]?\d|2[0-3]):([0-5]\d)',low)
    if m:
        h,mi=map(int,m.groups());ap='am' if h<12 else 'pm';h12=h%12 or 12
        out += [f'{h}:{mi:02d}',f'{h:02d}:{mi:02d}',f'{h12}:{mi:02d} {ap}',f'{h12}:{mi:02d}{ap}']
        if mi==0:out += [f'{h12} {ap}',f'{h12}{ap}']
    if re.fullmatch(r'[-+]?\d+(?:\.\d+)?',low):
        try:
            n=float(low)
            if n.is_integer():out += [str(int(n)),f'{int(n):,}']
        except Exception:pass
    return list(dict.fromkeys(out))

def scalar_match(text,value):
    tn=norm(text)
    return any(_phrase(tn,x) for x in _variants(value))

def _leaf_values(v):
    if isinstance(v,dict):
        if v.get('text') is not None and ('artifact_id' in v or 'ordinal' in v):return [v['text']]
        out=[]
        for k,x in v.items():
            if k in {'binding_id','artifact_id','ordinal','selected_ordinal','source_turn_id'}:continue
            out.extend(_leaf_values(x))
        return out
    if isinstance(v,list):
        out=[]
        for x in v:out.extend(_leaf_values(x))
        return out
    return [v]

def value_match(text,value,surfaces=()):
    # Descriptive surface forms are evaluator-only annotations of the exact user claim.
    if any(scalar_match(text,s) for s in surfaces if str(s).strip()):return True
    if isinstance(value,bool):return False
    leaves=[x for x in _leaf_values(value) if x is not None and str(x).strip() and str(x).casefold() not in {'true','false','none','null'}]
    return bool(leaves) and all(scalar_match(text,x) for x in leaves)

class SemanticEvidenceResolver:
    def __init__(self,data_root,artifact_registry=None):
        root=Path(data_root);self.root=root;self.reg=_reg(artifact_registry)
        self.turns=read_jsonl(root/'trajectory_raw.jsonl');self.order={t['turn_id']:i for i,t in enumerate(self.turns)}
        self.events=read_jsonl(root/'trusted_events.jsonl');self.event_by_turn={e['turn_id']:e for e in self.events}
        dp=root/'hidden/descriptive_event_annotations.jsonl';self.desc={x['event_id']:x for x in read_jsonl(dp)} if dp.exists() else {}
        self._resolved={e['event_id']:self._resolve(e) for e in self.events}
    def _resolve(self,e):
        x=copy.deepcopy(e);v=(x.get('assertion') or {}).get('value')
        if isinstance(v,dict) and set(v)=={'binding_id'} and self.reg is not None:
            try:x['assertion']['value']=self.reg.resolve_id(v['binding_id'])
            except Exception:pass
        return x
    def surfaces(self,e):return list((self.desc.get(e['event_id']) or {}).get('surface_forms') or [])
    def event_evidence(self,text,e):
        er=self._resolved.get(e['event_id'],e);ass=er.get('assertion') or {}
        return 'value' in ass and value_match(text,ass.get('value'),self.surfaces(er))
    def fact_evidence(self,text,fact):
        e=self.event_by_turn.get(str(fact.get('source_turn_id') or ''))
        if e is not None:
            er=self._resolved.get(e['event_id'],e);tar=er.get('target') or {}
            if (not tar.get('entity') or tar.get('entity')==fact.get('entity')) and (not tar.get('predicate') or tar.get('predicate')==fact.get('predicate')):
                if value_match(text,fact.get('value'),self.surfaces(er)):return True
        return value_match(text,fact.get('value'),())
    def support_metrics(self,q,expected,ineligible):
        items=[str(x) for x in ((q or {}).get('model_visible_support') or [])]
        req_hits=[any(self.fact_evidence(item,f) for item in items) for f in expected]
        exposed=[];match_detail=[]
        for f in ineligible:
            hits=[i for i,item in enumerate(items) if self.fact_evidence(item,f)]
            if hits:exposed.append(f);match_detail.append({'entity':f.get('entity'),'predicate':f.get('predicate'),'value':f.get('value'),'support_item_indexes':hits})
        return {'needed_sra':1.0 if (not expected or all(req_hits)) else 0.0,'required_hits':req_hits,'exposure':1.0 if exposed else 0.0,'exposed_facts':exposed,'exposure_match_detail':match_detail,'support_items':len(items)}

    @staticmethod
    def _rid(r):
        return str(r.get('id') or r.get('memory_id') or r.get('raw_id') or r.get('physical_id') or '')

    def compile_record_index(self, records):
        """Precompute event->physical-record evidence over a frozen trace universe."""
        uniq={}
        for r in records:
            rid=self._rid(r) or ('text:'+str(hash(record_text(r))))
            uniq[rid]=r
        out={}
        for e0 in self.events:
            e=self._resolved[e0['event_id']];ass=e.get('assertion') or {}
            if 'value' not in ass:continue
            value=ass.get('value')
            if isinstance(value,dict) and value.get('binding_id'):continue
            surfaces=self.surfaces(e);cand=[]
            for rid,r in uniq.items():
                tx=record_text(r)
                if tx.startswith(ASSISTANT_PREFIX):continue
                if value_match(tx,value,surfaces):
                    nt=set(norm(tx).split());score=max([len(nt&set(norm(sf).split())) for sf in surfaces] or [0])
                    cand.append((score,rid))
            out[e['event_id']]=sorted(cand,key=lambda z:(-z[0],z[1]))
        return {'records':uniq,'candidates':out}

    def project_indexed(self, records, index, *, through_turn_id=None, include_through=True):
        lim=self.order.get(through_turn_id,10**9) if through_turn_id else 10**9
        current={}
        for r in records:
            rid=self._rid(r) or ('text:'+str(hash(record_text(r))))
            current[rid]=r
        used=set();facts=[];matches=0
        for seq,e0 in enumerate(self.events,1):
            if not (self.order.get(e0['turn_id'],10**9)<lim or (include_through and self.order.get(e0['turn_id'],10**9)==lim)):continue
            e=self._resolved[e0['event_id']];ass=e.get('assertion') or {}
            if 'value' not in ass:continue
            value=ass.get('value')
            if isinstance(value,dict) and value.get('binding_id'):continue
            choices=[(score,rid) for score,rid in index['candidates'].get(e['event_id'],[]) if rid in current and rid not in used]
            if not choices:continue
            score,rid=max(choices,key=lambda z:(z[0],-list(current).index(z[1]) if z[1] in current else 0));used.add(rid);matches+=1;r=current[rid]
            f=fact_from_event_local(e,sequence=seq,value=value,fact_id=f"eval-v26:{e['event_id']}:{seq}")
            if f is None:continue
            f.version=1+sum(1 for x in facts if x['entity']==f.entity and x['predicate']==f.predicate and canon(x.get('scope') or {})==canon(f.scope))
            x=f.to_dict();x['physical_ids']=[rid];facts.append(x)
        return {'facts':facts,'matched_event_count':matches,'physical_record_count':len(current),'event_values_synthesized':False,'evaluator_only':True,'version':VERSION}
    def project(self,records,*,through_turn_id=None,include_through=True):
        lim=self.order.get(through_turn_id,10**9) if through_turn_id else 10**9
        usable=[self._resolved[e['event_id']] for e in self.events if self.order.get(e['turn_id'],10**9)<lim or (include_through and self.order.get(e['turn_id'],10**9)==lim)]
        physical=[r for r in records if not record_text(r).startswith(ASSISTANT_PREFIX)];facts=[];used=set();matches=0
        # Pre-normalize physical texts once per projection.
        texts=[record_text(r) for r in physical]
        for seq,e in enumerate(usable,1):
            ass=e.get('assertion') or {}
            if 'value' not in ass:continue
            value=ass.get('value')
            if isinstance(value,dict) and value.get('binding_id'):continue
            surfaces=self.surfaces(e);cands=[]
            for ri,tx in enumerate(texts):
                if ri in used:continue
                if value_match(tx,value,surfaces):
                    # Prefer the record with the strongest event-surface token overlap.
                    nt=set(norm(tx).split());score=max([len(nt&set(norm(s).split())) for s in surfaces] or [0]);cands.append((score,-ri,ri))
            if not cands:continue
            _,_,ri=max(cands);used.add(ri);matches+=1;r=physical[ri]
            f=fact_from_event_local(e,sequence=seq,value=value,fact_id=f"eval-v26:{e['event_id']}:{seq}")
            if f is None:continue
            f.version=1+sum(1 for x in facts if x['entity']==f.entity and x['predicate']==f.predicate and canon(x.get('scope') or {})==canon(f.scope))
            x=f.to_dict();x['physical_ids']=[str(r.get('id') or r.get('memory_id') or r.get('raw_id') or r.get('physical_id') or '')];facts.append(x)
        return {'facts':facts,'matched_event_count':matches,'physical_record_count':len(physical),'event_values_synthesized':False,'evaluator_only':True,'version':VERSION}
