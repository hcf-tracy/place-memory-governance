from __future__ import annotations
import copy,hashlib,json,re
from pathlib import Path
from typing import Any
from .io import read_jsonl,write_json
from .llm import answer_natural_turn,get_last_diagnostics

def _selection_ordinals(binding):
    sel=binding.get('selection') or {}
    if 'ordinal' in sel:return [int(sel['ordinal'])]
    if 'ordinals' in sel:return [int(x) for x in sel['ordinals']]
    return [int(x) for x in binding.get('selection_ordinals') or []]

def _required(bindings):
    req={}
    for b in bindings:
        ns=_selection_ordinals(b);m=max([x for x in ns if x>0],default=0)
        if any(x==-1 for x in ns):m=max(m,3)
        req[b['source_turn_id']]=max(req.get(b['source_turn_id'],0),m)
    return req

def _items(answer,source_turn_id):
    xs=[]
    for raw in str(answer).splitlines():
        s=re.sub(r'^\s*(?:[-*•]\s*)?(?:\d+[.)]\s*)?','',raw.strip()).strip()
        if s:xs.append(s)
    out=[]
    for i,text in enumerate(xs,1):
        digest=hashlib.sha256(f'{source_turn_id}\0{i}\0{text}'.encode()).hexdigest()[:16]
        out.append({'artifact_id':f'runtime:{source_turn_id}:{i}:{digest}','source_turn_id':source_turn_id,'ordinal':i,'text':text})
    return out

class RuntimeArtifactRegistry:
    def __init__(self,obj=None):
        self.obj=copy.deepcopy(obj or {'schema_version':'hf13-runtime-artifact-registry-v1','responses':{},'bindings':{},'provider_diagnostics':[],'status':'COMPLETE_CLEAN'})
    @classmethod
    def load(cls,path):return cls(json.loads(Path(path).read_text()))
    def save(self,path):write_json(path,self.obj);return self.obj
    def register_response(self,turn_id,answer,min_items=1):
        items=_items(answer,turn_id)
        if len(items)<int(min_items):raise RuntimeError(f'artifact source {turn_id} produced {len(items)} items; need {min_items}')
        self.obj['responses'][turn_id]={'turn_id':turn_id,'answer':str(answer),'items':items}
    def resolve_binding(self,binding):
        src=self.obj['responses'].get(binding['source_turn_id']) or {};items=src.get('items') or [];out=[]
        for n in _selection_ordinals(binding):
            idx=len(items)-1 if n==-1 else n-1
            if not 0<=idx<len(items):raise RuntimeError(f"binding {binding['binding_id']} cannot resolve ordinal {n} from actual response {binding['source_turn_id']}")
            x=copy.deepcopy(items[idx]);x['selected_ordinal']=n;out.append(x)
        val=out[0] if len(out)==1 else out
        self.obj.setdefault('bindings',{})[binding['binding_id']]={'binding_id':binding['binding_id'],'source_turn_id':binding['source_turn_id'],'resolved_value':copy.deepcopy(val)}
        return val
    def resolve_id(self,binding_id):
        x=(self.obj.get('bindings') or {}).get(binding_id)
        if not x:raise KeyError(f'Runtime binding not resolved: {binding_id}')
        return copy.deepcopy(x['resolved_value'])
    def resolve_object(self,obj):
        if isinstance(obj,dict):
            if set(obj)=={'binding_id'}:return self.resolve_id(obj['binding_id'])
            return {k:self.resolve_object(v) for k,v in obj.items()}
        if isinstance(obj,list):return [self.resolve_object(x) for x in obj]
        return copy.deepcopy(obj)

def build_shared_registry(data_root='data/longitudinal',*,cfg=None,mode='reference',output=None):
    root=Path(data_root);turns=read_jsonl(root/'trajectory_raw.jsonl');bindings=read_jsonl(root/'runtime_bindings.jsonl');req=_required(bindings);by_source={}
    for b in bindings:by_source.setdefault(b['source_turn_id'],[]).append(b)
    reg=RuntimeArtifactRegistry();ctx=[];conv=None;diags=[]
    for t in turns:
        if conv!=t['conversation_id']:ctx=[];conv=t['conversation_id']
        ctx.append({'role':'user','content':t['utterance']})
        if t['turn_id'] not in req:continue
        n=req[t['turn_id']]
        if mode=='reference':answer='\n'.join(f'{i}. Runtime candidate {i} for {t["turn_id"]}' for i in range(1,max(1,n)+1));diag={'status':'clean','attempts':0,'errors':[]}
        else:
            answer=answer_natural_turn(t['utterance'],cfg or {},context=ctx[:-1],memory_support=[],min_numbered_items=n);diag=get_last_diagnostics();diags.append({'turn_id':t['turn_id'],**diag})
        reg.register_response(t['turn_id'],answer,n);ctx.append({'role':'assistant','content':answer})
        for b in by_source.get(t['turn_id'],[]):reg.resolve_binding(b)
    status='FAILED' if any(d.get('status')=='failed' for d in diags) else ('COMPLETE_WITH_RECOVERED_ERRORS' if any(d.get('status')=='recovered' for d in diags) else 'COMPLETE_CLEAN')
    reg.obj['mode']=mode;reg.obj['source_count']=len(reg.obj['responses']);reg.obj['binding_count']=len(reg.obj['bindings']);reg.obj['provider_diagnostics']=diags;reg.obj['status']=status
    if output:reg.save(output)
    return reg
