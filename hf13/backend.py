from __future__ import annotations
import copy, hashlib, json, re, shutil
from pathlib import Path
from typing import Any
from .schema import validate_native_metadata
from .css import SEMANTIC_FIELDS, fact_signature, state_signature, canon

STATE_PREFIX='[HF13_STATE] '
ASSISTANT_PREFIX='[HF13_ASSISTANT_NONAUTHORITATIVE] '
OBSERVATION_PREFIX='[HF13_OBSERVATION] '

def record_text(r):return str(r.get('memory') or r.get('text') or r.get('content') or '')
def _rid(r):return str(r.get('id') or r.get('memory_id') or r.get('raw_id') or '')
def _record_snapshot(rows):return sorted((_rid(r),record_text(r),canon(r.get('metadata') or {})) for r in rows)

def _semantic_payload(f):
    x={k:copy.deepcopy(f.get(k)) for k in SEMANTIC_FIELDS}
    x['fact_id']=str(f.get('fact_id') or '')
    x['sequence']=int(f.get('sequence') or 0);x['source_turn_id']=str(f.get('source_turn_id') or '')
    return x

def fact_text(f):return STATE_PREFIX+json.dumps(_semantic_payload(f),ensure_ascii=False,sort_keys=True,separators=(',',':'),default=str)
def parse_fact_text(text):
    s=str(text or '')
    idx=s.find(STATE_PREFIX)
    if idx<0:return None
    line=s[idx+len(STATE_PREFIX):].splitlines()[0].strip()
    try:
        obj=json.loads(line)
        if not isinstance(obj,dict):return None
        return obj
    except Exception:return None

def _hash_fact(f):return hashlib.sha256(canon(_semantic_payload(f)).encode()).hexdigest()

def _project_records(records):
    facts=[];mismatches=[];managed_count=0
    for r in records:
        md=r.get('metadata') if isinstance(r.get('metadata'),dict) else {}
        if not md.get('hf13_managed'):continue
        managed_count+=1;obj=parse_fact_text(record_text(r))
        if not obj:
            mismatches.append({'record_id':_rid(r),'reason':'unparseable_managed_text'});continue
        expected_hash=md.get('hf13_state_sha256');actual_hash=_hash_fact(obj)
        if expected_hash and expected_hash!=actual_hash:mismatches.append({'record_id':_rid(r),'reason':'text_metadata_hash_mismatch'})
        if md.get('hf13_fact_id') and str(md['hf13_fact_id'])!=str(obj.get('fact_id')):mismatches.append({'record_id':_rid(r),'reason':'fact_id_mismatch'})
        x=copy.deepcopy(obj);x['physical_ids']=[_rid(r)];facts.append(x)
    return {'facts':facts,'coverage':{'managed_records':managed_count,'parsed_records':len(facts)},'mismatches':mismatches}

class ReferenceStore:
    def __init__(self):self._rows={};self.counter=0;self.last_native_diagnostics={};self.last_search_diagnostics={}
    def _bucket(self,user_id):return self._rows.setdefault(str(user_id),[])
    def native_add(self,utterance,*,user_id,metadata=None,role='user',max_retries=1):
        md=dict(metadata or {});validate_native_metadata(md);self.counter+=1
        text=str(utterance) if role=='user' else ASSISTANT_PREFIX+str(utterance)[:1200]
        row={'id':f'ref-{self.counter}','memory':text,'metadata':{},'role':role};self._bucket(user_id).append(row)
        self.last_native_diagnostics={'status':'clean','attempts':1,'provider_internal_error_count':0,'error_lines':[]}
        return {'result':{'results':[{'id':row['id'],'event':'ADD'}]},'status':'clean','attempts':1,'diagnostics':copy.deepcopy(self.last_native_diagnostics)}
    def add_observation(self,utterance,*,user_id,observation_id,role='user'):
        self.counter+=1
        text=str(utterance) if role=='user' else ASSISTANT_PREFIX+str(utterance)[:1200]
        md={'hf13_observation':True,'hf13_observation_id':str(observation_id),'hf13_role':str(role),'hf13_infer':False}
        row={'id':f'ref-o-{self.counter}','memory':text,'metadata':md,'role':role};self._bucket(user_id).append(row)
        return {'result':{'results':[{'id':row['id'],'event':'ADD'}]},'status':'clean','attempts':1,'diagnostics':{'status':'clean','attempts':1,'provider_internal_error_count':0,'error_lines':[],'infer':False,'verified':True}}
    def purge_observations(self,*,user_id,observation_ids=None,forbidden_values=None):
        """Physically remove CSS-owned inert observations selected by CSS.

        Inputs are derived only from CSS runtime state/transitions. Hidden evaluator
        gold is deliberately not accepted by this API.
        """
        ids={str(x) for x in (observation_ids or []) if str(x)}
        vals=[str(x).casefold() for x in (forbidden_values or []) if str(x)]
        rows=self._bucket(user_id);removed=[];keep=[]
        for r in rows:
            md=r.get('metadata') or {};oid=str(md.get('hf13_observation_id') or '')
            is_obs=bool(md.get('hf13_observation')) and not bool(md.get('hf13_managed'))
            text=record_text(r).casefold();match=is_obs and (oid in ids or any(v in text for v in vals))
            if match:removed.append(_rid(r))
            else:keep.append(r)
        self._rows[str(user_id)]=keep
        leaked=[_rid(r) for r in self._bucket(user_id) if (r.get('metadata') or {}).get('hf13_observation') and any(v in record_text(r).casefold() for v in vals)]
        if leaked:raise RuntimeError(f'reference CSS observation purge verification failed: {leaked}')
        return {'status':'clean','verified':True,'attempts':1,'removed_count':len(removed),'removed_physical_ids':removed,'provider_internal_error_count':0,'error_lines':[],'infer':False,'hidden_gold_used':False}
    def _managed_rows(self,user_id):return [r for r in self._bucket(user_id) if (r.get('metadata') or {}).get('hf13_managed')]
    def reconcile_managed(self,facts,*,user_id,max_repair_attempts=2):
        desired={str(f['fact_id']):copy.deepcopy(f) for f in facts};rows=self._bucket(user_id)
        existing={str((r.get('metadata') or {}).get('hf13_fact_id')):r for r in rows if (r.get('metadata') or {}).get('hf13_managed')}
        keep=[]
        for r in rows:
            md=r.get('metadata') or {};fid=str(md.get('hf13_fact_id') or '')
            if not md.get('hf13_managed'):keep.append(r);continue
            f=desired.get(fid)
            if f is not None and md.get('hf13_state_sha256')==_hash_fact(f) and parse_fact_text(record_text(r)) is not None:keep.append(r)
        kept_ids={str((r.get('metadata') or {}).get('hf13_fact_id')) for r in keep if (r.get('metadata') or {}).get('hf13_managed')}
        for fid,f in desired.items():
            if fid in kept_ids:continue
            self.counter+=1;keep.append({'id':f'ref-m-{self.counter}','memory':fact_text(f),'metadata':{'hf13_managed':True,'hf13_fact_id':fid,'hf13_state_sha256':_hash_fact(f)}})
        self._rows[str(user_id)]=keep
        audit=self.project_managed(user_id=user_id)
        if audit['mismatches'] or state_signature(audit['facts'])!=state_signature(facts):raise RuntimeError('reference CSS physical certification failed')
        return {'verified':True,'attempts':1,'managed_count':len(audit['facts']),'coverage':audit['coverage'],'mismatches':[],'status':'clean'}
    def diagnostic_inject_managed_fault(self,*,user_id,fact_id,kind,case_id):
        """Evaluator-only physical fault injection for the Fault diagnostic.

        This mutates the isolated diagnostic backend representation. It is never
        used by Primary/runtime governance and never receives evaluator gold.
        """
        rows=self._bucket(user_id);target=None
        for r in rows:
            md=r.get('metadata') or {}
            if md.get('hf13_managed') and str(md.get('hf13_fact_id'))==str(fact_id):target=r;break
        if target is None:raise RuntimeError(f'fault target not physically present: {fact_id}')
        obj=parse_fact_text(record_text(target))
        if obj is None:raise RuntimeError(f'fault target is not parseable: {fact_id}')
        if kind=='delete_current':
            rows.remove(target)
        else:
            z=copy.deepcopy(obj)
            if kind=='duplicate_current':z['fact_id']=str(obj.get('fact_id'))+f':FAULT_DUP:{case_id}'
            elif kind=='contradict_current':
                z['fact_id']=str(obj.get('fact_id'))+f':FAULT_CONTRADICT:{case_id}';z['value']='CONTRADICT-'+str(obj.get('value'))
            elif kind=='corrupt_record':
                rows.remove(target);z['permission']='restricted' if z.get('permission')!='restricted' else 'allowed';z['authority']='assistant';z['control_status']='noncontrolling'
            else:raise ValueError(kind)
            self.counter+=1;fid=str(z.get('fact_id'));rows.append({'id':f'ref-fault-{self.counter}','memory':fact_text(z),'metadata':{'hf13_managed':True,'hf13_fact_id':fid,'hf13_state_sha256':_hash_fact(z),'hf13_fault_injected':True,'hf13_infer':False}})
        audit=self.project_managed(user_id=user_id)
        return {'status':'clean','kind':kind,'case_id':case_id,'target_fact_id':fact_id,'physical_count':len(audit['facts']),'mismatches':audit['mismatches'],'infer':False}
    def project_managed(self,*,user_id):return _project_records(self._bucket(user_id))
    def search(self,query,*,user_id,limit=10,max_retries=2):
        q=set(re.findall(r'[a-z0-9]+',str(query).casefold()));sc=[]
        for i,r in enumerate(self._bucket(user_id)):
            toks=set(re.findall(r'[a-z0-9]+',record_text(r).casefold()));score=len(q&toks)
            sc.append((score,-i,r))
        sc.sort(key=lambda z:(-z[0],-z[1]));self.last_search_diagnostics={'status':'clean','attempts':1,'retry_count':0,'error_lines':[],'read_only':True};return [copy.deepcopy(z[2]) for z in sc[:limit]]
    def get_all(self,*,user_id):return copy.deepcopy(self._bucket(user_id))
    def close(self):pass

class Mem0Store:
    ERROR_TOKENS=('error processing memory action','error in new memory actions response','invalid json response','jsondecodeerror','failed to update','failed to delete','could not parse','exception','traceback')
    def __init__(self,cfg,*,result_root,namespace):
        from lgcm.memory import Mem0Backend
        mem_cfg=copy.deepcopy((cfg.get('memory') or {}).get('mem0_config') or {})
        vs=mem_cfg.setdefault('vector_store',{}).setdefault('config',{});digest=hashlib.sha256(namespace.encode()).hexdigest()[:20]
        scratch=Path(result_root)/'qdrant'/f'ns_{digest}';shutil.rmtree(scratch,ignore_errors=True);scratch.mkdir(parents=True,exist_ok=True)
        vs['path']=str(scratch);vs['collection_name']=re.sub(r'\W+','_',str(vs.get('collection_name') or 'hf13'))[:40]+'_'+digest
        self.scratch=scratch;self.backend=Mem0Backend(mem_cfg);self.mem=self.backend.memory;self.last_native_diagnostics={};self.last_search_diagnostics={}
    def _all(self,user_id):return self.backend.get_all_scope(user_id=user_id,limit=5000)
    def native_add(self,utterance,*,user_id,metadata=None,role='user',max_retries=1):
        md=dict(metadata or {});validate_native_metadata(md)
        text=str(utterance) if role=='user' else ASSISTANT_PREFIX+str(utterance)[:1200]
        messages=[{'role':'user' if role=='user' else 'assistant','content':text}]
        before=_record_snapshot(self._all(user_id));infra_errors=[]
        for attempt in range(1,int(max_retries)+2):
            exc=None;result=None
            try:result=self.backend.add_inferred(messages,user_id=user_id,metadata={})
            except Exception as e:exc=e
            diag=copy.deepcopy(getattr(self.backend,'last_inferred_diagnostics',{}) or {})
            err_lines=list(diag.get('error_lines') or [])
            if exc:err_lines.append(f'{type(exc).__name__}: {exc}')
            after_rows=self._all(user_id);after=_record_snapshot(after_rows);mutated=after!=before
            # Mem0/Qwen may return normally while its own memory-action JSON/action
            # failed. That is an evaluated-system outcome, not a harness failure.
            model_error=bool(err_lines) and exc is None
            if exc is not None:
                low=(type(exc).__name__+' '+str(exc)+' '+' '.join(err_lines)).casefold()
                model_error=any(x in low for x in ('jsondecode','invalid json','could not parse','error processing memory action','memory action','unterminated string'))
            if model_error:
                self.last_native_diagnostics={'status':'model_operation_error','attempts':attempt,'provider_internal_error_count':len(err_lines),'error_lines':err_lines,'mutated':mutated,'evaluated_system_error':True,'infrastructure_error':False}
                return {'result':result,'status':'model_operation_error','attempts':attempt,'diagnostics':copy.deepcopy(self.last_native_diagnostics)}
            if exc is None and not err_lines:
                status='recovered' if infra_errors else 'clean';self.last_native_diagnostics={'status':status,'attempts':attempt,'provider_internal_error_count':len(infra_errors),'error_lines':infra_errors,'mutated':mutated,'evaluated_system_error':False,'infrastructure_error':bool(infra_errors)}
                return {'result':result,'status':status,'attempts':attempt,'diagnostics':copy.deepcopy(self.last_native_diagnostics)}
            infra_errors.extend(err_lines or [f'{type(exc).__name__}: {exc}'])
            if mutated:raise RuntimeError(f'Native Mem0 infrastructure/provider exception after physical mutation; refusing unsafe retry: {infra_errors[-3:]}')
            if attempt>int(max_retries):raise RuntimeError(f'Native Mem0 infrastructure/provider exception not recovered after bounded retry: {infra_errors[-3:]}')
        raise RuntimeError('unreachable native_add retry state')
    def add_observation(self,utterance,*,user_id,observation_id,role='user',max_retries=2):
        # CSS/PLACE observations are explicit storage only. They must never invoke
        # Mem0/Qwen infer=True after CSS has taken control of the state plane.
        text=str(utterance) if role=='user' else ASSISTANT_PREFIX+str(utterance)[:1200]
        oid=str(observation_id);md={'hf13_observation':True,'hf13_observation_id':oid,'hf13_role':str(role),'hf13_infer':False}
        prior_errors=[]
        def present(rows):
            return [r for r in rows if (r.get('metadata') or {}).get('hf13_observation_id')==oid and record_text(r)==text]
        existing=present(self._all(user_id))
        if existing:return {'result':None,'status':'clean','attempts':0,'diagnostics':{'status':'clean','attempts':0,'provider_internal_error_count':0,'error_lines':[],'infer':False,'verified':True,'idempotent_existing':True}}
        for attempt in range(1,int(max_retries)+2):
            exc=None;result=None
            try:result=self.mem.add(text,user_id=user_id,metadata=md,infer=False)
            except Exception as e:exc=e
            rows=self._all(user_id);hits=present(rows)
            if hits:
                status='recovered' if exc is not None or prior_errors else 'clean'
                errs=prior_errors+([f'{type(exc).__name__}: {exc}'] if exc else [])
                return {'result':result,'status':status,'attempts':attempt,'diagnostics':{'status':status,'attempts':attempt,'provider_internal_error_count':len(errs),'error_lines':errs,'infer':False,'verified':True,'physical_ids':[_rid(r) for r in hits]}}
            if exc is not None:prior_errors.append(f'{type(exc).__name__}: {exc}')
            else:prior_errors.append('infer=False observation add returned without a verifiable physical record')
            if attempt>int(max_retries):break
        raise RuntimeError(f'CSS/PLACE infer=False observation write failed verification after bounded retry: {prior_errors[-3:]}')
    def purge_observations(self,*,user_id,observation_ids=None,forbidden_values=None,max_retries=2):
        """Bounded, physically verified deletion of CSS-owned observations.

        The caller supplies observation IDs / literal values derived from CSS's own
        governed state. This function has no evaluator or hidden-data input surface.
        """
        ids={str(x) for x in (observation_ids or []) if str(x)}
        vals=[str(x).casefold() for x in (forbidden_values or []) if str(x)]
        errors=[];removed=set()
        def targets(rows):
            out=[]
            for r in rows:
                md=r.get('metadata') if isinstance(r.get('metadata'),dict) else {};oid=str(md.get('hf13_observation_id') or '')
                is_obs=bool(md.get('hf13_observation')) and not bool(md.get('hf13_managed'))
                if is_obs and (oid in ids or any(v in record_text(r).casefold() for v in vals)):out.append(r)
            return out
        for attempt in range(1,int(max_retries)+2):
            ts=targets(self._all(user_id))
            if not ts:
                status='recovered' if errors else 'clean'
                return {'status':status,'verified':True,'attempts':attempt-1,'removed_count':len(removed),'removed_physical_ids':sorted(removed),'error_lines':errors,'provider_internal_error_count':len(errors),'infer':False,'hidden_gold_used':False}
            for r in ts:
                rid=_rid(r)
                if not rid:errors.append('observation purge target missing physical id');continue
                try:self.backend.delete(rid);removed.add(rid)
                except Exception as e:errors.append(f'delete {rid}: {type(e).__name__}: {e}')
        leaked=[_rid(r) for r in targets(self._all(user_id))]
        raise RuntimeError(f'CSS observation purge failed physical verification after bounded retry: leaked={leaked}; errors={errors[-5:]}')
    def reconcile_managed(self,facts,*,user_id,max_repair_attempts=2):
        last=None;all_errors=[]
        for attempt in range(1,int(max_repair_attempts)+2):
            existing=self._all(user_id);desired={str(f['fact_id']):f for f in facts}
            present={}
            for r in existing:
                md=r.get('metadata') if isinstance(r.get('metadata'),dict) else {}
                if md.get('hf13_managed'):present[str(md.get('hf13_fact_id') or '')]=r
            op_errors=[]
            for fid,r in list(present.items()):
                f=desired.get(fid);md=r.get('metadata') or {}
                if f is None or md.get('hf13_state_sha256')!=_hash_fact(f) or parse_fact_text(record_text(r)) is None:
                    rid=_rid(r)
                    if rid:
                        try:self.backend.delete(rid)
                        except Exception as e:op_errors.append(f'delete {rid}: {type(e).__name__}: {e}')
            current_ids={str((r.get('metadata') or {}).get('hf13_fact_id') or '') for r in self._all(user_id) if (r.get('metadata') or {}).get('hf13_managed')}
            for fid,f in desired.items():
                if fid in current_ids:continue
                try:self.mem.add(fact_text(f),user_id=user_id,metadata={'hf13_managed':True,'hf13_fact_id':fid,'hf13_state_sha256':_hash_fact(f),'hf13_infer':False},infer=False)
                except Exception as e:op_errors.append(f'add {fid}: {type(e).__name__}: {e}')
            audit=self.project_managed(user_id=user_id);all_errors.extend(op_errors)
            if not audit['mismatches'] and state_signature(audit['facts'])==state_signature(facts):
                status='recovered' if all_errors else 'clean'
                return {'verified':True,'attempts':attempt,'managed_count':len(audit['facts']),'coverage':audit['coverage'],'mismatches':[],'status':status,'error_lines':all_errors,'provider_internal_error_count':len(all_errors),'infer':False}
            last={'mismatches':audit['mismatches'],'expected_count':len(facts),'actual_count':len(audit['facts']),'operation_errors':op_errors}
        raise RuntimeError(f'CSS physical verification failed after bounded repair: {last}; errors={all_errors[-5:]}')
    def diagnostic_inject_managed_fault(self,*,user_id,fact_id,kind,case_id):
        """Evaluator-only physical fault injection against the isolated Mem0 backend."""
        rows=self._all(user_id);target=None
        for r in rows:
            md=r.get('metadata') if isinstance(r.get('metadata'),dict) else {}
            if md.get('hf13_managed') and str(md.get('hf13_fact_id'))==str(fact_id):target=r;break
        if target is None:raise RuntimeError(f'fault target not physically present: {fact_id}')
        obj=parse_fact_text(record_text(target));rid=_rid(target)
        if obj is None or not rid:raise RuntimeError(f'fault target is not physically mutable: {fact_id}')
        if kind=='delete_current':
            self.backend.delete(rid)
        else:
            z=copy.deepcopy(obj)
            if kind=='duplicate_current':z['fact_id']=str(obj.get('fact_id'))+f':FAULT_DUP:{case_id}'
            elif kind=='contradict_current':
                z['fact_id']=str(obj.get('fact_id'))+f':FAULT_CONTRADICT:{case_id}';z['value']='CONTRADICT-'+str(obj.get('value'))
            elif kind=='corrupt_record':
                self.backend.delete(rid);z['permission']='restricted' if z.get('permission')!='restricted' else 'allowed';z['authority']='assistant';z['control_status']='noncontrolling'
            else:raise ValueError(kind)
            fid=str(z.get('fact_id'));self.mem.add(fact_text(z),user_id=user_id,metadata={'hf13_managed':True,'hf13_fact_id':fid,'hf13_state_sha256':_hash_fact(z),'hf13_fault_injected':True,'hf13_infer':False},infer=False)
        audit=self.project_managed(user_id=user_id)
        if audit['mismatches']:raise RuntimeError(f'physical fault injection produced invalid managed records: {audit["mismatches"]}')
        return {'status':'clean','kind':kind,'case_id':case_id,'target_fact_id':fact_id,'physical_count':len(audit['facts']),'mismatches':[],'infer':False}
    def project_managed(self,*,user_id):return _project_records(self._all(user_id))
    def search(self,query,*,user_id,limit=10,max_retries=2):
        errors=[]
        for attempt in range(1,int(max_retries)+2):
            try:
                out=self.backend.search(query,user_id,limit=limit);status='recovered' if errors else 'clean';self.last_search_diagnostics={'status':status,'attempts':attempt,'retry_count':attempt-1,'error_lines':errors,'read_only':True};return out
            except Exception as e:
                errors.append(f'{type(e).__name__}: {e}')
                if attempt>int(max_retries):break
        self.last_search_diagnostics={'status':'failed','attempts':int(max_retries)+1,'retry_count':int(max_retries),'error_lines':errors,'read_only':True};raise RuntimeError(f'Mem0 read-only search failed after bounded retry: {errors[-3:]}')
    def get_all(self,*,user_id):return self._all(user_id)
    def close(self):
        try:self.backend.close()
        finally:shutil.rmtree(self.scratch,ignore_errors=True)

def make_store(kind,cfg=None,*,result_root='results/hf13',namespace='hf13'):
    if kind=='reference':return ReferenceStore()
    if kind=='mem0':return Mem0Store(cfg or {},result_root=result_root,namespace=namespace)
    raise ValueError(kind)
