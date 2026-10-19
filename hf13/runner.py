from __future__ import annotations
import copy,json
from pathlib import Path
from .backend import make_store,record_text
from .css import CSSKernel
from .dataset import ConditionView,HarnessSchedule
from .io import append_jsonl
from .llm import answer_natural_turn,answer_with_support,get_last_diagnostics
from .qasc import authorize
from .physical_readback import project_physical_records,authorize_physical_readback
from .artifact_registry import RuntimeArtifactRegistry,build_shared_registry
from .status import from_diagnostics

def _rid(r):return str(r.get('id') or r.get('memory_id') or r.get('raw_id') or '')
def _record_view(r):return {'physical_id':_rid(r),'text':record_text(r),'metadata':copy.deepcopy(r.get('metadata') or {})}
def _snapshot(records):return sorted((_rid(r),record_text(r),json.dumps(r.get('metadata') or {},sort_keys=True,ensure_ascii=False,default=str)) for r in records)
def _render_fact(f):
    val=f.get('value')
    if isinstance(val,dict) and val.get('text'):val=val['text']
    return f"{f.get('entity')}.{f.get('predicate')} = {json.dumps(val,ensure_ascii=False,default=str)} [lifecycle={f.get('lifecycle')}; authority={f.get('authority')}; permission={f.get('permission')}]"
def _reference_answer(support):return 'INSUFFICIENT_MEMORY' if not support else ' | '.join(support)
def _status_from(diags):return from_diagnostics(diags)
def _registry(x,data_root,cfg,answer_mode):
    if isinstance(x,RuntimeArtifactRegistry):return x
    if isinstance(x,(str,Path)):return RuntimeArtifactRegistry.load(x)
    if isinstance(x,dict):return RuntimeArtifactRegistry(x)
    return build_shared_registry(data_root,cfg=cfg,mode='reference' if answer_mode=='reference' else 'llm')
def _resolve_event(ev,reg):return reg.resolve_object(copy.deepcopy(ev))
def _required_source(reg,tid):return ((reg.obj.get('responses') or {}).get(tid) or {}).get('answer')
def _proj_managed(store,uid):return store.project_managed(user_id=uid)['facts']
def _budget_support(support,cfg):
    n=(((cfg or {}).get('evaluation') or {}).get('model_visible_support_limit'))
    return list(support) if n in (None,0,'') else list(support)[:int(n)]

def _value_leaves(v):
    if isinstance(v,dict):
        out=[]
        for x in v.values():out.extend(_value_leaves(x))
        return out
    if isinstance(v,list):
        out=[]
        for x in v:out.extend(_value_leaves(x))
        return out
    s=str(v)
    return [] if s.casefold() in {'true','false','none','null'} or len(s.strip())<4 else [s]

def _purge_css_observations(store,uid,pre_state,transition):
    """Apply CSS-derived PURGE to CSS-owned inert observations as well.

    The target set is derived entirely from the pre-transition CSS state and
    internal PURGE operations. Hidden evaluator gold is never an input.
    """
    purge_ids={str(x.get('fact_id')) for x in ((transition or {}).get('internal_transitions') or []) if x.get('op')=='PURGE'}
    if not purge_ids:return None
    doomed=[f for f in pre_state if str(f.get('fact_id')) in purge_ids]
    obs_ids={f"user:{f.get('source_turn_id')}" for f in doomed if f.get('source_turn_id')}
    values=[]
    for f in doomed:
        if f.get('predicate')=='values':values.extend(_value_leaves(f.get('value')))
    return store.purge_observations(user_id=uid,observation_ids=sorted(obs_ids),forbidden_values=sorted(set(values)))

def run(condition,*,data_root='data/longitudinal',backend='reference',cfg=None,result_path=None,answer_mode='reference',run_id='hf13',artifact_registry=None,artifact_cache=None,closed_loop=False,progress_path=None):
    view=ConditionView(data_root,condition);schedule=HarnessSchedule(data_root);reg=_registry(artifact_registry if artifact_registry is not None else artifact_cache,data_root,cfg,answer_mode)
    store=make_store(backend,cfg,result_root=Path(result_path).parent if result_path else 'results/hf13',namespace=f'{run_id}:{condition}:'+('closed' if closed_loop else 'primary'))
    uid=f'{run_id}:{condition}:user';kernel=CSSKernel() if condition in {'css','place'} else None;rows=[];ctx=[];conv=None;all_diags=[];out=Path(result_path) if result_path else None
    if out:out.parent.mkdir(parents=True,exist_ok=True);out.write_text('',encoding='utf-8')
    try:
      for idx,t in enumerate(view.turns,1):
        if conv!=t['conversation_id']:ctx=[];conv=t['conversation_id']
        if kernel:
            ex=kernel.advance_time(t['timestamp'])
            if ex:store.reconcile_managed(kernel.snapshot(),user_id=uid)
        isq=schedule.is_query_checkpoint(t['turn_id']);qsem=view.query_for(t['turn_id']);qrec=None;assistant=None;turn_diags=[]
        if isq:
            backend_before=store.get_all(user_id=uid);snap_before=_snapshot(backend_before);retrieved=store.search(t['utterance'],user_id=uid,limit=int(((cfg or {}).get('memory') or {}).get('retrieval_limit',10)));turn_diags.append({'component':'retrieval_search',**copy.deepcopy(getattr(store,'last_search_diagnostics',{}) or {})});retrieval_candidates=[_record_view(r) for r in retrieved]
            physical_readback=project_physical_records(backend_before)
            if condition=='qasc':
                physical=[];authorized=authorize_physical_readback(physical_readback,qsem,raw_query=t['utterance']) if qsem else [];semantic_support=[str(f.get('text') or '') for f in authorized];model_support=_budget_support(semantic_support,cfg)
            elif condition=='place':
                physical=_proj_managed(store,uid);ids={_rid(r) for r in backend_before};authorized=authorize(physical,qsem,as_of=qsem.get('as_of') or t['timestamp'],physical_record_ids=ids) if qsem else [];semantic_support=[_render_fact(f) for f in authorized];model_support=_budget_support(semantic_support,cfg)
            elif condition=='css':physical=_proj_managed(store,uid);authorized=[];model_support=_budget_support([x['text'] for x in retrieval_candidates],cfg)
            else:physical=[];authorized=[];model_support=_budget_support([x['text'] for x in retrieval_candidates],cfg)
            # QASC/PLACE read planes are provably non-mutating.
            if condition in {'qasc','place'}:
                snap_after=_snapshot(store.get_all(user_id=uid))
                if snap_after!=snap_before:raise RuntimeError('QASC read plane mutated physical storage')
            if answer_mode=='reference':assistant=_reference_answer(model_support)
            else:
                assistant=answer_with_support(t['utterance'],model_support,cfg or {},context=ctx);turn_diags.append({'component':'answer_llm',**get_last_diagnostics()})
            qrec={'query_semantics_used':copy.deepcopy(qsem) if condition in {'qasc','place'} else None,'physical_state':physical,'physical_semantic_readback':physical_readback,'physical_backend_records':[_record_view(r) for r in backend_before],'retrieval_candidates':retrieval_candidates,'authorized_support':authorized,'query_authorized_support':semantic_support if condition in {'qasc','place'} else [],'model_visible_support':model_support,'response':assistant,'qasc_storage_unchanged':True if condition in {'qasc','place'} else None}
        binding=schedule.bindings.get(t['turn_id']);resolved=reg.resolve_id(binding['binding_id']) if binding else None;ev=view.event_for(t['turn_id']) if condition in {'css','place'} else None;transition=None;cert=None
        if condition in {'css','place'}:
            # CSS owns the state plane. Every raw utterance is retained as an inert,
            # infer=False observation for native retrieval/context, while only the
            # trusted user-level event may drive CSS governance transitions.
            od=store.add_observation(t['utterance'],user_id=uid,observation_id=f'user:{t["turn_id"]}',role='user')
            turn_diags.append({'component':'css_raw_observation','input_kind':'raw_sentence_observation','trusted_event_passed_to_storage':False,'metadata_governance_keys':[],'infer':False,**od.get('diagnostics',{})})
            if ev is not None:
                ev=_resolve_event(ev,reg);pre_css=kernel.snapshot();transition=kernel.apply(ev,as_of=t['timestamp']);pd=_purge_css_observations(store,uid,pre_css,transition)
                if pd is not None:turn_diags.append({'component':'css_observation_purge','shared_write_plane':'css','hidden_gold_used':False,**pd})
                cert=store.reconcile_managed(kernel.snapshot(),user_id=uid)
                turn_diags.append({'component':'css_physical_certification',**{k:v for k,v in cert.items() if k not in {'coverage','mismatches'}},'verified':bool(cert.get('verified'))})
        else:
            # Native/QASC are the genuine Native Mem0 write plane: raw sentence,
            # empty benchmark metadata, infer=True inside Mem0.
            d=store.native_add(t['utterance'],user_id=uid,metadata={},role='user',max_retries=1);turn_diags.append({'component':'memory_write','input_kind':'raw_sentence_only','trusted_event_passed':False,'metadata_keys':[],**d.get('diagnostics',{})})
        if assistant is None:
            frozen=_required_source(reg,t['turn_id'])
            if frozen:assistant=frozen
            elif answer_mode=='reference':assistant='Acknowledged.'
            else:
                ret=store.search(t['utterance'],user_id=uid,limit=int(((cfg or {}).get('memory') or {}).get('retrieval_limit',10)));turn_diags.append({'component':'context_search',**copy.deepcopy(getattr(store,'last_search_diagnostics',{}) or {})});assistant=answer_natural_turn(t['utterance'],cfg or {},context=ctx,memory_support=[record_text(r) for r in ret]);turn_diags.append({'component':'answer_llm',**get_last_diagnostics()})
        if closed_loop:
            # Assistant outputs are persisted only as explicitly non-authoritative
            # observations. They never get a second chance to infer user state.
            d=store.add_observation(assistant,user_id=uid,observation_id=f'assistant:{t["turn_id"]}',role='assistant');turn_diags.append({'component':'closed_loop_assistant_observation','infer':False,'authoritative':False,**d.get('diagnostics',{})})
        roles=schedule.roles.get(t['turn_id'],[]);need_state='state_checkpoint' in roles or isq
        state_after=_proj_managed(store,uid) if need_state and condition in {'css','place'} else []
        backend_after=store.get_all(user_id=uid) if need_state else []
        row={'turn_id':t['turn_id'],'conversation_id':t['conversation_id'],'raw_utterance':t['utterance'],'condition':condition,'closed_loop':bool(closed_loop),'trusted_event_used':copy.deepcopy(ev) if ev else None,'transition':transition,'query':qrec,'assistant_response':assistant,'physical_state':state_after,'physical_facts':state_after,'physical_semantic_readback':project_physical_records(backend_after) if need_state else [],'physical_backend_records':[_record_view(r) for r in backend_after],'physical_certification':cert,'record_count':len(store.get_all(user_id=uid)),'runtime_binding_resolved':copy.deepcopy(resolved),'runtime_registry_schema':reg.obj.get('schema_version'),'provider_diagnostics':turn_diags,'turn_status':_status_from(turn_diags)}
        rows.append(row);all_diags.extend(turn_diags)
        if out:append_jsonl(out,row)
        ctx.extend([{'role':'user','content':t['utterance']},{'role':'assistant','content':assistant}])
        if progress_path:
            Path(progress_path).parent.mkdir(parents=True,exist_ok=True);Path(progress_path).write_text(json.dumps({'stage':'turns','done':idx,'total':len(view.turns),'current_turn_id':t['turn_id']})+'\n')
      final=_status_from(all_diags)
      if progress_path:Path(progress_path).write_text(json.dumps({'stage':'complete','done':len(view.turns),'total':len(view.turns),'status':final})+'\n')
      return rows
    finally:store.close()
