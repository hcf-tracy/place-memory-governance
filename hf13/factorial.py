"""Structurally factorized Primary / Closed-Loop execution.

Two write planes are executed exactly once per turn:
  Native producer -> shared by Native and QASC read branches.
  CSS producer    -> shared by CSS and PLACE read branches.

Read-branch assistant outputs are kept in branch-local, non-authoritative
observation/context ledgers. They may influence later answer generation through
branch-local dialogue context, but they are never written into either state
producer and therefore cannot feed back into state formation.
"""
from __future__ import annotations
import copy, hashlib, json
from pathlib import Path

from .artifact_registry import RuntimeArtifactRegistry, build_shared_registry
from .backend import make_store, record_text
from .css import CSSKernel
from .dataset import ConditionView, HarnessSchedule
from .io import append_jsonl
from .llm import answer_natural_turn, answer_with_support, get_last_diagnostics
from .physical_readback import project_physical_records, authorize_physical_readback
from .qasc import authorize
from .runner import _rid, _record_view, _snapshot, _render_fact, _reference_answer, _budget_support, _resolve_event, _required_source, _purge_css_observations
from .status import from_diagnostics, merge_statuses

CONDS=('native','qasc','css','place')
PLANE={'native':'native','qasc':'native','css':'css','place':'css'}
PAIR={'native':'qasc','qasc':'native','css':'place','place':'css'}

def _registry(x,data_root,cfg,answer_mode):
    if isinstance(x,RuntimeArtifactRegistry): return x
    if isinstance(x,(str,Path)): return RuntimeArtifactRegistry.load(x)
    if isinstance(x,dict): return RuntimeArtifactRegistry(x)
    return build_shared_registry(data_root,cfg=cfg,mode='reference' if answer_mode=='reference' else 'llm')

def _ledger_digest(entries):
    payload=json.dumps(entries,ensure_ascii=False,sort_keys=True,separators=(',',':'),default=str)
    return hashlib.sha256(payload.encode()).hexdigest()

def _write_jsonl_reset(path):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);p.write_text('',encoding='utf-8');return p

def _answer_query(condition,t,qsem,backend_before,retrieval_candidates,managed,cfg,answer_mode,ctx,store,uid):
    diags=[];physical_readback=project_physical_records(backend_before)
    if condition=='qasc':
        physical=[];authorized=authorize_physical_readback(physical_readback,qsem,raw_query=t['utterance']) if qsem else []
        semantic_support=[str(f.get('text') or '') for f in authorized];model_support=_budget_support(semantic_support,cfg)
    elif condition=='place':
        physical=managed;ids={_rid(r) for r in backend_before};authorized=authorize(physical,qsem,as_of=qsem.get('as_of') or t['timestamp'],physical_record_ids=ids) if qsem else []
        semantic_support=[_render_fact(f) for f in authorized];model_support=_budget_support(semantic_support,cfg)
    elif condition=='css':
        physical=managed;authorized=[];semantic_support=[];model_support=_budget_support([x['text'] for x in retrieval_candidates],cfg)
    else:
        physical=[];authorized=[];semantic_support=[];model_support=_budget_support([x['text'] for x in retrieval_candidates],cfg)
    if condition in {'qasc','place'}:
        before=_snapshot(backend_before);after=_snapshot(store.get_all(user_id=uid))
        if before!=after: raise RuntimeError('QASC read plane mutated shared physical storage')
    if answer_mode=='reference': assistant=_reference_answer(model_support)
    else:
        assistant=answer_with_support(t['utterance'],model_support,cfg or {},context=ctx)
        diags.append({'component':'answer_llm',**get_last_diagnostics()})
    return assistant,{
        'query_semantics_used':copy.deepcopy(qsem) if condition in {'qasc','place'} else None,
        'physical_state':copy.deepcopy(physical),
        'physical_semantic_readback':copy.deepcopy(physical_readback),
        'physical_backend_records':[_record_view(r) for r in backend_before],
        'retrieval_candidates':copy.deepcopy(retrieval_candidates),
        'authorized_support':copy.deepcopy(authorized),
        'query_authorized_support':copy.deepcopy(semantic_support) if condition in {'qasc','place'} else [],
        'model_visible_support':copy.deepcopy(model_support),
        'response':assistant,
        'qasc_storage_unchanged':True if condition in {'qasc','place'} else None,
    },diags

def run_factorial(*,data_root='data/longitudinal',backend='reference',cfg=None,answer_mode='reference',run_id='hf13-factorial',artifact_registry=None,closed_loop=False,output_root=None,progress_path=None,answer_override=None):
    """Execute the 2x2 design with one write per write-plane and two read branches.

    answer_override(condition, turn, default_answer) is test-only and may deliberately
    force divergent branch answers without touching the state producer.
    """
    root=Path(data_root);schedule=HarnessSchedule(root);turns=ConditionView(root,'native').turns
    event_view=ConditionView(root,'css');query_view=ConditionView(root,'qasc')
    reg=_registry(artifact_registry,root,cfg,answer_mode)
    surface='closed_loop' if closed_loop else 'primary'
    result_root=Path(output_root) if output_root else Path('results/hf13')
    stores={
        'native':make_store(backend,cfg or {},result_root=result_root,namespace=f'{run_id}:native_write:{surface}'),
        'css':make_store(backend,cfg or {},result_root=result_root,namespace=f'{run_id}:css_write:{surface}'),
    }
    uids={'native':f'{run_id}:native_write:user','css':f'{run_id}:css_write:user'}
    kernel=CSSKernel();rows={c:[] for c in CONDS};ctx={c:[] for c in CONDS};ledgers={c:[] for c in CONDS};conv=None
    trace_paths={};ledger_paths={}
    if output_root:
        for c in CONDS:
            d=Path(output_root)/c;d.mkdir(parents=True,exist_ok=True)
            trace_paths[c]=_write_jsonl_reset(d/'trace.jsonl')
            if closed_loop: ledger_paths[c]=_write_jsonl_reset(d/'observation_ledger.jsonl')
    try:
      for idx,t in enumerate(turns,1):
        if conv!=t['conversation_id']:
            for c in CONDS: ctx[c]=[]
            conv=t['conversation_id']
        # CSS time advancement is one shared write-plane operation.
        css_pre_diags=[];expired=kernel.advance_time(t['timestamp'])
        if expired:
            cert=stores['css'].reconcile_managed(kernel.snapshot(),user_id=uids['css'])
            css_pre_diags.append({'component':'css_shared_time_advance_certification','shared_write_plane':'css','infer':False,**{k:v for k,v in cert.items() if k not in {'coverage','mismatches'}}})

        isq=schedule.is_query_checkpoint(t['turn_id']);qsem=query_view.query_for(t['turn_id']) if isq else None
        branch_diags={c:copy.deepcopy(css_pre_diags) if c in {'css','place'} else [] for c in CONDS};qrecs={c:None for c in CONDS};assistants={}

        # Read each shared state producer once before the user write. Both branches
        # receive the exact same producer snapshot/candidate pool.
        if isq:
            for plane,branches in (('native',('native','qasc')),('css',('css','place'))):
                store=stores[plane];uid=uids[plane];backend_before=store.get_all(user_id=uid)
                retrieved=store.search(t['utterance'],user_id=uid,limit=int(((cfg or {}).get('memory') or {}).get('retrieval_limit',10)))
                rdiag={'component':'shared_retrieval_search','shared_write_plane':plane,**copy.deepcopy(getattr(store,'last_search_diagnostics',{}) or {})}
                candidates=[_record_view(r) for r in retrieved]
                managed=store.project_managed(user_id=uid)['facts'] if plane=='css' else []
                for c in branches:
                    branch_diags[c].append(copy.deepcopy(rdiag))
                    a,q,d=_answer_query(c,t,qsem,backend_before,candidates,managed,cfg,answer_mode,ctx[c],store,uid)
                    if answer_override is not None: a=answer_override(c,t,a);q['response']=a
                    assistants[c]=a;qrecs[c]=q;branch_diags[c].extend(d)

        # Exactly one Native write. Its evaluated-system status is inherited by
        # both Native and QASC branches rather than independently resampled.
        nd=stores['native'].native_add(t['utterance'],user_id=uids['native'],metadata={},role='user',max_retries=1)
        ndg={'component':'memory_write','input_kind':'raw_sentence_only','trusted_event_passed':False,'metadata_keys':[],'shared_write_plane':'native','shared_write_operation_id':f'{surface}:{t["turn_id"]}:native',**nd.get('diagnostics',{})}
        for c in ('native','qasc'): branch_diags[c].append(copy.deepcopy(ndg))

        # Exactly one CSS write. Raw observation + canonical trusted event are
        # realized once and inherited by CSS and PLACE.
        od=stores['css'].add_observation(t['utterance'],user_id=uids['css'],observation_id=f'user:{t["turn_id"]}',role='user')
        odg={'component':'css_raw_observation','input_kind':'raw_sentence_observation','trusted_event_passed_to_storage':False,'metadata_governance_keys':[],'infer':False,'shared_write_plane':'css','shared_write_operation_id':f'{surface}:{t["turn_id"]}:css',**od.get('diagnostics',{})}
        for c in ('css','place'): branch_diags[c].append(copy.deepcopy(odg))
        ev=event_view.event_for(t['turn_id']);transition=None;cert=None
        if ev is not None:
            ev=_resolve_event(ev,reg);pre_css=kernel.snapshot();transition=kernel.apply(ev,as_of=t['timestamp']);pd=_purge_css_observations(stores['css'],uids['css'],pre_css,transition)
            if pd is not None:
                pdg={'component':'css_observation_purge','shared_write_plane':'css','hidden_gold_used':False,**pd}
                for c in ('css','place'):branch_diags[c].append(copy.deepcopy(pdg))
            cert=stores['css'].reconcile_managed(kernel.snapshot(),user_id=uids['css'])
            cd={'component':'css_physical_certification','shared_write_plane':'css','infer':False,**{k:v for k,v in cert.items() if k not in {'coverage','mismatches'}},'verified':bool(cert.get('verified'))}
            for c in ('css','place'): branch_diags[c].append(copy.deepcopy(cd))

        # Natural-turn answers branch after the shared write. The memory support is
        # sampled once per producer; only branch-local dialogue context can differ.
        if not isq:
            frozen=_required_source(reg,t['turn_id'])
            if frozen:
                for c in CONDS: assistants[c]=frozen if answer_override is None else answer_override(c,t,frozen)
            else:
                for plane,branches in (('native',('native','qasc')),('css',('css','place'))):
                    store=stores[plane];uid=uids[plane]
                    if answer_mode=='reference':
                        for c in branches: assistants[c]='Acknowledged.' if answer_override is None else answer_override(c,t,'Acknowledged.')
                    else:
                        ret=store.search(t['utterance'],user_id=uid,limit=int(((cfg or {}).get('memory') or {}).get('retrieval_limit',10)))
                        sdiag={'component':'shared_context_search','shared_write_plane':plane,**copy.deepcopy(getattr(store,'last_search_diagnostics',{}) or {})}
                        support=[record_text(r) for r in ret]
                        for c in branches:
                            branch_diags[c].append(copy.deepcopy(sdiag));a=answer_natural_turn(t['utterance'],cfg or {},context=ctx[c],memory_support=support)
                            branch_diags[c].append({'component':'answer_llm',**get_last_diagnostics()});assistants[c]=a if answer_override is None else answer_override(c,t,a)

        binding=schedule.bindings.get(t['turn_id']);resolved=reg.resolve_id(binding['binding_id']) if binding else None
        native_after=stores['native'].get_all(user_id=uids['native']);css_after=stores['css'].get_all(user_id=uids['css']);css_state=stores['css'].project_managed(user_id=uids['css'])['facts']
        roles=schedule.roles.get(t['turn_id'],[]);need_state='state_checkpoint' in roles or isq
        for c in CONDS:
            plane=PLANE[c];backend_after=native_after if plane=='native' else css_after;state_after=css_state if need_state and plane=='css' else []
            ledger_entry=None
            if closed_loop:
                ledger_entry={'turn_id':t['turn_id'],'conversation_id':t['conversation_id'],'role':'assistant','authoritative':False,'infer':False,'text':assistants[c]}
                ledgers[c].append(ledger_entry)
                branch_diags[c].append({'component':'closed_loop_observation_ledger_append','status':'clean','authoritative':False,'infer':False,'state_producer_mutated':False})
                if c in ledger_paths: append_jsonl(ledger_paths[c],ledger_entry)
            row={
                'turn_id':t['turn_id'],'conversation_id':t['conversation_id'],'raw_utterance':t['utterance'],'condition':c,'closed_loop':bool(closed_loop),
                'trusted_event_used':copy.deepcopy(ev) if c in {'css','place'} and ev else None,'transition':copy.deepcopy(transition) if c in {'css','place'} else None,
                'query':copy.deepcopy(qrecs[c]),'assistant_response':assistants[c],
                'physical_state':copy.deepcopy(state_after),'physical_facts':copy.deepcopy(state_after),
                'physical_semantic_readback':project_physical_records(backend_after) if need_state else [],
                'physical_backend_records':[_record_view(r) for r in backend_after] if need_state else [],
                'physical_certification':copy.deepcopy(cert) if c in {'css','place'} else None,'record_count':len(backend_after),
                'runtime_binding_resolved':copy.deepcopy(resolved),'runtime_registry_schema':reg.obj.get('schema_version'),
                'provider_diagnostics':branch_diags[c],'turn_status':from_diagnostics(branch_diags[c]),
                'write_plane_execution':'shared_write_once_read_branch','shared_write_plane':plane,'shared_write_with':PAIR[c],
                'shared_write_operation_id':f'{surface}:{t["turn_id"]}:{plane}',
                'observation_ledger_entry_id':f'{c}:{t["turn_id"]}' if closed_loop else None,
                'observation_ledger_count':len(ledgers[c]) if closed_loop else 0,
                'observation_ledger_digest':_ledger_digest(ledgers[c]) if closed_loop else None,
                'observation_ledger_state_authoritative':False if closed_loop else None,
            }
            rows[c].append(row)
            if c in trace_paths: append_jsonl(trace_paths[c],row)
            ctx[c].extend([{'role':'user','content':t['utterance']},{'role':'assistant','content':assistants[c]}])
        if progress_path:
            Path(progress_path).parent.mkdir(parents=True,exist_ok=True)
            Path(progress_path).write_text(json.dumps({'stage':'turns','done':idx,'total':len(turns),'current_turn_id':t['turn_id'],'execution':'write_once_read_branch'})+'\n')
      if progress_path:
          Path(progress_path).write_text(json.dumps({'stage':'complete','done':len(turns),'total':len(turns),'execution':'write_once_read_branch','status':merge_statuses([r['turn_status'] for c in CONDS for r in rows[c]])})+'\n')
      return rows
    finally:
      for s in stores.values(): s.close()
