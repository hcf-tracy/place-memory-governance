from __future__ import annotations
import copy,json
from collections import defaultdict
from pathlib import Path
from .dataset import EvaluatorData
from .io import read_jsonl,append_jsonl
from .css import CSSKernel,fact_signature,state_signature
from .qasc import authorize
from .llm import answer_with_support,get_last_diagnostics
from .evaluator import f1_sets,runtime_gold,answer_score,_forbidden_mentions,_forbidden_answer_mentions
from .artifact_registry import RuntimeArtifactRegistry,build_shared_registry
from .backend import make_store
from .evaluator_projection import EvaluatorPhysicalProjector
from .status import from_diagnostics


def _sf1(pred,gold):
    return f1_sets([fact_signature(x) for x in pred],[fact_signature(x) for x in gold])

def _render(f):
    return f"{f.get('entity')}.{f.get('predicate')}={json.dumps(f.get('value'),ensure_ascii=False,default=str)}"

def _registry(x,data_root,cfg,answer_mode):
    if isinstance(x,RuntimeArtifactRegistry):return x
    if isinstance(x,(str,Path)):return RuntimeArtifactRegistry.load(x)
    if isinstance(x,dict):return RuntimeArtifactRegistry(x)
    return build_shared_registry(data_root,cfg=cfg,mode='reference' if answer_mode=='reference' else 'llm')

def _css_intended_prequery(data_root,registry,target_turn_ids):
    """Replay trusted user-level events only; no evaluator gold is read here."""
    root=Path(data_root);events={x['turn_id']:x for x in read_jsonl(root/'trusted_events.jsonl')};targets=set(target_turn_ids);kernel=CSSKernel();out={}
    for t in read_jsonl(root/'trajectory_raw.jsonl'):
        kernel.advance_time(t['timestamp'])
        if t['turn_id'] in targets:out[t['turn_id']]=kernel.snapshot()
        ev=events.get(t['turn_id'])
        if ev is not None:kernel.apply(registry.resolve_object(copy.deepcopy(ev)),as_of=t['timestamp'])
    missing=targets-set(out)
    if missing:raise RuntimeError(f'missing CSS intended snapshots for fault cases: {sorted(missing)}')
    return out

def _select_target_fact(facts,qsem,*,required=True):
    ids={pid for f in facts for pid in (f.get('physical_ids') or [])}
    try:
        auth=authorize(facts,qsem,physical_record_ids=ids) if qsem else []
    except Exception:
        auth=[]
    candidates=[f for f in auth if f.get('fact_id')]
    if not candidates and qsem:
        target=qsem.get('target') or {};entity=target.get('entity');predicate=target.get('predicate')
        candidates=[f for f in facts if (not entity or f.get('entity')==entity) and (not predicate or f.get('predicate')==predicate)]
    if not candidates:candidates=[f for f in facts if f.get('lifecycle')=='active' and f.get('control_status')=='controlling']
    if not candidates:candidates=list(facts)
    if not candidates:
        if required:raise RuntimeError('fault case has no physical fact to perturb')
        return None
    candidates.sort(key=lambda f:(0 if f.get('lifecycle')=='active' else 1,0 if f.get('control_status')=='controlling' else 1,-int(f.get('sequence') or 0)))
    return candidates[0]

def _slot_facts(facts,target_fact):
    return [f for f in facts if f.get('entity')==target_fact.get('entity') and f.get('predicate')==target_fact.get('predicate')]

def _repair_from_intended(store,user_id,intended):
    """Repair target is CSS's own intended state. Evaluator gold is intentionally absent."""
    cert=store.reconcile_managed(copy.deepcopy(intended),user_id=user_id)
    post=store.project_managed(user_id=user_id)
    verified=bool(cert.get('verified')) and not post.get('mismatches') and state_signature(post['facts'])==state_signature(intended)
    return post['facts'],{**cert,'verified':verified,'repair_target_source':'css_intended_state'}

def run_fault(primary_rows,condition,*,data_root='data/longitudinal',runtime_data_root=None,answer_mode='reference',cfg=None,progress_path=None,artifact_registry=None,backend='reference',result_root=None,missing_target_policy='error'):
    """Physical fault diagnostic with a strict evaluator boundary.

    Phase 1 performs baseline realization, physical fault injection, CSS repair,
    semantic readback, and answer generation without loading hidden gold.
    Phase 2 opens evaluator-only gold and scores the already-fixed outputs.
    """
    if missing_target_policy not in {'error','record_ineligible'}:raise ValueError('missing_target_policy must be error or record_ineligible')
    if isinstance(primary_rows,(str,Path)):primary_rows=read_jsonl(primary_rows)
    runtime_root=Path(runtime_data_root or data_root)
    ev=EvaluatorData(data_root);by={r['turn_id']:r for r in primary_rows};pp=Path(progress_path) if progress_path else None
    if pp:pp.parent.mkdir(parents=True,exist_ok=True);pp.write_text('',encoding='utf-8')
    runtime_qsem={x['turn_id']:x for x in read_jsonl(runtime_root/'trusted_queries.jsonl')}
    fault_selector_qsem={x['turn_id']:x for x in read_jsonl(Path(data_root)/'trusted_queries.jsonl')}
    registry=_registry(artifact_registry,runtime_root,cfg,answer_mode);eval_projector=EvaluatorPhysicalProjector(data_root,registry) if condition in {'native','qasc'} else None;tids={x['base_query_turn_id'] for x in ev.fault_cases}
    intended_by=_css_intended_prequery(runtime_root,registry,tids) if condition in {'css','place'} else {}
    rr=Path(result_root) if result_root else ((pp.parent if pp else Path('results/hf13_fault'))/f'physical_{condition}')
    store=make_store(backend,cfg or {},result_root=rr,namespace=f'fault:{condition}');physical_rows=[]
    try:
        for i,case in enumerate(ev.fault_cases,1):
            tid=case['base_query_turn_id'];r=by[tid];sem=runtime_qsem.get(tid);selector_sem=fault_selector_qsem[tid];rq=(r.get('query') or {});base=(eval_projector.project(rq.get('physical_backend_records') or [],through_turn_id=tid,include_through=False)['facts'] if eval_projector else copy.deepcopy(rq.get('physical_state') or []))
            intended=copy.deepcopy(intended_by[tid]) if condition in {'css','place'} else None
            if intended is not None and state_signature(base)!=state_signature(intended):
                raise RuntimeError(f'{case["case_id"]}: Primary CSS/PLACE physical state does not match independently replayed CSS intended state')
            baseline=intended if intended is not None else base;uid=f'fault:{condition}:{case["case_id"]}';diags=[]
            base_cert=store.reconcile_managed(copy.deepcopy(baseline),user_id=uid);diags.append({'component':'fault_baseline_realization',**base_cert})
            realized=store.project_managed(user_id=uid);target=_select_target_fact(realized['facts'],selector_sem,required=(missing_target_policy=='error'))
            if target is None:
                # Parser diagnostics may legitimately fail before a Fault can be injected.
                # This is a measured parser-induced inapplicability, not a harness failure.
                post=copy.deepcopy(realized['facts']);faulted=copy.deepcopy(post);fault_detected=False;repair_success=None;target_id=None
                diags.append({'component':'fault_applicability','status':'clean','fault_applicable':False,'reason':'parser_induced_no_physical_target'})
                if condition in {'qasc','place'}:
                    ids={pid for f in post for pid in (f.get('physical_ids') or [])};auth=authorize(post,sem,physical_record_ids=ids) if sem else [];support=[_render(f) for f in auth]
                else:auth=[];support=[_render(f) for f in post]
                if answer_mode=='reference':answer='INSUFFICIENT_MEMORY' if not support else ' | '.join(support);adiag={'status':'clean'}
                else:answer=answer_with_support(str(r['raw_utterance']),support,cfg or {});adiag=get_last_diagnostics()
                diags.append({'component':'fault_answer','fault_applicable':False,**adiag})
                physical_rows.append({'case':case,'turn_id':tid,'fault_applicable':False,'fault_ineligibility_reason':'parser_induced_no_physical_target','target':None,'target_id':None,'faulted':faulted,'post':post,'repair_success':None,'fault_detected':False,'support':support,'answer':answer,'diags':diags})
                continue
            target_id=target['fact_id']
            inj=store.diagnostic_inject_managed_fault(user_id=uid,fact_id=target_id,kind=case['fault_class'],case_id=case['case_id']);diags.append({'component':'fault_physical_injection',**inj})
            pre_proj=store.project_managed(user_id=uid);faulted=pre_proj['facts'];fault_detected=state_signature(faulted)!=state_signature(baseline)
            if not fault_detected:raise RuntimeError(f'{case["case_id"]}: physical fault injection did not alter semantic readback')
            if condition in {'css','place'}:
                post,repair_diag=_repair_from_intended(store,uid,intended);diags.append({'component':'fault_css_physical_repair',**repair_diag});repair_success=1.0 if repair_diag['verified'] else 0.0
            else:post=copy.deepcopy(faulted);repair_diag=None;repair_success=0.0
            if condition in {'qasc','place'}:
                ids={pid for f in post for pid in (f.get('physical_ids') or [])};auth=authorize(post,sem,physical_record_ids=ids) if sem else [];support=[_render(f) for f in auth]
            else:auth=[];support=[_render(f) for f in post]
            if answer_mode=='reference':answer='INSUFFICIENT_MEMORY' if not support else ' | '.join(support);adiag={'status':'clean'}
            else:answer=answer_with_support(str(r['raw_utterance']),support,cfg or {});adiag=get_last_diagnostics()
            diags.append({'component':'fault_answer',**adiag})
            physical_rows.append({'case':case,'turn_id':tid,'fault_applicable':True,'fault_ineligibility_reason':None,'target':copy.deepcopy(target),'target_id':target_id,'faulted':copy.deepcopy(faulted),'post':copy.deepcopy(post),'repair_success':repair_success,'fault_detected':fault_detected,'support':support,'answer':answer,'diags':diags})
    finally:
        store.close()

    # Evaluator-only truth is opened only after every physical repair/output exists.
    _,gq=runtime_gold(data_root,registry);out=[]
    for i,x in enumerate(physical_rows,1):
        case=x['case'];tid=x['turn_id'];g=gq[tid];expected_state=g.get('expected_state') or [];expected=g.get('expected_facts') or [];ineligible=g.get('ineligible_facts') or []
        applicable=bool(x.get('fault_applicable',True));pre=_sf1(x['faulted'],expected_state) if applicable else None;postf=_sf1(x['post'],expected_state) if applicable else None
        if applicable:
            gold_slot=_slot_facts(expected_state,x['target']);pre_slot=_sf1(_slot_facts(x['faulted'],x['target']),gold_slot);post_slot=_sf1(_slot_facts(x['post'],x['target']),gold_slot)
        else:pre_slot=post_slot=None
        raw_query=by[tid].get('raw_utterance') or '';af1=answer_score(x['answer'],expected,ineligible,g.get('expect_abstain',False),query=g.get('query_semantics'),raw_query=raw_query);exposure=1.0 if _forbidden_mentions('\n'.join(x['support']),ineligible) else 0.0;gvr=1.0 if _forbidden_answer_mentions(x['answer'],ineligible,raw_query) else 0.0
        row={'case_id':case['case_id'],'base_query_turn_id':tid,'family':case['family'],'fault_class':case['fault_class'],'condition':condition,
             'fault_applicable':applicable,'fault_ineligibility_reason':x.get('fault_ineligibility_reason'),'fault_target':({'fact_id':x['target_id'],'entity':x['target'].get('entity'),'predicate':x['target'].get('predicate')} if applicable else None),'fault_injected_physically':applicable,'fault_detected_by_semantic_readback':x['fault_detected'],
             'repair_target_source':('css_intended_state' if condition in {'css','place'} and applicable else None),'repair_used_hidden_gold':False,'pre_physical_state_signature':state_signature(x['faulted']),'post_physical_state_signature':state_signature(x['post']),
             'pre_state_f1':pre,'post_state_f1':postf,'repair_delta_f1':(postf-pre if applicable else None),'pre_faulted_slot_f1':pre_slot,'post_faulted_slot_f1':post_slot,'faulted_slot_delta_f1':(post_slot-pre_slot if applicable else None),'repair_success':x['repair_success'],
             'post_answer_f1':af1,'unauthorized_support_exposure':exposure,'gvr':gvr,'answer':x['answer'],'provider_diagnostics':x['diags'],'status':from_diagnostics(x['diags']),'progress_index':i,'progress_total':len(ev.fault_cases)}
        out.append(row)
        if pp:append_jsonl(pp,row)
    return out

def _mean(xs):
    vals=[float(x) for x in xs if x is not None]
    return sum(vals)/len(vals) if vals else None

def _group(rows,key):
    out={}
    for name in sorted({str(r.get(key,'unknown')) for r in rows}):
        xs=[r for r in rows if str(r.get(key,'unknown'))==name];app=[r for r in xs if r.get('fault_applicable',True)];succ=sum(r.get('repair_success') is not None and r['repair_success']>=.999999 for r in app)
        out[name]={'n':len(xs),'applicable_n':len(app),'inapplicable_n':len(xs)-len(app),'applicability_rate':len(app)/len(xs) if xs else None,'repair_success_n':succ,'repair_success':(succ/len(app) if app else None),'full_state_delta_f1':_mean(r.get('repair_delta_f1') for r in app),'faulted_slot_delta_f1':_mean(r.get('faulted_slot_delta_f1') for r in app)}
    return out

def summarize(rows):
    n=len(rows);app=[r for r in rows if r.get('fault_applicable',True)];succ=sum(r.get('repair_success') is not None and r['repair_success']>=.999999 for r in app);reasons={}
    for r in rows:
        if not r.get('fault_applicable',True):reasons[r.get('fault_ineligibility_reason') or 'unknown']=reasons.get(r.get('fault_ineligibility_reason') or 'unknown',0)+1
    return {'n':n,'applicable_n':len(app),'inapplicable_n':n-len(app),'applicability_rate':(len(app)/n if n else None),'ineligibility_reasons':reasons,'repair_delta_f1':_mean(r.get('repair_delta_f1') for r in app),'pre_state_f1':_mean(r.get('pre_state_f1') for r in app),'post_state_f1':_mean(r.get('post_state_f1') for r in app),
            'pre_faulted_slot_f1':_mean(r.get('pre_faulted_slot_f1') for r in app),'post_faulted_slot_f1':_mean(r.get('post_faulted_slot_f1') for r in app),'faulted_slot_delta_f1':_mean(r.get('faulted_slot_delta_f1') for r in app),
            'repair_success':(succ/len(app) if app else None),'repair_success_n':succ,'repair_n':len(app),'post_answer_f1':_mean(r.get('post_answer_f1') for r in rows),'exposure':_mean(r.get('unauthorized_support_exposure') for r in rows),'gvr':_mean(r.get('gvr') for r in rows),
            'by_fault_class':_group(rows,'fault_class'),'by_family':_group(rows,'family'),'recovered_error_count':sum(r['status']=='COMPLETE_WITH_RECOVERED_ERRORS' for r in rows)}
