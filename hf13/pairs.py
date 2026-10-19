from __future__ import annotations
import copy,json
from pathlib import Path
from collections import defaultdict
from .backend import make_store,record_text
from .css import CSSKernel,fact_from_event_local,fact_signature,canon
from .dataset import EvaluatorData
from .llm import answer_with_support,get_last_diagnostics
from .qasc import authorize
from .physical_readback import project_physical_records,authorize_physical_readback
from .io import append_jsonl
from .status import from_diagnostics

def _render(f):return f"{f.get('entity')}.{f.get('predicate')}={json.dumps(f.get('value'),ensure_ascii=False,default=str)} [{f.get('lifecycle')}]"
def _literal(v):return str(v).casefold()
def _pair_project(records,events):
    out=[];used=set();seq=0
    for e in events:
        if not e:continue
        ass=e.get('assertion') or {}
        if 'value' not in ass:continue
        v=ass['value'];needle=_literal(v)
        for i,r in enumerate(records):
            if i in used:continue
            if needle and needle in record_text(r).casefold():
                used.add(i);seq+=1;f=fact_from_event_local(e,sequence=seq,value=v,fact_id=f"pair-native:{e['event_id']}:{seq}")
                if f:
                    prior=[x for x in out if x['entity']==f.entity and x['predicate']==f.predicate and canon(x.get('scope') or {})==canon(f.scope)];f.version=1+len(prior);x=f.to_dict();x['physical_ids']=[str(r.get('id') or r.get('memory_id') or '')];out.append(x)
                break
    return out

def _gold_match(facts,gold):
    # gold pair contracts intentionally specify only the authorized semantic slot/value.
    if len(facts)!=len(gold):return False
    for g in gold:
        if not any(f.get('entity')==g.get('entity') and f.get('predicate')==g.get('predicate') and canon(f.get('value'))==canon(g.get('value')) and canon(f.get('scope') or {})==canon(g.get('scope') or {}) and f.get('lifecycle')=='active' for f in facts):return False
    return True

def _outcome(answer,pair):
    # Semantic outcome is evaluated against hidden pair intervention values, never
    # against whatever a parser happened to predict.
    a=str(answer or '').casefold();labels=[];current=str(pair['gold_authorized'][0]['value'])
    iv=pair.get('pair_intervention') or {};olds=[str(x) for x in (iv.get('world_a'),iv.get('world_b')) if x is not None and str(x)!=current]
    if current.casefold() in a:labels.append('CURRENT')
    for v in sorted(set(olds)):
        if v.casefold() in a:labels.append('OLD:'+v)
    if 'insufficient_memory' in a or 'insufficient memory' in a:labels.append('INSUFFICIENT')
    return tuple(labels or ['OTHER'])

def _answer(query,support,answer_mode,cfg):
    if answer_mode=='reference':return 'INSUFFICIENT_MEMORY' if not support else ' | '.join(support),{'status':'clean','attempts':0,'errors':[]}
    x=answer_with_support(query,support,cfg or {});return x,get_last_diagnostics()

def _run_world(pair,world,condition,backend,cfg,answer_mode,result_root):
    sentences=pair[f'{world}_sentences'];events=pair['events_a' if world=='world_a' else 'events_b'];uid=f"pair:{pair['pair_id']}:{condition}:{world}";store=make_store(backend,cfg,result_root=result_root,namespace=uid);kernel=CSSKernel();diags=[]
    try:
        if condition in {'native','qasc'}:
            for s in sentences:
                d=store.native_add(s,user_id=uid,metadata={},role='user');diags.append(d.get('diagnostics',{}))
            facts=_pair_project(store.get_all(user_id=uid),events)
        else:
            # Trusted pair mode has one event per leading event-bearing sentence. In
            # parsed mode a missed event falls back to the raw sentence, matching the
            # longitudinal parsed PLACE runner rather than crashing or inventing gold.
            parsed_mode=bool(pair.get('parsed_mode'));event_n=int(pair.get('event_sentence_count',len(events)))
            for j,sentence in enumerate(sentences):
                ev=events[j] if j<event_n and j<len(events) else None
                od=store.add_observation(sentence,user_id=uid,observation_id=f'pair:{pair["pair_id"]}:{world}:{j+1}',role='user');diags.append({'component':'css_pair_raw_observation',**od.get('diagnostics',{})})
                if ev:
                    kernel.apply(copy.deepcopy(ev));cert=store.reconcile_managed(kernel.snapshot(),user_id=uid);diags.append({'component':'css_pair_certification','status':cert.get('status'),'verified':cert.get('verified'),'infer':False})
            facts=store.project_managed(user_id=uid)['facts']
        qsem=pair.get('query_semantics');af=authorize(facts,qsem) if qsem else []  # evaluator-only causal denominator for every condition
        physical_records=store.get_all(user_id=uid);readback=project_physical_records(physical_records)
        retrieved=[record_text(r) for r in store.search(pair['query'],user_id=uid,limit=10)];diags.append({'component':'pair_query_search',**copy.deepcopy(getattr(store,'last_search_diagnostics',{}) or {})})
        if condition=='qasc':
            runtime_auth=authorize_physical_readback(readback,qsem,raw_query=pair['query']) if qsem else [];support=[str(x.get('text') or '') for x in runtime_auth]
        elif condition=='place':runtime_auth=af;support=[_render(f) for f in af]
        else:runtime_auth=[];support=list(retrieved)
        answer,ad=_answer(pair['query'],support,answer_mode,cfg);diags.append(ad)
        nret=[record_text(r) for r in store.search(pair['needed_query'],user_id=uid,limit=10)];diags.append({'component':'pair_needed_search',**copy.deepcopy(getattr(store,'last_search_diagnostics',{}) or {})});nans,nd=_answer(pair['needed_query'],nret,answer_mode,cfg);diags.append(nd)
        needed_ok=pair['needed_value'].casefold() in ('\n'.join(nret)+' '+nans).casefold() and pair['needed_value'].casefold() in nans.casefold()
        return {'physical_state':facts,'physical_semantic_readback':readback,'retrieval_candidates':retrieved,'authorized_support':runtime_auth,'evaluator_authorized_state':af,'query_authorized_support':support if condition in {'qasc','place'} else [],'model_visible_support':support,'response':answer,'facts':facts,'authorized':af,'support':support,'answer':answer,'outcome':_outcome(answer,pair),'needed_sra':1.0 if needed_ok else 0.0,'provider_diagnostics':diags}
    finally:store.close()

def run_pair(pair,condition,*,backend='reference',cfg=None,answer_mode='reference',result_root='results/hf13_pairs'):
    A=_run_world(pair,'world_a',condition,backend,cfg,answer_mode,result_root);B=_run_world(pair,'world_b',condition,backend,cfg,answer_mode,result_root)
    sigA=sorted(fact_signature(x) for x in A['authorized']);sigB=sorted(fact_signature(x) for x in B['authorized']);equal=sigA==sigB;goldA=_gold_match(A['authorized'],pair['gold_authorized']);goldB=_gold_match(B['authorized'],pair['gold_authorized']);eligible=bool(equal and goldA and goldB);state_mediated=not equal;equal_wrong=bool(equal and not (goldA and goldB));semantic_div=A['outcome']!=B['outcome'];raw_div=A['answer']!=B['answer'];olr=bool(eligible and semantic_div)
    current=str(pair['gold_authorized'][0]['value']).casefold();correctA=current in A['answer'].casefold();correctB=current in B['answer'].casefold();old_violation=any(x.startswith('OLD:') for x in A['outcome']+B['outcome']);needed=min(A['needed_sra'],B['needed_sra']);jaur=1.0 if eligible and not olr and needed==1.0 and correctA and correctB and not old_violation else 0.0
    diags=A['provider_diagnostics']+B['provider_diagnostics'];status=from_diagnostics(diags)
    return {'pair_id':pair['pair_id'],'family':pair['family'],'condition':condition,'physical_authorized_state_equal':equal,'gold_correct_a':goldA,'gold_correct_b':goldB,'eligible':eligible,'state_mediated_divergence':state_mediated,'equal_but_wrong':equal_wrong,'semantic_outcome_diverged':semantic_div,'raw_answer_diverged':raw_div,'ipdr_event':bool(eligible and raw_div and not semantic_div),'olr_event':olr,'gvr_event':bool(old_violation),'needed_sra':needed,'jaur':jaur,'outcome_a':A['outcome'],'outcome_b':B['outcome'],'answer_a':A['answer'],'answer_b':B['answer'],'model_visible_support_a':A['model_visible_support'],'model_visible_support_b':B['model_visible_support'],'authorized_support_a':A['authorized_support'],'authorized_support_b':B['authorized_support'],'authorized_a':sigA,'authorized_b':sigB,'provider_diagnostics':diags,'status':status}

def summarize(rows):
    fam={};groups=defaultdict(list)
    for r in rows:groups[r['family']].append(r)
    def sm(xs):
        n=len(xs);eligible=[x for x in xs if x['eligible']]
        return {'n':n,'eligible_n':len(eligible),'olr':(sum(x['olr_event'] for x in eligible)/len(eligible) if eligible else None),'state_mediated_n':sum(x['state_mediated_divergence'] for x in xs),'equal_but_wrong_n':sum(x['equal_but_wrong'] for x in xs),'ipdr_n':sum(x['ipdr_event'] for x in xs),'needed_sra':sum(x['needed_sra'] for x in xs)/n,'gvr':sum(x['gvr_event'] for x in xs)/n,'jaur':sum(x['jaur'] for x in xs)/n}
    for k,xs in groups.items():fam[k]=sm(xs)
    allm=sm(rows);allm['family_metrics']=fam;den=sum(v['olr'] is not None for v in fam.values());allm['macro_family_olr']=(sum(v['olr'] for v in fam.values() if v['olr'] is not None)/den if den else None)
    return allm

def run_all(data_root='data/longitudinal',condition='place',progress_path=None,**kw):
    pairs=EvaluatorData(data_root).causal_pairs;out=[];pp=Path(progress_path) if progress_path else None
    if pp:pp.parent.mkdir(parents=True,exist_ok=True);pp.write_text('',encoding='utf-8')
    for i,p in enumerate(pairs,1):
        r=run_pair(p,condition,**kw);r['progress_index']=i;r['progress_total']=len(pairs);out.append(r)
        if pp:append_jsonl(pp,r)
    return out
