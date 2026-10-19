from __future__ import annotations
from collections import defaultdict
from pathlib import Path
from .dataset import EvaluatorData
from .io import read_jsonl,append_jsonl
from .llm import answer_with_support,get_last_diagnostics
from .evaluator import runtime_gold,answer_score,_mentioned,_forbidden_mentions,_forbidden_answer_mentions
from .evaluator_projection import EvaluatorPhysicalProjector


def _gold_text(expected):
    vals=[]
    for g in expected:
        v=g.get('value')
        if isinstance(v,dict) and v.get('text'):v=v['text']
        vals.append(str(v))
    return '; '.join(vals) or 'no authorized persistent fact'

def needed_sra_score(saved,visible):
    return 1.0 if saved and visible else 0.0

def _conflict(ineligible,case_id):
    if ineligible:
        v=ineligible[0].get('value');return str(v.get('text') if isinstance(v,dict) and v.get('text') else v)
    return 'CONFLICT-'+case_id.replace(':','-')

def run_utility(primary_rows,condition,*,data_root='data/longitudinal',answer_mode='reference',cfg=None,progress_path=None,artifact_registry=None):
    if isinstance(primary_rows,(str,Path)):primary_rows=read_jsonl(primary_rows)
    ev=EvaluatorData(data_root);_,gq=runtime_gold(data_root,artifact_registry);by={r['turn_id']:r for r in primary_rows};projector=EvaluatorPhysicalProjector(data_root,artifact_registry) if condition in {'native','qasc'} else None;out=[];pp=Path(progress_path) if progress_path else None
    if pp:pp.parent.mkdir(parents=True,exist_ok=True);pp.write_text('',encoding='utf-8')
    for i,case in enumerate(ev.utility_cases,1):
        r=by[case['base_query_turn_id']];q=r.get('query') or {};g=gq[case['base_query_turn_id']];expected=g.get('expected_facts') or [];ineligible=g.get('ineligible_facts') or [];physical=(projector.project(q.get('physical_backend_records') or [],through_turn_id=case['base_query_turn_id'],include_through=False)['facts'] if projector else (q.get('physical_state') or []));support=list(q.get('model_visible_support') or []);mode=case['mode'];conflict=None
        if mode=='needed':context='Visible context intentionally does not contain the requested persistent-memory fact.'
        elif mode=='redundant':context='Visible context independently states the answer: '+_gold_text(expected)
        else:conflict=_conflict(ineligible,case['case_id']);context='Visible context contains a potentially stale/conflicting claim: '+conflict
        if answer_mode=='reference':
            if mode=='redundant':answer=_gold_text(expected)
            elif support:answer=' | '.join(support)
            else:answer='INSUFFICIENT_MEMORY' if mode!='conflict' else conflict
            diag={'status':'clean'}
        else:answer=answer_with_support(str(r['raw_utterance'])+'\nCONTEXT NOTE: '+context,support,cfg or {});diag=get_last_diagnostics()
        raw_query=r.get('raw_utterance') or '';af1=answer_score(answer,expected,ineligible,g.get('expect_abstain',False),query=g.get('query_semantics'),raw_query=raw_query);support_text='\n'.join(support)
        needed_applicable=bool(expected) if mode=='needed' else None
        saved=all(any(f.get('entity')==x.get('entity') and f.get('predicate')==x.get('predicate') and f.get('value')==x.get('value') for f in physical) for x in expected) if expected else None
        visible=all(_mentioned(support_text,x.get('value')) for x in expected) if expected else None
        # Needed SRA is support preservation only. Answer correctness is a separate gate.
        needed_sra=1.0 if mode!='needed' else (needed_sra_score(saved,visible) if needed_applicable else None)
        harm=1.0 if mode=='conflict' and conflict and str(conflict).casefold() in str(answer).casefold() and af1<.999999 else 0.0
        exposure=1.0 if _forbidden_mentions(support_text,ineligible) else 0.0;gvr=1.0 if _forbidden_answer_mentions(answer,ineligible,raw_query) else 0.0
        row={'case_id':case['case_id'],'base_query_turn_id':case['base_query_turn_id'],'family':case['family'],'mode':mode,'condition':condition,'answer':answer,'answer_f1':af1,'needed_sra':needed_sra,'needed_applicable':needed_applicable,'needed_fact_physically_preserved':saved,'needed_fact_model_visible':visible,'harm':harm,'unauthorized_support_exposure':exposure,'gvr':gvr,'model_visible_support_count':len(support),'provider_diagnostics':[diag],'status':'FAILED' if diag.get('status')=='failed' else ('COMPLETE_WITH_RECOVERED_ERRORS' if diag.get('status')=='recovered' else 'COMPLETE_CLEAN'),'progress_index':i,'progress_total':len(ev.utility_cases)};out.append(row)
        if pp:append_jsonl(pp,row)
    return out

def _family_summary(rows):
    out={}
    for family in sorted({r['family'] for r in rows}):
        xs=[r for r in rows if r['family']==family];needed_all=[r for r in xs if r['mode']=='needed'];needed=[r for r in needed_all if r.get('needed_applicable') is not False and r.get('needed_sra') is not None];conf=[r for r in xs if r['mode']=='conflict']
        out[family]={'n':len(xs),'needed_total_n':len(needed_all),'needed_n':len(needed),'needed_inapplicable_n':len(needed_all)-len(needed),'needed_success_n':sum(r['needed_sra']>=.999999 for r in needed),'needed_sra':(sum(r['needed_sra'] for r in needed)/len(needed) if needed else None),'conflict_n':len(conf),'conflict_harm_n':sum(r['harm']>=.999999 for r in conf),'harm':(sum(r['harm'] for r in conf)/len(conf) if conf else None)}
    return out

def summarize(rows):
    needed_all=[r for r in rows if r['mode']=='needed'];needed=[r for r in needed_all if r.get('needed_applicable') is not False and r.get('needed_sra') is not None];conf=[r for r in rows if r['mode']=='conflict'];needed_success=sum(r['needed_sra']>=.999999 for r in needed);harm_n=sum(r['harm']>=.999999 for r in conf)
    return {'n':len(rows),'needed_total_n':len(needed_all),'needed_n':len(needed),'needed_inapplicable_n':len(needed_all)-len(needed),'needed_success_n':needed_success,'needed_sra':(needed_success/len(needed) if needed else None),'conflict_n':len(conf),'conflict_harm_n':harm_n,'harm':harm_n/len(conf),'exposure':sum(r['unauthorized_support_exposure'] for r in rows)/len(rows),'gvr':sum(r['gvr'] for r in rows)/len(rows),'answer_f1':sum(r['answer_f1'] for r in rows)/len(rows),'by_family':_family_summary(rows),'recovered_error_count':sum(r['status']=='COMPLETE_WITH_RECOVERED_ERRORS' for r in rows)}
