#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, importlib.util, json, os, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
LEGACY=ROOT/'scripts'/'_frozen_inputs_compat.py'
spec=importlib.util.spec_from_file_location('hf13_rescore_v1_impl',LEGACY)
v1=importlib.util.module_from_spec(spec); assert spec.loader; spec.loader.exec_module(v1)

from hf13.answer_evaluator_strict_general import (
    score_answer_strict_general,EVALUATOR_VERSION,structured_expected_match,scalar_match,
)
from hf13.answer_validation import add_bertscore
from hf13.utility import _conflict
from hf13.semantic_measurement_v26 import SemanticEvidenceResolver
from hf13.evaluator import runtime_gold, f1_sets, _semkey, _authorized_contract
from hf13.io import read_jsonl

V1_SCORE=v1.score_answer


def score_wrapper(response, gold, raw_query, stored_old, *, context):
    z=V1_SCORE(response,gold,raw_query,stored_old,context=context)
    r=score_answer_strict_general(response,gold,raw_query,original_overrides=False)
    z.update({
        'answer_evaluable_strict_general': bool(r.evaluable),
        'corrected_answer_f1_strict_general': r.answer_f1,
        'corrected_gvr_strict_general': r.gvr,
        'corrected_answer_delta_strict_general': (None if r.answer_f1 is None else float(r.answer_f1)-float(stored_old)),
        'corrected_answer_tp_strict_general': r.tp,
        'corrected_forbidden_match_count_strict_general': r.forbidden_match_count,
        'corrected_obligation_count_strict_general': r.obligation_count,
        'corrected_obligation_scores_strict_general': r.obligation_scores,
        'corrected_exclusion_reason_strict_general': r.exclusion_reason,
        'corrected_obligation_notes_strict_general': r.obligation_notes,
        'corrected_obligations_strict_general': r.obligations,
    })
    return z

# Reuse the frozen artifact readers / reconstruction gates from v1, but replace
# only the scoring callback.  Every row must still reproduce the original stored
# Answer F1 before v3.2-strict-general is allowed to score it.
v1.score_answer=score_wrapper


def _semantic_measurement_strict_general(rows, campaign, data_root, model, experiment):
    """Correct support visibility/exposure only; preserve canonical state measurement.

    v2.6 experimentally re-projected Native/QASC state from free-text physical records.
    Candidate audit showed many records can match multiple historical events, so v3.2-strict-general
    does not use that ambiguous projection for State F1. Query/state metrics remain the
    frozen canonical evaluator values; only support-side representation matching is
    corrected from the actual model-visible support.
    """
    registry=campaign/model/'runtime_artifacts'/'runtime_artifact_registry.json'
    _,gq=runtime_gold(data_root,registry)
    resolver=SemanticEvidenceResolver(data_root,registry)
    byrow={(r['condition'],r['turn_id']):r for r in rows if r.get('experiment')==experiment and r.get('model')==model}
    state_summary={}
    for cond in ('native','css','qasc','place'):
        tp=campaign/model/experiment/cond/'trace.jsonl'
        trace=read_jsonl(tp); by={r['turn_id']:r for r in trace}
        # Preserve the original/canonical state aggregate from the frozen evaluation.
        rp=campaign/model/experiment/cond/('evaluation.json' if experiment=='primary' else 'result.json')
        ev=json.loads(rp.read_text())
        state_summary[cond]={'state_checkpoint_n':ev.get('state_checkpoint_count'),
                             'corrected_state_f1_strict_general':ev.get('state_f1_mean')}
        for tid,g in gq.items():
            tr=by[tid]; q=tr.get('query') or {}; rr=byrow[(cond,tid)]
            sm=resolver.support_metrics(q,g.get('expected_facts') or [],g.get('ineligible_facts') or [])
            auth=None
            if cond in {'qasc','place'}:
                auth=_authorized_contract(q.get('authorized_support') or [],g.get('expected_facts') or [],g.get('ineligible_facts') or [])
            support_ok=bool(sm['needed_sra']>=.999999 and sm['exposure']==0.0 and auth is not False)
            rr.update({
                'corrected_query_state_f1_strict_general':rr.get('query_state_f1'),
                'corrected_needed_sra_strict_general':sm['needed_sra'],
                'corrected_unauthorized_support_exposure_strict_general':sm['exposure'],
                'corrected_support_correct_strict_general':support_ok,
                'corrected_exposed_forbidden_count_strict_general':len(sm['exposed_facts']),
                'corrected_support_exposure_detail_strict_general':sm['exposure_match_detail'],
            })
        qrows=[r for r in rows if r.get('experiment')==experiment and r.get('model')==model and r.get('condition')==cond]
        state_summary[cond].update({
            'query_checkpoint_n':len(qrows),
            'corrected_query_state_f1_strict_general':mean(r.get('corrected_query_state_f1_strict_general') for r in qrows),
            'corrected_needed_sra_strict_general':mean(r.get('corrected_needed_sra_strict_general') for r in qrows),
            'corrected_exposure_strict_general':mean(r.get('corrected_unauthorized_support_exposure_strict_general') for r in qrows),
        })
    return state_summary

def _utility_measurement_strict_general(rows, campaign, data_root, model, goldq):
    registry=campaign/model/'runtime_artifacts'/'runtime_artifact_registry.json'
    resolver=SemanticEvidenceResolver(data_root,registry)
    for cond in ('native','css','qasc','place'):
        trace={r['turn_id']:r for r in read_jsonl(campaign/model/'primary'/cond/'trace.jsonl')}
        for r in rows:
            if r.get('experiment')!='utility' or r.get('model')!=model or r.get('condition')!=cond: continue
            tid=r['turn_id']; g=goldq[tid]; q=(trace[tid].get('query') or {})
            sm=resolver.support_metrics(q,g.get('expected_facts') or [],g.get('ineligible_facts') or [])
            r['corrected_unauthorized_support_exposure_strict_general']=sm['exposure']
            r['corrected_support_exposure_detail_strict_general']=sm['exposure_match_detail']
            if r.get('mode')=='needed':
                saved=bool(r.get('needed_fact_physically_preserved'))
                r['corrected_needed_fact_physically_preserved_strict_general']=saved
                r['corrected_needed_fact_model_visible_strict_general']=bool(sm['needed_sra']>=.999999)
                r['corrected_needed_sra_strict_general']=float(saved and sm['needed_sra']>=.999999) if r.get('needed_applicable') is not False else None
            else:
                r['corrected_needed_sra_strict_general']=r.get('needed_sra')

def add_strict_general_derived(rows):
    for r in rows:
        ev=bool(r.get('answer_evaluable_strict_general'))
        if r['experiment'] in {'primary','closed_loop'}:
            if not ev:
                r['corrected_jaur_strict_general']=None
            else:
                state=float(r.get('corrected_query_state_f1_strict_general') or 0.0)>=.999999
                support=bool(r.get('corrected_support_correct_strict_general'))
                needed=float(r.get('corrected_needed_sra_strict_general') or 0.0)>=.999999
                exposure=float(r.get('corrected_unauthorized_support_exposure_strict_general') or 0.0)==0.0
                r['corrected_jaur_strict_general']=float(state and support and needed and exposure and float(r.get('corrected_gvr_strict_general') or 0)==0.0 and float(r.get('corrected_answer_f1_strict_general') or 0)>=.999999)
        elif r['experiment']=='parser_robustness':
            if not ev:
                r['corrected_jaur_strict_general']=None
            else:
                state=float(r.get('query_state_f1') or 0.0)>=.999999
                support=float(r.get('authorized_support_f1') or 0.0)>=.999999
                exposure=float(r.get('unauthorized_support_exposure') or 0.0)==0.0
                r['corrected_jaur_strict_general']=float(state and support and exposure and float(r.get('corrected_gvr_strict_general') or 0)==0.0 and float(r.get('corrected_answer_f1_strict_general') or 0)>=.999999)
        else:
            r['corrected_jaur_strict_general']=None


def add_utility_harm_strict_general(rows, goldq):
    """Recompute Utility conflict harm with the corrected v3.2-strict-general answer metric.

    Original Utility defines harm only for ``mode=conflict`` and requires both
    (a) adoption/repetition of the injected conflicting value and (b) an
    incorrect answer.  The legacy row stores harm under the old matcher, so a
    corrected answer release must emit a corrected harm rather than silently
    leaving v1-derived harm beside v3.2-strict-general Answer F1.
    """
    for r in rows:
        if r.get('experiment')!='utility':
            continue
        if r.get('mode')!='conflict':
            r['corrected_conflict_mentioned_strict_general']=None
            r['corrected_harm_strict_general']=0.0
            continue
        g=goldq[r['turn_id']]
        ineligible=g.get('ineligible_facts') or []
        conflict=_conflict(ineligible,r['row_id'])
        response=str(r.get('response') or '')
        mentioned=False
        if ineligible:
            # Require the semantic conflicting value, not merely a formatting-
            # identical string. Structured values require their semantic leaves.
            mentioned=structured_expected_match(response,ineligible[0],require_entity=False)
        if not mentioned:
            mentioned=scalar_match(response,conflict,predicate=(ineligible[0].get('predicate') if ineligible else None))
        r['corrected_conflict_mentioned_strict_general']=bool(mentioned)
        if not r.get('answer_evaluable_strict_general'):
            r['corrected_harm_strict_general']=None
        else:
            r['corrected_harm_strict_general']=float(bool(mentioned) and float(r.get('corrected_answer_f1_strict_general') or 0.0)<.999999)


def add_parser_case_metadata_strict_general(rows, campaign, model):
    """Attach parser-error flags needed to recompute answer-dependent propagation."""
    p=campaign/model/'parser_robustness'/'result.json'
    obj=json.loads(p.read_text(encoding='utf-8'))
    by={c['case_id']:c for c in (obj.get('case_rows') or [])}
    for r in rows:
        if r.get('experiment')!='parser_robustness' or r.get('model')!=model:
            continue
        cid=str(r['row_id']).rsplit(':',1)[0]
        src=by[cid]
        r['parser_case_id']=cid
        r['parser_event_error']=bool(src.get('event_parser_error'))
        r['parser_query_error']=bool(src.get('query_parser_error'))


def _primary_effects_strict_general(summary):
    out={}
    for model,conds in (summary.get('primary') or {}).items():
        if not all(c in conds for c in ('native','css','qasc','place')):
            continue
        n,c,q,p=(conds[x] for x in ('native','css','qasc','place'))
        out[model]={
            'answer_place_minus_native':p['corrected_answer_f1_strict_general']-n['corrected_answer_f1_strict_general'],
            'jaur_place_minus_native':p['corrected_jaur_strict_general']-n['corrected_jaur_strict_general'],
            'jaur_interaction':(p['corrected_jaur_strict_general']-c['corrected_jaur_strict_general'])-(q['corrected_jaur_strict_general']-n['corrected_jaur_strict_general']),
            'note':'point estimates on the common v3.2-strict-general-evaluable Primary queries; no new interval is implied',
        }
    return out


def _utility_harm_summary_strict_general(rows):
    out={}
    for model in ('7b','3b'):
        mr=[r for r in rows if r.get('experiment')=='utility' and r.get('model')==model]
        if not mr: continue
        out[model]={}
        for cond in ('native','css','qasc','place'):
            xs=[r for r in mr if r.get('condition')==cond and r.get('mode')=='conflict']
            ev=[r for r in xs if r.get('answer_evaluable_strict_general') and r.get('corrected_harm_strict_general') is not None]
            out[model][cond]={
                'conflict_n_total':len(xs),
                'conflict_n_evaluable_strict_general':len(ev),
                'original_harm_same_rows':mean(r.get('original_harm') for r in ev),
                'representation_v1_harm_same_rows':mean(r.get('representation_rescored_harm') for r in ev),
                'corrected_harm_strict_general':mean(r.get('corrected_harm_strict_general') for r in ev),
                'corrected_conflict_mention_rate_strict_general':mean(float(bool(r.get('corrected_conflict_mentioned_strict_general'))) for r in ev),
            }
    return out


def _parser_downstream_strict_general(rows):
    """Recompute answer-dependent parser deltas/error propagation under v3.2-strict-general.

    Parser event/query structured F1 is intentionally untouched. Only downstream
    quantities whose original definitions directly used Answer F1/JAUR are
    recomputed here.
    """
    out={}
    for model in ('7b','3b'):
        mr=[r for r in rows if r.get('experiment')=='parser_robustness' and r.get('model')==model]
        if not mr: continue
        cases={}
        for r in mr:
            cases.setdefault(r.get('parser_case_id') or str(r['row_id']).rsplit(':',1)[0],{})[r['condition']]=r
        usable={cid:z for cid,z in cases.items() if all(c in z for c in ('TT','PT','TP','PP')) and all(z[c].get('answer_evaluable_strict_general') for c in ('TT','PT','TP','PP'))}
        deltas={}
        for cond in ('PT','TP','PP'):
            deltas[cond]={
                'answer_f1_minus_TT':mean(z[cond]['corrected_answer_f1_strict_general']-z['TT']['corrected_answer_f1_strict_general'] for z in usable.values()),
                'jaur_minus_TT':mean(z[cond]['corrected_jaur_strict_general']-z['TT']['corrected_jaur_strict_general'] for z in usable.values()),
            }
        event_harm=query_harm=pp_harm=0
        event_err_n=query_err_n=any_err_n=0
        for z in usable.values():
            tt,pt,tp,pp=(z[c] for c in ('TT','PT','TP','PP'))
            ee=bool(tt.get('parser_event_error')); qe=bool(tt.get('parser_query_error')); ae=ee or qe
            event_err_n+=int(ee); query_err_n+=int(qe); any_err_n+=int(ae)
            if ee and (float(pt.get('query_state_f1') or 0)<.999999 or float(pt.get('authorized_support_f1') or 0)<.999999 or float(pt['corrected_answer_f1_strict_general'])+1e-12<float(tt['corrected_answer_f1_strict_general']) or float(pt['corrected_jaur_strict_general'])<float(tt['corrected_jaur_strict_general'])):
                event_harm+=1
            if qe and (float(tp.get('authorized_support_f1') or 0)<.999999 or float(tp['corrected_answer_f1_strict_general'])+1e-12<float(tt['corrected_answer_f1_strict_general']) or float(tp['corrected_jaur_strict_general'])<float(tt['corrected_jaur_strict_general'])):
                query_harm+=1
            if ae and (float(pp.get('query_state_f1') or 0)<.999999 or float(pp.get('authorized_support_f1') or 0)<.999999 or float(pp['corrected_answer_f1_strict_general'])+1e-12<float(tt['corrected_answer_f1_strict_general']) or float(pp['corrected_jaur_strict_general'])<float(tt['corrected_jaur_strict_general'])):
                pp_harm+=1
        out[model]={
            'case_count_total':len(cases),'case_count_strict_general_evaluable':len(usable),
            'deltas_vs_TT':deltas,
            'error_propagation':{
                'event_parser_error_n':event_err_n,'event_error_propagated_n_strict_general':event_harm,
                'query_parser_error_n':query_err_n,'query_error_propagated_n_strict_general':query_harm,
                'any_parser_error_n':any_err_n,'end_to_end_error_propagated_n_strict_general':pp_harm,
            },
            'note':'parser structured metrics are unchanged; only answer/JAUR-dependent downstream derivations are recomputed',
        }
    return out


def mean(xs):
    vals=[float(x) for x in xs if x is not None and str(x)!='']
    return sum(vals)/len(vals) if vals else None


def summarize(rows):
    out={}
    for exp in sorted({r['experiment'] for r in rows}):
        out[exp]={}
        for model in ('7b','3b'):
            mr=[r for r in rows if r['experiment']==exp and r['model']==model]
            if not mr: continue
            out[exp][model]={}
            for cond in sorted({r['condition'] for r in mr}):
                xs=[r for r in mr if r['condition']==cond]
                ev=[r for r in xs if r.get('answer_evaluable_strict_general')]
                out[exp][model][cond]={
                    'n_total':len(xs), 'n_evaluable_strict_general':len(ev), 'coverage_strict_general':len(ev)/len(xs) if xs else None,
                    'original_answer_f1_all_rows':mean(r.get('original_answer_f1') for r in xs),
                    'original_answer_f1_on_strict_general_evaluable_rows':mean(r.get('original_answer_f1') for r in ev),
                    'representation_equivalence_v1_on_strict_general_evaluable_rows':mean(r.get('representation_equivalence_answer_f1') for r in ev),
                    'corrected_answer_f1_strict_general':mean(r.get('corrected_answer_f1_strict_general') for r in ev),
                    'corrected_gvr_strict_general_all_rows':mean(r.get('corrected_gvr_strict_general') for r in xs),
                    'original_gvr_all_rows':mean(r.get('original_gvr') for r in xs),
                    'original_jaur_on_strict_general_evaluable_rows':mean(r.get('original_jaur') for r in ev),
                    'corrected_jaur_strict_general':mean(r.get('corrected_jaur_strict_general') for r in ev),
                    'corrected_query_state_f1_strict_general':mean(r.get('corrected_query_state_f1_strict_general') for r in xs),
                    'corrected_needed_sra_strict_general':mean(r.get('corrected_needed_sra_strict_general') for r in xs),
                    'corrected_exposure_strict_general':mean(r.get('corrected_unauthorized_support_exposure_strict_general') for r in xs),
                    'bertscore_f1':mean(r.get('bertscore_f1') for r in ev),
                    'changed_strict_general_rows':sum(r.get('corrected_answer_f1_strict_general') is not None and abs(float(r['corrected_answer_f1_strict_general'])-float(r['original_answer_f1']))>1e-12 for r in ev),
                    'excluded_turn_ids_strict_general':sorted({r['turn_id'] for r in xs if not r.get('answer_evaluable_strict_general')}),
                }
                if exp=='utility':
                    conf=[r for r in ev if r.get('mode')=='conflict' and r.get('corrected_harm_strict_general') is not None]
                    out[exp][model][cond].update({
                        'corrected_harm_strict_general':mean(r.get('corrected_harm_strict_general') for r in conf),
                        'original_harm_on_strict_general_evaluable_conflict_rows':mean(r.get('original_harm') for r in conf),
                        'conflict_n_evaluable_strict_general':len(conf),
                    })
    return out


def write_csv(path,rows):
    preferred=['experiment','model','condition','row_id','turn_id','conversation_id','family','mode','fault_class',
      'answer_evaluable_strict_general','corrected_exclusion_reason_strict_general','original_answer_f1','representation_equivalence_answer_f1','corrected_answer_f1_strict_general','corrected_answer_delta_strict_general',
      'original_gvr','representation_equivalence_gvr','corrected_gvr_strict_general','original_jaur','representation_rescored_jaur','corrected_jaur_strict_general',
      'query_state_f1','corrected_query_state_f1_strict_general','needed_sra','corrected_needed_sra_strict_general','unauthorized_support_exposure','corrected_unauthorized_support_exposure_strict_general','support_correct','corrected_support_correct_strict_general',
      'original_harm','representation_rescored_harm','corrected_conflict_mentioned_strict_general','corrected_harm_strict_general',
      'parser_case_id','parser_event_error','parser_query_error',
      'corrected_answer_tp_strict_general','corrected_forbidden_match_count_strict_general','corrected_obligation_count_strict_general','bertscore_f1','token_f1','rouge_l_f1','chrf','bleu4','raw_query','response']
    keys=[]; seen=set()
    for k in preferred:
        if any(k in r for r in rows): keys.append(k);seen.add(k)
    for r in rows:
        for k in r:
            if k not in seen and k not in {'canonical_references','representation_match_detail','semantic_sensitivity_detail','corrected_obligations_strict_general','corrected_obligation_scores_strict_general','corrected_obligation_notes_strict_general'}:
                keys.append(k);seen.add(k)
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader()
        for r in rows:
            w.writerow({k:(json.dumps(r.get(k),ensure_ascii=False,sort_keys=True) if isinstance(r.get(k),(dict,list)) else r.get(k)) for k in keys})


def report(summary,rows,meta):
    L=['# HF13 Corrected Semantic Answer Evaluation v3.2-strict-general','',
       'This is a **read-only evaluator correction over frozen outputs**. It does not rerun Qwen, Mem0, retrieval, CSS, QASC, state transitions, runtime artifacts, or parser predictions.','', 'Answer F1 v3.2-strict-general measures required semantic answer content independently of forbidden-memory disclosure. GVR measures forbidden disclosure separately; JAUR requires both semantic answer correctness and zero GVR.','',
       'v2 corrects three distinct measurement problems found by audit: (1) typed/surface representation mismatches, (2) query-action obligations that were previously reduced to generic cue words, and (3) answer-gold obligations that were auxiliary support or runtime-artifact references rather than what the user asked.','',
       '**Original submitted metrics are retained for provenance. Corrected v3.2-strict-general aggregates are computed only on rows whose answer obligation is safely evaluable; the same excluded query IDs apply to every model and condition.**','',
       f"Source campaign: `{meta['campaign']}`", f"Scopes: `{', '.join(meta['scopes'])}`", f"Rows processed: **{len(rows)}**", '']
    for exp,models in summary.items():
        L += [f'## {exp}','']
        for model,conds in models.items():
            L += [f'### {model.upper()}','',
              '| Condition | Total | Eval. | Coverage | **State@Query v3.2-strict-general** | **Needed SRA v3.2-strict-general** | **Exposure v3.2-strict-general** | Original F1 (same rows) | **Corrected F1 v3.2-strict-general** | **GVR v3.2-strict-general** | **JAUR v3.2-strict-general** |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
            order=[c for c in ('native','css','qasc','place','TT','PT','TP','PP') if c in conds]
            for c in order:
                z=conds[c]
                L.append(f"| {c} | {z['n_total']} | {z['n_evaluable_strict_general']} | {z['coverage_strict_general']:.3f} | **{v1.fmt(z.get('corrected_query_state_f1_strict_general'))}** | **{v1.fmt(z.get('corrected_needed_sra_strict_general'))}** | **{v1.fmt(z.get('corrected_exposure_strict_general'))}** | {v1.fmt(z['original_answer_f1_on_strict_general_evaluable_rows'])} | **{v1.fmt(z['corrected_answer_f1_strict_general'])}** | **{v1.fmt(z['corrected_gvr_strict_general_all_rows'])}** | **{v1.fmt(z['corrected_jaur_strict_general'])}** |")
            L += ['']
            if exp=='utility':
                L += ['Corrected Utility conflict harm (same v3.2-strict-general-evaluable conflict rows):','',
                      '| Condition | Conflict n | Original harm | Corrected harm v3.2-strict-general |','|---|---:|---:|---:|']
                for c in [x for x in ('native','css','qasc','place') if x in conds]:
                    z=conds[c]
                    L.append(f"| {c} | {z.get('conflict_n_evaluable_strict_general',0)} | {v1.fmt(z.get('original_harm_on_strict_general_evaluable_conflict_rows'))} | **{v1.fmt(z.get('corrected_harm_strict_general'))}** |")
                L += ['']
    excluded={}
    for r in rows:
        if not r.get('answer_evaluable_strict_general'):
            excluded.setdefault(r['turn_id'],r.get('corrected_exclusion_reason_strict_general'))
    L += ['## Query-level exclusions / validity flags','']
    for tid,reason in sorted(excluded.items()): L.append(f'- `{tid}` — {reason}')
    L += ['','## Important metric boundaries','',
      '- State F1 / query State F1 remain the frozen canonical HF13 state metrics. v3.2-strict-general corrects only representation-sensitive answer/support measurements; it does not semantically reconstruct Native/QASC state from free text.',
      '- Parser structured F1 and Fault repair-state metrics remain separate and are not rewritten by the answer/support evaluator.',
      '- Utility conflict harm is recomputed because its original definition depends on answer correctness; parser downstream deltas/error-propagation are also recomputed where they depend on Answer F1/JAUR.',
      '- GVR v3.2-strict-general uses asymmetric forbidden matching: any disclosed protected leaf is sufficient. This prevents partial passport-number disclosure from disappearing merely because a whole dict was not reproduced.',
      '- BERTScore remains supplementary; it never flips hard correctness.',
      '- Runtime-artifact binding validity is audited separately. The answer evaluator does not silently repair a wrong runtime binding.',
      '']
    return '\n'.join(L)+'\n'


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--campaign-root',default='results/hf13_live/hf13_full_scientific_rerun_20260913_004919')
    ap.add_argument('--data-root',default='data/longitudinal')
    ap.add_argument('--scopes',default='primary')
    ap.add_argument('--models',default='7b,3b')
    ap.add_argument('--output-dir',default='results/strict_general_answer_eval')
    ap.add_argument('--source-manifest',default='')
    ap.add_argument('--bertscore',choices=['off','auto','on'],default='auto')
    ap.add_argument('--bertscore-model',default='roberta-large')
    ap.add_argument('--bertscore-device',default=None)
    ap.add_argument('--bertscore-batch-size',type=int,default=16)
    args=ap.parse_args()
    root=Path.cwd(); campaign=(root/args.campaign_root).resolve(); data_root=(root/args.data_root).resolve(); out=(root/args.output_dir).resolve(); out.mkdir(parents=True,exist_ok=True)
    status=(campaign/'CAMPAIGN_STATUS').read_text().strip()
    if status not in {'COMPLETE_CLEAN','COMPLETE_WITH_MODEL_OPERATION_ERRORS','COMPLETE_WITH_RECOVERED_ERRORS'}: raise SystemExit(f'campaign not complete: {status}')
    scopes=tuple(x.strip() for x in args.scopes.split(',') if x.strip()); models=tuple(x.strip() for x in args.models.split(',') if x.strip())
    unknown=set(scopes)-set(v1.DEFAULT_SCOPES)
    if unknown: raise SystemExit(f'Unknown scopes: {sorted(unknown)}')
    if args.source_manifest:
        manifest=v1.load_manifest((root/args.source_manifest).resolve()); integrity=v1.verify_source_manifest(root,manifest)
    else:
        integrity={'schema_version':'hf13-run-local-source-integrity-v1','status':'RUN_LOCAL_COMPLETE_CAMPAIGN','campaign':campaign.name,'note':'new scientific rerun; no pre-rerun hash manifest supplied'}
    rawmap=v1.raw_turn_map(data_root); rows=[]; measurement={}
    for model in models:
        gq=v1.gold_for_model(campaign,model,data_root)
        if 'primary' in scopes:
            v1.add_primary_like_rows(rows,campaign,data_root,model,'primary',gq,rawmap)
            measurement.setdefault('primary',{})[model]=_semantic_measurement_strict_general(rows,campaign,data_root,model,'primary')
        if 'closed_loop' in scopes:
            v1.add_primary_like_rows(rows,campaign,data_root,model,'closed_loop',gq,rawmap)
            measurement.setdefault('closed_loop',{})[model]=_semantic_measurement_strict_general(rows,campaign,data_root,model,'closed_loop')
        if 'utility' in scopes:
            start=len(rows); v1.add_utility_rows(rows,campaign,data_root,model,gq,rawmap)
            _utility_measurement_strict_general(rows[start:],campaign,data_root,model,gq)
            add_utility_harm_strict_general(rows[start:],gq)
        if 'fault' in scopes: v1.add_fault_rows(rows,campaign,model,gq,rawmap)
        if 'parser_robustness' in scopes:
            start=len(rows); v1.add_parser_rows(rows,campaign,model,gq,rawmap)
            add_parser_case_metadata_strict_general(rows[start:],campaign,model)
    add_strict_general_derived(rows)
    bert={'status':'DISABLED'}
    if args.bertscore!='off':
        bert=add_bertscore(rows,model_type=args.bertscore_model,device=args.bertscore_device,batch_size=args.bertscore_batch_size)
        if args.bertscore=='on' and bert.get('status')!='COMPLETE': raise SystemExit('BERTScore requested strictly but unavailable: '+json.dumps(bert))
    sm=summarize(rows)
    meta={'schema_version':'hf13-corrected-semantic-answer-eval-v3.2-strict-general','status':'COMPLETE','evaluator_version':EVALUATOR_VERSION,'answer_f1_gvr_separated':True,'campaign':campaign.name,'scopes':list(scopes),'models':list(models),'source_integrity':integrity,'bertscore':bert,'rows':len(rows),'evaluable_rows_strict_general':sum(bool(r.get('answer_evaluable_strict_general')) for r in rows),'excluded_rows_strict_general':sum(not bool(r.get('answer_evaluable_strict_general')) for r in rows)}
    derived={
        'primary_effects_strict_general':_primary_effects_strict_general(sm),
        'utility_harm_strict_general':_utility_harm_summary_strict_general(rows),
        'parser_downstream_strict_general':_parser_downstream_strict_general(rows),
    }
    (out/'strict_general_answer_eval_summary.json').write_text(json.dumps({'meta':meta,'summary':sm,'semantic_measurement':measurement,'derived':derived},ensure_ascii=False,indent=2,sort_keys=True)+'\n')
    write_csv(out/'strict_general_answer_eval_rows.csv',rows)
    changed=[r for r in rows if r.get('corrected_answer_f1_strict_general') is not None and abs(float(r['corrected_answer_f1_strict_general'])-float(r['original_answer_f1']))>1e-12]
    with (out/'strict_general_answer_eval_changed_cases.jsonl').open('w',encoding='utf-8') as f:
        for r in changed:
            keep={k:r.get(k) for k in ['experiment','model','condition','row_id','turn_id','family','raw_query','response','original_answer_f1','representation_equivalence_answer_f1','corrected_answer_f1_strict_general','original_gvr','corrected_gvr_strict_general','original_jaur','corrected_jaur_strict_general','corrected_obligation_scores_strict_general','corrected_obligation_notes_strict_general','corrected_obligations_strict_general']}; f.write(json.dumps(keep,ensure_ascii=False,sort_keys=True)+'\n')
    (out/'strict_general_answer_eval_report.md').write_text(report(sm,rows,meta),encoding='utf-8')
    print(json.dumps({'status':'COMPLETE','rows':len(rows),'evaluable_rows_strict_general':meta['evaluable_rows_strict_general'],'excluded_rows_strict_general':meta['excluded_rows_strict_general'],'output_dir':str(out),'bertscore':bert},indent=2))

if __name__=='__main__': main()
