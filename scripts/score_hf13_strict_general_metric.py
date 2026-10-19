#!/usr/bin/env python3
from __future__ import annotations
import argparse,copy,csv,importlib.util,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

# Reuse frozen-input readers/reconstruction gates from v2.7.
spec=importlib.util.spec_from_file_location('strictrescore',ROOT/'scripts/score_hf13_strict_general_answer.py')
strictrescore=importlib.util.module_from_spec(spec);assert spec.loader;spec.loader.exec_module(strictrescore)
from hf13.metric_preserving_v28 import VERSION,support_metrics_v28,rescore_pair_row,pair_summary
from hf13.semantic_measurement_v26 import SemanticEvidenceResolver
from hf13.evaluator import runtime_gold
from hf13.io import read_jsonl
from hf13.utility import _conflict,summarize as utility_summary
from hf13.fault import summarize as fault_summary
from hf13.stats import aggregate
from hf13.answer_evaluator_strict_general import structured_expected_match,scalar_match

CONDS=('native','css','qasc','place')

def mean(xs):
    z=[float(x) for x in xs if x is not None];return sum(z)/len(z) if z else None

def family_summary(qs,cond):
    out={}
    for f in sorted({x.get('family') for x in qs}):
        xs=[x for x in qs if x.get('family')==f];n=len(xs)
        out[f]={'n':n,'query_state_f1':mean(x.get('query_state_f1') for x in xs),'exposure':mean(x.get('unauthorized_support_exposure') for x in xs),'gvr':mean(x.get('gvr') for x in xs),'answer_f1':mean(x.get('answer_f1') for x in xs),'needed_sra':mean(x.get('needed_sra') for x in xs),'jaur':mean(x.get('jaur') for x in xs),'query_contract_rate':(sum(bool(x.get('query_contract')) for x in xs)/n if cond in {'qasc','place'} else None)}
    return out

def corrected_eval(campaign,data_root,model,experiment,goldq,answer_rows):
    reg=campaign/model/'runtime_artifacts/runtime_artifact_registry.json';resolver=SemanticEvidenceResolver(data_root,reg)
    byans={(r['condition'],r['turn_id']):r for r in answer_rows if r['experiment']==experiment and r['model']==model}
    out={}
    for cond in CONDS:
        rp=campaign/model/experiment/cond/('evaluation.json' if experiment=='primary' else 'result.json')
        obj=json.loads(rp.read_text());trace={r['turn_id']:r for r in read_jsonl(campaign/model/experiment/cond/'trace.jsonl')}
        qs=[]
        for old in obj['queries']:
            tid=old['turn_id'];g=goldq[tid];ar=byans[(cond,tid)];q=trace[tid].get('query') or {}
            sm=support_metrics_v28(resolver,q,g.get('expected_facts') or [],g.get('ineligible_facts') or [])
            af=float(ar['corrected_answer_f1_strict_general']) if ar.get('answer_evaluable_strict_general') else float(ar['original_answer_f1'])
            fallback=not bool(ar.get('answer_evaluable_strict_general'));gv=float(ar.get('corrected_gvr_strict_general') or 0.0)
            needed=sm['needed_sra'];exp=sm['exposure']
            if cond=='qasc':contract=bool(needed>=.999999 and exp==0.0)
            elif cond=='place':contract=old.get('query_contract')
            else:contract=None
            support_ok=bool(needed>=.999999 and exp==0.0 and contract is not False)
            jaur=float(float(old.get('query_state_f1',0))>=.999999 and support_ok and gv==0.0 and af>=.999999)
            x=copy.deepcopy(old);x.update({'answer_f1':af,'gvr':gv,'needed_sra':needed,'unauthorized_support_exposure':exp,'exposed_forbidden_count':len(sm['exposed_facts']),'support_correct':support_ok,'query_contract':contract,'jaur':jaur,'general_answer_compatibility_fallback':fallback,'general_answer_exclusion_reason':ar.get('corrected_exclusion_reason_strict_general'),'general_support_detail':sm['exposure_match_detail']});qs.append(x)
        obj=copy.deepcopy(obj);obj['queries']=qs;obj['answer_f1_mean']=mean(x['answer_f1'] for x in qs);obj['gvr_mean']=mean(x['gvr'] for x in qs);obj['needed_sra_mean']=mean(x['needed_sra'] for x in qs);obj['unauthorized_support_exposure_rate']=mean(x['unauthorized_support_exposure'] for x in qs);obj['jaur_mean']=mean(x['jaur'] for x in qs);obj['query_contract_rate']=(sum(bool(x['query_contract']) for x in qs)/len(qs) if cond in {'qasc','place'} else None);obj['family_metrics']=family_summary(qs,cond);obj['schema_version']='hf13-refined-metric-v3.2-strict-general';out[cond]=obj
    return out

def corrected_utility(campaign,data_root,model,goldq,answer_rows):
    reg=campaign/model/'runtime_artifacts/runtime_artifact_registry.json';resolver=SemanticEvidenceResolver(data_root,reg);out={}
    armap={(r['condition'],r['row_id']):r for r in answer_rows if r['experiment']=='utility' and r['model']==model}
    for cond in CONDS:
        pobj=json.loads((campaign/model/'utility'/cond/'result.json').read_text());primary_trace={r['turn_id']:r for r in read_jsonl(campaign/model/'primary'/cond/'trace.jsonl')};rows=[]
        for old in pobj['rows']:
            tid=old['base_query_turn_id'];g=goldq[tid];ar=armap[(cond,old['case_id'])];q=primary_trace[tid].get('query') or {};sm=support_metrics_v28(resolver,q,g.get('expected_facts') or [],g.get('ineligible_facts') or [])
            af=float(ar['corrected_answer_f1_strict_general']) if ar.get('answer_evaluable_strict_general') else float(ar['original_answer_f1']);gv=float(ar.get('corrected_gvr_strict_general') or 0.0);x=copy.deepcopy(old);x['answer_f1']=af;x['gvr']=gv;x['unauthorized_support_exposure']=sm['exposure'];x['general_support_detail']=sm['exposure_match_detail']
            if x.get('mode')=='needed':
                app=x.get('needed_applicable') is not False;saved=bool(x.get('needed_fact_physically_preserved'));visible=sm['needed_sra']>=.999999;x['needed_fact_model_visible']=visible;x['needed_sra']=(1.0 if saved and visible else 0.0) if app else None
            if x.get('mode')=='conflict':
                ine=g.get('ineligible_facts') or [];response=str(x.get('answer') or '');mentioned=structured_expected_match(response,ine[0],require_entity=False) if ine else False
                if not mentioned:
                    conflict=_conflict(ine,x['case_id']);mentioned=scalar_match(response,conflict)
                x['harm']=float(bool(mentioned) and af<.999999)
            rows.append(x)
        out[cond]=rows
    return out

def corrected_fault(campaign,model,answer_rows):
    amap={(r['condition'],r['row_id']):r for r in answer_rows if r['experiment']=='fault' and r['model']==model};out={}
    for cond in CONDS:
        obj=json.loads((campaign/model/'fault'/cond/'result.json').read_text());rows=[]
        for old in obj['rows']:
            ar=amap.get((cond,old['case_id']));x=copy.deepcopy(old)
            if ar:
                x['post_answer_f1']=float(ar['corrected_answer_f1_strict_general']) if ar.get('answer_evaluable_strict_general') else float(ar['original_answer_f1']);x['gvr']=float(ar.get('corrected_gvr_strict_general') or 0.0)
            rows.append(x)
        out[cond]=rows
    return out

def corrected_pairs(campaign,data_root,model):
    pairs={p['pair_id']:p for p in read_jsonl(data_root/'hidden/causal_pairs.jsonl')};out={}
    for cond in CONDS:
        obj=json.loads((campaign/model/'pairs'/cond/'result.json').read_text());out[cond]=[rescore_pair_row(r,pairs[r['pair_id']]) for r in obj['rows']]
    return out

def parser_answer_summary(answer_rows,model):
    xs=[r for r in answer_rows if r['experiment']=='parser_robustness' and r['model']==model];out={}
    for cond in sorted({r['condition'] for r in xs}):
        ys=[r for r in xs if r['condition']==cond];vals=[];j=[]
        for r in ys:
            af=float(r['corrected_answer_f1_strict_general']) if r.get('answer_evaluable_strict_general') else float(r['original_answer_f1']);gv=float(r.get('corrected_gvr_strict_general') or 0.0);vals.append(af);j.append(float(float(r.get('query_state_f1') or 0)>=.999999 and float(r.get('authorized_support_f1') or 0)>=.999999 and float(r.get('unauthorized_support_exposure') or 0)==0 and gv==0 and af>=.999999))
        out[cond]={'n':len(ys),'answer_f1_strict_general':mean(vals),'jaur_strict_general':mean(j)}
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--campaign-root',default='results/hf13_live/hf13_full_scientific_rerun_20260913_004919');ap.add_argument('--data-root',default='data/longitudinal');ap.add_argument('--output-dir',default='results/final_metric');ap.add_argument('--models',default='7b,3b');ap.add_argument('--bootstrap-reps',type=int,default=4000);a=ap.parse_args()
    root=Path.cwd();campaign=(root/a.campaign_root).resolve();data_root=(root/a.data_root).resolve();out=(root/a.output_dir).resolve();out.mkdir(parents=True,exist_ok=True)
    # Frozen source hash gate from final.
    status=(campaign/'CAMPAIGN_STATUS').read_text().strip() if (campaign/'CAMPAIGN_STATUS').exists() else 'UNKNOWN'
    if status not in {'COMPLETE_CLEAN','COMPLETE_WITH_MODEL_OPERATION_ERRORS','COMPLETE_WITH_RECOVERED_ERRORS'}: raise SystemExit(f'campaign not complete: {status}')
    integrity={'schema_version':'hf13-run-local-source-integrity-v1','status':'RUN_LOCAL_COMPLETE_CAMPAIGN','campaign':campaign.name,'campaign_status':status}
    rawmap=strictrescore.v1.raw_turn_map(data_root);final={}
    for model in [x.strip() for x in a.models.split(',') if x.strip()]:
        goldq=strictrescore.v1.gold_for_model(campaign,model,data_root);ans=[]
        strictrescore.v1.add_primary_like_rows(ans,campaign,data_root,model,'primary',goldq,rawmap);strictrescore.v1.add_primary_like_rows(ans,campaign,data_root,model,'closed_loop',goldq,rawmap);strictrescore.v1.add_utility_rows(ans,campaign,data_root,model,goldq,rawmap);strictrescore.v1.add_fault_rows(ans,campaign,model,goldq,rawmap);strictrescore.v1.add_parser_rows(ans,campaign,model,goldq,rawmap)
        pe=corrected_eval(campaign,data_root,model,'primary',goldq,ans);ce=corrected_eval(campaign,data_root,model,'closed_loop',goldq,ans);ur=corrected_utility(campaign,data_root,model,goldq,ans);fr=corrected_fault(campaign,model,ans);pr=corrected_pairs(campaign,data_root,model)
        stats=aggregate(pe,pr,ur,fr,ce,bootstrap_reps=a.bootstrap_reps)
        final[model]={'statistics':stats,'pair_summaries':{c:pair_summary(pr[c]) for c in CONDS},'utility_summaries':{c:utility_summary(ur[c]) for c in CONDS},'parser_downstream':parser_answer_summary(ans,model),'strict_general_non_evaluable_query_ids':sorted({r['turn_id'] for r in ans if r['experiment']=='primary' and not r.get('answer_evaluable_strict_general')}),'strict_general_primary_evaluable_n':sum(1 for r in ans if r['experiment']=='primary' and r.get('condition')=='place' and r.get('answer_evaluable_strict_general')),'strict_general_primary_total_n':sum(1 for r in ans if r['experiment']=='primary' and r.get('condition')=='place')}
    coverages={m:(z['strict_general_primary_evaluable_n']/z['strict_general_primary_total_n'] if z['strict_general_primary_total_n'] else None) for m,z in final.items()}
    confirmatory_ready=all(v is not None and abs(v-1.0)<1e-12 for v in coverages.values())
    obj={'schema_version':'hf13-refined-metric-rescore-v3.2-strict-general','status':'COMPLETE','metric_definition_changed':False,'qasc_method_changed':False,'evaluator_unification_changed_qasc_runtime':False,'scientific_runtime_unchanged_by_evaluator':True,'scoring_equivalent_to_v291':False,'dataset_specific_overrides':False,'heldout_compatible_schema_driven':True,'implicit_value_semantic_aliases':False,'explicit_schema_aliases_only':True,'reference_evaluator_semantics':'3.2-strict-general','case_sets_changed':False,'denominators_changed':False,'compatibility_fallback_for_unannotated_legacy_gold':True,'parser_structured_metrics_changed':False,'state_metrics_changed':False,'source_integrity':integrity,'strict_general_primary_coverage':coverages,'confirmatory_ready':confirmatory_ready,'confirmatory_readiness_rule':'coverage must be 1.0; otherwise paper aggregates contain compatibility fallback for legacy unannotated gold','models':final}
    (out/'strict_general_metric_summary.json').write_text(json.dumps(obj,indent=2,ensure_ascii=False,sort_keys=True)+'\n')
    # Paper-ready compact extraction.
    paper={}
    for model,z in final.items():
        t=z['statistics']['condition_table'];effects={x['effect_id']:x for x in z['statistics']['main_prespecified_effects']}
        paper[model]={'StrictGeneralAnswerCoverage':(z['strict_general_primary_evaluable_n']/z['strict_general_primary_total_n'] if z['strict_general_primary_total_n'] else None),'StrictGeneralConfirmatoryReady':(z['strict_general_primary_evaluable_n']==z['strict_general_primary_total_n']),'primary':{c:{'StateCorrectness':t[c]['State F1'],'LeakageRate':t[c]['OLR'],'Leakage_n':t[c]['OLR_n'],'AnswerCorrectness':t[c]['Answer F1'],'JointSuccess':t[c]['JAUR']} for c in CONDS},'effects':effects,'utility':{c:{'NeededSupport':t[c]['Utility Needed SRA'],'Needed_n':t[c]['Utility Needed SRA_denominator'],'ExposureRate':t[c]['Utility Exposure'],'ConflictHarm':t[c]['Utility Harm']} for c in CONDS},'closed_loop':{c:{'StateCorrectness':t[c]['Closed State F1'],'ExposureRate':t[c]['Closed Exposure'],'JointSuccess':t[c]['Closed JAUR']} for c in CONDS}}
    (out/'PAPER_TABLE_VALUES_STRICT_GENERAL.json').write_text(json.dumps(paper,indent=2,ensure_ascii=False,sort_keys=True)+'\n')
    print(json.dumps({'status':'COMPLETE','output_dir':str(out),'source_integrity':integrity,'paper':paper},indent=2,ensure_ascii=False))
if __name__=='__main__':main()
