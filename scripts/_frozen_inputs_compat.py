#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, hashlib, json, math, os
from collections import defaultdict
from pathlib import Path
from typing import Any

from hf13.answer_validation import canonical_references, token_prf, rouge_l_prf, chrf, sentence_bleu4, _best_over_refs, add_bertscore
from hf13.evaluator import runtime_gold, answer_score
from hf13.heldout_answer_rescore import equivalence_answer_score, forbidden_semantic_mentions
from hf13.io import read_jsonl

CONDITIONS = ("native", "css", "qasc", "place")
MODELS = ("7b", "3b")
DEFAULT_SCOPES = ("primary", "closed_loop", "utility", "fault", "parser_robustness")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(path: Path | None) -> dict | None:
    if not path or not path.exists():
        return None
    return json.loads(path.read_text(encoding='utf-8'))


def verify_source_manifest(root: Path, manifest: dict | None) -> dict:
    if not manifest:
        return {"status": "NOT_REQUESTED", "checked": 0}
    bad = []
    entries = manifest.get('files') or []
    for e in entries:
        p = root / e['path']
        if not p.exists():
            bad.append({"path": e['path'], "reason": "missing"})
            continue
        got = sha256_file(p)
        if got != e['sha256']:
            bad.append({"path": e['path'], "reason": "sha256_mismatch", "expected": e['sha256'], "actual": got})
    if bad:
        raise SystemExit('Frozen-source integrity failure:\n' + json.dumps(bad, indent=2))
    return {"status": "PASS", "checked": len(entries), "manifest_schema": manifest.get('schema_version')}


def mean(xs):
    vals = [float(x) for x in xs if x is not None]
    return sum(vals) / len(vals) if vals else None


def fmt(x):
    return 'N/A' if x is None else f'{x:.4f}'


def raw_turn_map(data_root: Path) -> dict[str, dict]:
    return {r['turn_id']: r for r in read_jsonl(data_root / 'trajectory_raw.jsonl')}

def raw_text(row: dict) -> str:
    return str(row.get('raw_utterance') or row.get('utterance') or row.get('text') or '')


def primary_trace_map(campaign: Path, model: str, condition: str) -> dict[str, dict]:
    p = campaign / model / 'primary' / condition / 'trace.jsonl'
    return {r['turn_id']: r for r in read_jsonl(p)}


def gold_for_model(campaign: Path, model: str, data_root: Path):
    registry = campaign / model / 'runtime_artifacts' / 'runtime_artifact_registry.json'
    _, gq = runtime_gold(data_root, registry)
    return gq


def supplement_metrics(row: dict, response: str, gold: dict):
    refs = canonical_references(gold)
    row['canonical_references'] = refs
    row['response'] = response
    tp = _best_over_refs(response, refs, token_prf)
    rp = _best_over_refs(response, refs, rouge_l_prf)
    row['token_precision'], row['token_recall'], row['token_f1'] = tp
    row['rouge_l_precision'], row['rouge_l_recall'], row['rouge_l_f1'] = rp
    row['chrf'] = _best_over_refs(response, refs, chrf)
    row['bleu4'] = _best_over_refs(response, refs, sentence_bleu4)


def score_answer(response: str, gold: dict, raw_query: str, stored_old: float, *, context: str):
    expected = gold.get('expected_facts') or []
    ineligible = gold.get('ineligible_facts') or []
    old_recomputed = answer_score(response, expected, ineligible, gold.get('expect_abstain', False), query=gold.get('query_semantics'), raw_query=raw_query)
    if abs(float(stored_old) - float(old_recomputed)) > 1e-9:
        raise RuntimeError(f'{context}: stored old Answer F1={stored_old} but original evaluator recomputes {old_recomputed}')
    hard, hard_detail = equivalence_answer_score(response, gold, raw_query, semantic_aliases=False)
    sens, sens_detail = equivalence_answer_score(response, gold, raw_query, semantic_aliases=True)
    new_gvr = 1.0 if forbidden_semantic_mentions(response, ineligible, raw_query, semantic_aliases=False) else 0.0
    sens_gvr = 1.0 if forbidden_semantic_mentions(response, ineligible, raw_query, semantic_aliases=True) else 0.0
    return {
        'original_answer_f1_recomputed': old_recomputed,
        'representation_equivalence_answer_f1': hard,
        'semantic_sensitivity_answer_f1': sens,
        'representation_equivalence_gvr': new_gvr,
        'semantic_sensitivity_gvr': sens_gvr,
        'answer_f1_delta': hard - float(stored_old),
        'representation_match_detail': hard_detail,
        'semantic_sensitivity_detail': sens_detail,
    }


def add_primary_like_rows(rows: list[dict], campaign: Path, data_root: Path, model: str, experiment: str, goldq: dict, rawmap: dict):
    for cond in CONDITIONS:
        tracep = campaign / model / experiment / cond / 'trace.jsonl'
        resultp = campaign / model / experiment / cond / ('evaluation.json' if experiment == 'primary' else 'result.json')
        trace = {r['turn_id']: r for r in read_jsonl(tracep)}
        result = json.loads(resultp.read_text(encoding='utf-8'))
        qmetrics = {q['turn_id']: q for q in result['queries']}
        for tid, m in qmetrics.items():
            tr = trace[tid]
            q = tr.get('query') or {}
            response = q.get('response') if 'response' in q else q.get('answer')
            response = '' if response is None else str(response)
            raw_query = str(tr.get('raw_utterance') or raw_text(rawmap.get(tid, {})))
            gold = goldq[tid]
            z = {
                'experiment': experiment, 'model': model, 'condition': cond,
                'row_id': f'{experiment}:{model}:{cond}:{tid}', 'turn_id': tid,
                'conversation_id': m.get('conversation_id'), 'family': m.get('family'),
                'original_answer_f1': float(m['answer_f1']), 'original_gvr': float(m.get('gvr', 0.0)),
                'original_jaur': float(m.get('jaur', 0.0)), 'query_state_f1': m.get('query_state_f1'),
                'needed_sra': m.get('needed_sra'), 'support_correct': m.get('support_correct'),
                'unauthorized_support_exposure': m.get('unauthorized_support_exposure'),
                'source_file': str(resultp.relative_to(campaign.parent.parent.parent)),
                'raw_query': raw_query,
            }
            z.update(score_answer(response, gold, raw_query, m['answer_f1'], context=z['row_id']))
            z['representation_rescored_jaur'] = 1.0 if (
                float(m.get('query_state_f1', 0.0)) >= .999999 and bool(m.get('support_correct')) and
                float(m.get('needed_sra', 0.0)) >= .999999 and
                float(m.get('unauthorized_support_exposure', 0.0)) == 0.0 and
                z['representation_equivalence_gvr'] == 0.0 and
                z['representation_equivalence_answer_f1'] >= .999999
            ) else 0.0
            z['semantic_sensitivity_jaur'] = 1.0 if (
                float(m.get('query_state_f1', 0.0)) >= .999999 and bool(m.get('support_correct')) and
                float(m.get('needed_sra', 0.0)) >= .999999 and
                float(m.get('unauthorized_support_exposure', 0.0)) == 0.0 and
                z['semantic_sensitivity_gvr'] == 0.0 and z['semantic_sensitivity_answer_f1'] >= .999999
            ) else 0.0
            supplement_metrics(z, response, gold)
            rows.append(z)


def add_utility_rows(rows: list[dict], campaign: Path, data_root: Path, model: str, goldq: dict, rawmap: dict):
    from hf13.utility import _conflict
    for cond in CONDITIONS:
        p = campaign / model / 'utility' / cond / 'result.json'
        obj = json.loads(p.read_text(encoding='utf-8'))
        for r in obj['rows']:
            tid = r['base_query_turn_id']; gold = goldq[tid]; response = str(r.get('answer') or '')
            raw_query = raw_text(rawmap[tid])
            z = {
                'experiment':'utility','model':model,'condition':cond,'row_id':r['case_id'],
                'turn_id':tid,'conversation_id':rawmap[tid].get('conversation_id'),'family':r.get('family'),
                'mode':r.get('mode'),'original_answer_f1':float(r['answer_f1']),'original_gvr':float(r.get('gvr',0.0)),
                'original_harm':float(r.get('harm',0.0)),'needed_sra':r.get('needed_sra'),
                'unauthorized_support_exposure':r.get('unauthorized_support_exposure'),'raw_query':raw_query,
                'source_file': str(p.relative_to(campaign.parent.parent.parent)),
                'original_jaur': None, 'representation_rescored_jaur': None, 'semantic_sensitivity_jaur': None,
            }
            z.update(score_answer(response,gold,raw_query,r['answer_f1'],context=f'utility:{model}:{cond}:{r["case_id"]}'))
            conflict = _conflict(gold.get('ineligible_facts') or [], r['case_id']) if r.get('mode') == 'conflict' else None
            z['representation_rescored_harm'] = 1.0 if (r.get('mode')=='conflict' and conflict and str(conflict).casefold() in response.casefold() and z['representation_equivalence_answer_f1'] < .999999) else 0.0
            supplement_metrics(z,response,gold); rows.append(z)


def add_fault_rows(rows: list[dict], campaign: Path, model: str, goldq: dict, rawmap: dict):
    for cond in CONDITIONS:
        p = campaign / model / 'fault' / cond / 'result.json'
        obj = json.loads(p.read_text(encoding='utf-8'))
        for r in obj['rows']:
            if r.get('post_answer_f1') is None:
                continue
            tid=r['base_query_turn_id']; gold=goldq[tid]; response=str(r.get('answer') or ''); raw_query=raw_text(rawmap[tid])
            z={'experiment':'fault','model':model,'condition':cond,'row_id':r['case_id'],'turn_id':tid,'conversation_id':rawmap[tid].get('conversation_id'),'family':r.get('family'),'fault_class':r.get('fault_class'),'fault_applicable':r.get('fault_applicable'),'original_answer_f1':float(r['post_answer_f1']),'original_gvr':float(r.get('gvr',0.0)),'original_jaur':None,'representation_rescored_jaur':None,'semantic_sensitivity_jaur':None,'unauthorized_support_exposure':r.get('unauthorized_support_exposure'),'raw_query':raw_query,'source_file':str(p.relative_to(campaign.parent.parent.parent))}
            z.update(score_answer(response,gold,raw_query,r['post_answer_f1'],context=f'fault:{model}:{cond}:{r["case_id"]}'))
            supplement_metrics(z,response,gold); rows.append(z)


def add_parser_rows(rows: list[dict], campaign: Path, model: str, goldq: dict, rawmap: dict):
    p=campaign/model/'parser_robustness'/'result.json'
    obj=json.loads(p.read_text(encoding='utf-8'))
    for case in obj.get('case_rows') or []:
        tid=case['query_turn_id']; gold=goldq[tid]; raw_query=raw_text(rawmap[tid])
        for cond, r in (case.get('conditions') or {}).items():
            response=str(r.get('response') or '')
            z={'experiment':'parser_robustness','model':model,'condition':cond,'row_id':f'{case["case_id"]}:{cond}','turn_id':tid,'conversation_id':rawmap[tid].get('conversation_id'),'family':case.get('family'),'original_answer_f1':float(r['answer_f1']),'original_gvr':float(r.get('gvr',0.0)),'original_jaur':float(r.get('jaur',0.0)),'query_state_f1':r.get('state_f1'),'authorized_support_f1':r.get('authorized_support_f1'),'unauthorized_support_exposure':r.get('exposure'),'raw_query':raw_query,'source_file':str(p.relative_to(campaign.parent.parent.parent))}
            z.update(score_answer(response,gold,raw_query,r['answer_f1'],context=f'parser:{model}:{case["case_id"]}:{cond}'))
            z['representation_rescored_jaur']=1.0 if (float(r.get('state_f1',0))>=.999999 and float(r.get('authorized_support_f1',0))>=.999999 and float(r.get('exposure',0))==0 and z['representation_equivalence_gvr']==0 and z['representation_equivalence_answer_f1']>=.999999) else 0.0
            z['semantic_sensitivity_jaur']=1.0 if (float(r.get('state_f1',0))>=.999999 and float(r.get('authorized_support_f1',0))>=.999999 and float(r.get('exposure',0))==0 and z['semantic_sensitivity_gvr']==0 and z['semantic_sensitivity_answer_f1']>=.999999) else 0.0
            supplement_metrics(z,response,gold); rows.append(z)


def grouped_summary(rows: list[dict]) -> dict:
    out={}
    experiments=sorted({r['experiment'] for r in rows})
    for exp in experiments:
        out[exp]={}
        for model in MODELS:
            mr=[r for r in rows if r['experiment']==exp and r['model']==model]
            if not mr: continue
            out[exp][model]={}
            for cond in sorted({r['condition'] for r in mr}):
                xs=[r for r in mr if r['condition']==cond]
                out[exp][model][cond]={
                    'n':len(xs),
                    'original_answer_f1':mean([x.get('original_answer_f1') for x in xs]),
                    'representation_equivalence_answer_f1':mean([x.get('representation_equivalence_answer_f1') for x in xs]),
                    'semantic_sensitivity_answer_f1':mean([x.get('semantic_sensitivity_answer_f1') for x in xs]),
                    'answer_f1_delta':mean([x.get('answer_f1_delta') for x in xs]),
                    'original_gvr':mean([x.get('original_gvr') for x in xs]),
                    'representation_equivalence_gvr':mean([x.get('representation_equivalence_gvr') for x in xs]),
                    'original_jaur':mean([x.get('original_jaur') for x in xs]),
                    'representation_rescored_jaur':mean([x.get('representation_rescored_jaur') for x in xs]),
                    'semantic_sensitivity_jaur':mean([x.get('semantic_sensitivity_jaur') for x in xs]),
                    'bertscore_f1':mean([x.get('bertscore_f1') for x in xs]),
                    'token_f1':mean([x.get('token_f1') for x in xs]),
                    'rouge_l_f1':mean([x.get('rouge_l_f1') for x in xs]),
                    'chrf':mean([x.get('chrf') for x in xs]),
                    'bleu4':mean([x.get('bleu4') for x in xs]),
                    'changed_answer_rows':sum(abs(float(x.get('answer_f1_delta') or 0.0))>1e-12 for x in xs),
                    'positive_answer_delta_rows':sum(float(x.get('answer_f1_delta') or 0.0)>1e-12 for x in xs),
                    'negative_answer_delta_rows':sum(float(x.get('answer_f1_delta') or 0.0)<-1e-12 for x in xs),
                }
    return out


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True,exist_ok=True)
    keys=[]
    preferred=['experiment','model','condition','row_id','turn_id','conversation_id','family','mode','fault_class','original_answer_f1','representation_equivalence_answer_f1','semantic_sensitivity_answer_f1','answer_f1_delta','original_gvr','representation_equivalence_gvr','original_jaur','representation_rescored_jaur','semantic_sensitivity_jaur','bertscore_precision','bertscore_recall','bertscore_f1','token_f1','rouge_l_f1','chrf','bleu4','raw_query','response']
    seen=set()
    for k in preferred:
        if any(k in r for r in rows): keys.append(k); seen.add(k)
    for r in rows:
        for k in r:
            if k not in seen and k not in {'canonical_references','representation_match_detail','semantic_sensitivity_detail'}:
                keys.append(k); seen.add(k)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for r in rows:
            rr={k:(json.dumps(r[k],ensure_ascii=False,sort_keys=True) if isinstance(r.get(k),(dict,list)) else r.get(k)) for k in keys};w.writerow(rr)


def report_md(summary: dict, meta: dict) -> str:
    L=['# HF13 Original-Experiment Answer Rescore v1','',
       'Read-only rescore over frozen outputs from the original full scientific rerun. No Qwen inference, Mem0 write, retrieval, CSS, QASC, state transition, or parser prediction is rerun. Original metrics remain preserved beside the rescored values.','',
       '**Representation-Equivalence v1 is byte-identical to the matcher used for the frozen naturalistic held-out rescore. BERTScore is supplementary and never changes hard Answer F1/JAUR.**','',
       f"Source campaign: `{meta['campaign']}`",f"Scopes: `{', '.join(meta['scopes'])}`",f"Rows rescored: **{meta['row_count']}**",f"Changed hard-answer rows: **{meta['changed_rows']}**",'']
    for exp,models in summary.items():
        L += [f'## {exp}', '']
        for model,conds in models.items():
            L += [f'### {model.upper()}', '', '| Condition | n | Original Answer F1 | Representation-Eq F1 | Δ | Original GVR | Eq GVR | Original JAUR | Eq JAUR | BERTScore | Changed |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
            order=[c for c in CONDITIONS if c in conds] + [c for c in ('TT','PT','TP','PP') if c in conds]
            for c in order:
                z=conds[c]
                L.append(f"| {c} | {z['n']} | {fmt(z['original_answer_f1'])} | {fmt(z['representation_equivalence_answer_f1'])} | {fmt(z['answer_f1_delta'])} | {fmt(z['original_gvr'])} | {fmt(z['representation_equivalence_gvr'])} | {fmt(z['original_jaur'])} | {fmt(z['representation_rescored_jaur'])} | {fmt(z['bertscore_f1'])} | {z['changed_answer_rows']} |")
            L += ['']
    L += ['## Scope notes','',
          '- `primary` and `closed_loop`: corrected Answer F1/GVR and the derived JAUR are emitted while state/support/exposure components remain frozen.',
          '- `utility`: corrected Answer F1/GVR are emitted; conflict harm is recomputed with the same original conflict construction and the corrected Answer F1.',
          '- `fault`: only the post-fault answer-side metrics are rescored; repair/state metrics are untouched.',
          '- `parser_robustness`: parser structured metrics (event/query/intent/target/scope F1) are untouched; only downstream generated-answer F1/GVR/JAUR are rescored.',
          '- `pairs`, `parser_smoke`, state-plane gates, and aggregate state metrics do not use the Answer F1 matcher and are not rescored.',
          '', '## Interpretation','',
          'This output can be used as a corrected answer-evaluation release only if the same frozen Representation-Equivalence v1 policy is accepted consistently. The original submitted metrics remain available for provenance and should not be deleted.']
    return '\n'.join(L)+'\n'


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--campaign-root',default='results/hf13_live/hf13_full_scientific_rerun_20260913_004919')
    ap.add_argument('--data-root',default='data/longitudinal')
    ap.add_argument('--scopes',default=','.join(DEFAULT_SCOPES),help='comma-separated: primary,closed_loop,utility,fault,parser_robustness')
    ap.add_argument('--models',default='7b,3b')
    ap.add_argument('--output-dir',default='results/original_answer_rescore_frozen_20260913_004919')
    ap.add_argument('--source-manifest',default='rescore_manifests/original_frozen_inputs_20260913_004919.json')
    ap.add_argument('--bertscore',choices=['off','auto','on'],default='auto')
    ap.add_argument('--bertscore-model',default='roberta-large')
    ap.add_argument('--bertscore-device',default=None)
    ap.add_argument('--bertscore-batch-size',type=int,default=16)
    args=ap.parse_args()
    root=Path.cwd(); campaign=(root/args.campaign_root).resolve(); data_root=(root/args.data_root).resolve(); out=(root/args.output_dir).resolve(); out.mkdir(parents=True,exist_ok=True)
    campaign_status=(campaign/'CAMPAIGN_STATUS').read_text(encoding='utf-8').strip()
    accepted={'COMPLETE_CLEAN','COMPLETE_WITH_MODEL_OPERATION_ERRORS','COMPLETE_WITH_RECOVERED_ERRORS'}
    if campaign_status not in accepted:
        raise SystemExit(f'Campaign is not a completed frozen run ({campaign_status}): {campaign}')
    scopes=tuple(x.strip() for x in args.scopes.split(',') if x.strip()); models=tuple(x.strip() for x in args.models.split(',') if x.strip())
    unknown=set(scopes)-set(DEFAULT_SCOPES)
    if unknown: raise SystemExit(f'Unknown scopes: {sorted(unknown)}')
    manifest=load_manifest((root/args.source_manifest).resolve() if args.source_manifest else None)
    integrity=verify_source_manifest(root,manifest)
    rawmap=raw_turn_map(data_root); rows=[]
    for model in models:
        gq=gold_for_model(campaign,model,data_root)
        if 'primary' in scopes: add_primary_like_rows(rows,campaign,data_root,model,'primary',gq,rawmap)
        if 'closed_loop' in scopes: add_primary_like_rows(rows,campaign,data_root,model,'closed_loop',gq,rawmap)
        if 'utility' in scopes: add_utility_rows(rows,campaign,data_root,model,gq,rawmap)
        if 'fault' in scopes: add_fault_rows(rows,campaign,model,gq,rawmap)
        if 'parser_robustness' in scopes: add_parser_rows(rows,campaign,model,gq,rawmap)
    bert={'status':'DISABLED'}
    if args.bertscore!='off':
        bert=add_bertscore(rows,model_type=args.bertscore_model,device=args.bertscore_device,batch_size=args.bertscore_batch_size)
        if args.bertscore=='on' and bert.get('status')!='COMPLETE':
            raise SystemExit('BERTScore requested strictly but unavailable: '+json.dumps(bert))
    summary=grouped_summary(rows)
    changed=[r for r in rows if abs(float(r.get('answer_f1_delta') or 0.0))>1e-12 or (r.get('original_jaur') is not None and r.get('representation_rescored_jaur') is not None and abs(float(r['representation_rescored_jaur'])-float(r['original_jaur']))>1e-12)]
    result={
        'schema_version':'hf13-original-experiment-answer-rescore-v1', 'status':'COMPLETE',
        'campaign':str(campaign), 'campaign_status':campaign_status, 'scopes':list(scopes), 'models':list(models),
        'row_count':len(rows),'changed_rows':len(changed),'source_integrity':integrity,'bertscore':bert,
        'execution_contract':{
            'post_hoc_read_only':True,'reruns_model_calls':False,'reruns_memory_writes':False,'reruns_retrieval':False,'reruns_css_qasc':False,'reruns_parser_predictions':False,'modifies_original_result_files':False,'original_metrics_preserved':True,'hard_correctness_uses_bertscore':False,
            'representation_equivalence_policy':'exact frozen Representation-Equivalence v1 copied byte-for-byte from the naturalistic-heldout rescore package; structured objects require all scalar leaves; broad semantic aliases are sensitivity-only',
            'old_score_reconstruction_gate':'Every row must reproduce the stored original Answer F1 with the original evaluator before it is eligible for rescore.'
        },
        'summary':summary,
        'not_rescored':{
            'pairs':'No Answer F1 matcher is used for the paired-world causal outcome metric.',
            'parser_smoke':'Structured parser metrics are separate from generated-answer scoring.',
            'state_plane_gate':'State-plane invariance is independent of answer text.',
            'aggregate_state_metrics':'Derived from frozen experiment metrics/state; no answer-text re-evaluation required here.'
        }
    }
    write_csv(out/'original_answer_rescore_rows.csv',rows)
    (out/'original_answer_rescore_summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    (out/'original_answer_rescore_changed_cases.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in changed),encoding='utf-8')
    (out/'original_answer_rescore_report.md').write_text(report_md(summary,{**result,'changed_rows':len(changed)}),encoding='utf-8')
    print(json.dumps({'status':'COMPLETE','rows':len(rows),'changed_rows':len(changed),'output_dir':str(out),'source_integrity':integrity,'bertscore':bert},indent=2))

# Internal frozen input and compatibility helpers. Not a public evaluator CLI.
