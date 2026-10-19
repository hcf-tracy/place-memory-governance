from __future__ import annotations
import argparse,json
from pathlib import Path
from .evaluator import evaluate
from .io import write_json
from .pairs import run_all as run_pairs,summarize as pair_summary
from .parser_diag import run_diagnostic
from .parser_robustness import run_parser_robustness,run_parser_contract_smoke
from .runner import run
from .config_runtime import load_config
from .artifact_registry import build_shared_registry
from .utility import run_utility,summarize as utility_summary
from .fault import run_fault,summarize as fault_summary
from .closed_loop import run_closed_loop
from .status import merge_statuses
from .factorial import run_factorial
from .evaluator import evaluate_rows

def cfg(path):return load_config(path) if path else {}
def _regarg(p):p.add_argument('--artifact-registry','--artifact-cache',dest='artifact_registry')
def _status(rows):
    return merge_statuses([r.get('status') or r.get('turn_status') for r in rows])
def main():
    ap=argparse.ArgumentParser(prog='hf13');sp=ap.add_subparsers(dest='cmd',required=True)
    for name in ('runtime-artifacts','artifacts'):
        p=sp.add_parser(name);p.add_argument('--config');p.add_argument('--data-root',default='data/longitudinal');p.add_argument('--mode',choices=['reference','llm'],default='reference');p.add_argument('--output',required=True)
    p=sp.add_parser('run');p.add_argument('--condition',choices=['native','css','qasc','place'],required=True);p.add_argument('--backend',choices=['reference','mem0'],default='reference');p.add_argument('--config');p.add_argument('--data-root',default='data/longitudinal');p.add_argument('--output',required=True);p.add_argument('--answer-mode',choices=['reference','llm'],default='reference');p.add_argument('--run-id',default='hf13');_regarg(p);p.add_argument('--closed-loop',action='store_true');p.add_argument('--progress-json')
    p=sp.add_parser('factorial-run');p.add_argument('--backend',choices=['reference','mem0'],default='reference');p.add_argument('--config');p.add_argument('--data-root',default='data/longitudinal');p.add_argument('--output-root',required=True);p.add_argument('--answer-mode',choices=['reference','llm'],default='reference');p.add_argument('--run-id',default='hf13-factorial');_regarg(p);p.add_argument('--closed-loop',action='store_true');p.add_argument('--progress-json');p.add_argument('--output')
    p=sp.add_parser('evaluate');p.add_argument('--result',required=True);p.add_argument('--data-root',default='data/longitudinal');p.add_argument('--output');_regarg(p)
    p=sp.add_parser('pairs');p.add_argument('--condition',choices=['native','css','qasc','place'],required=True);p.add_argument('--backend',choices=['reference','mem0'],default='reference');p.add_argument('--config');p.add_argument('--data-root',default='data/longitudinal');p.add_argument('--answer-mode',choices=['reference','llm'],default='reference');p.add_argument('--output');p.add_argument('--progress-jsonl')
    for cmd in ('utility','fault'):
        p=sp.add_parser(cmd);p.add_argument('--condition',choices=['native','css','qasc','place'],required=True);p.add_argument('--primary-result',required=True);p.add_argument('--config');p.add_argument('--data-root',default='data/longitudinal');p.add_argument('--answer-mode',choices=['reference','llm'],default='reference');p.add_argument('--output');p.add_argument('--progress-jsonl');_regarg(p)
        if cmd=='fault':p.add_argument('--backend',choices=['reference','mem0'],default='reference')
    p=sp.add_parser('closed-loop');p.add_argument('--condition',choices=['native','css','qasc','place'],required=True);p.add_argument('--backend',choices=['reference','mem0'],default='reference');p.add_argument('--config');p.add_argument('--data-root',default='data/longitudinal');p.add_argument('--answer-mode',choices=['reference','llm'],default='reference');_regarg(p);p.add_argument('--run-id',default='hf13-closed');p.add_argument('--trace-output',required=True);p.add_argument('--output');p.add_argument('--progress-json')
    p=sp.add_parser('parser-diagnostic');p.add_argument('--mode',choices=['deterministic','llm'],default='deterministic');p.add_argument('--config');p.add_argument('--data-root',default='data/longitudinal');_regarg(p);p.add_argument('--backend',choices=['reference','mem0'],default='reference');p.add_argument('--answer-mode',choices=['reference','llm'],default='reference');p.add_argument('--output');p.add_argument('--progress-json');p.add_argument('--checkpoint-dir')
    p=sp.add_parser('parser-robustness');p.add_argument('--parser-mode',choices=['deterministic','llm'],default='llm');p.add_argument('--config');p.add_argument('--data-root',default='data/longitudinal');_regarg(p);p.add_argument('--answer-mode',choices=['reference','llm'],default='llm');p.add_argument('--output');p.add_argument('--progress-json');p.add_argument('--checkpoint-dir');p.add_argument('--bootstrap-reps',type=int,default=4000)
    p=sp.add_parser('parser-smoke');p.add_argument('--parser-mode',choices=['deterministic','llm'],default='llm');p.add_argument('--config');p.add_argument('--data-root',default='data/longitudinal');_regarg(p);p.add_argument('--output')
    a=ap.parse_args();obj=None
    if a.cmd in {'runtime-artifacts','artifacts'}:obj=build_shared_registry(a.data_root,cfg=cfg(a.config),mode=a.mode,output=a.output).obj
    elif a.cmd=='run':
        rows=run(a.condition,data_root=a.data_root,backend=a.backend,cfg=cfg(a.config),result_path=a.output,answer_mode=a.answer_mode,run_id=a.run_id,artifact_registry=a.artifact_registry,closed_loop=a.closed_loop,progress_path=a.progress_json);obj={'rows':len(rows),'status':_status(rows)}
    elif a.cmd=='factorial-run':
        rows=run_factorial(data_root=a.data_root,backend=a.backend,cfg=cfg(a.config),answer_mode=a.answer_mode,run_id=a.run_id,artifact_registry=a.artifact_registry,closed_loop=a.closed_loop,output_root=a.output_root,progress_path=a.progress_json)
        evals={}
        for c,rs in rows.items():
            ev=evaluate_rows(rs,a.data_root,artifact_registry=a.artifact_registry)
            if a.closed_loop:
                ev['closed_loop']=True;ev['assistant_write_count']=0;ev['assistant_observation_ledger_count']=len(rs);ev['closed_loop_olr']=None;ev['closed_loop_olr_reason']='paired causal worlds not executed';ev['execution_architecture']='write_once_read_branch'
            evals[c]=ev
            d=Path(a.output_root)/c;write_json(d/('result.json' if a.closed_loop else 'evaluation.json'),ev)
        obj={'execution_architecture':'write_once_read_branch','closed_loop':bool(a.closed_loop),'conditions':{c:{'row_count':len(rows[c]),'status':evals[c]['status']} for c in rows},'status':merge_statuses([evals[c]['status'] for c in evals])}
    elif a.cmd=='evaluate':obj=evaluate(a.result,a.data_root,a.output,artifact_registry=a.artifact_registry)
    elif a.cmd=='pairs':
        rows=run_pairs(a.data_root,a.condition,backend=a.backend,cfg=cfg(a.config),answer_mode=a.answer_mode,progress_path=a.progress_jsonl);obj={'rows':rows,'summary':pair_summary(rows),'status':_status(rows)}
    elif a.cmd=='utility':
        rows=run_utility(a.primary_result,a.condition,data_root=a.data_root,answer_mode=a.answer_mode,cfg=cfg(a.config),progress_path=a.progress_jsonl,artifact_registry=a.artifact_registry);obj={'rows':rows,'summary':utility_summary(rows),'status':_status(rows)}
    elif a.cmd=='fault':
        rows=run_fault(a.primary_result,a.condition,data_root=a.data_root,answer_mode=a.answer_mode,cfg=cfg(a.config),progress_path=a.progress_jsonl,artifact_registry=a.artifact_registry,backend=a.backend);obj={'rows':rows,'summary':fault_summary(rows),'status':_status(rows)}
    elif a.cmd=='closed-loop':
        x=run_closed_loop(a.condition,data_root=a.data_root,backend=a.backend,cfg=cfg(a.config),answer_mode=a.answer_mode,run_id=a.run_id,artifact_registry=a.artifact_registry,result_path=a.trace_output,progress_path=a.progress_json);obj={'evaluation':x['evaluation'],'status':x['evaluation']['status'],'row_count':len(x['rows'])}
    elif a.cmd=='parser-diagnostic':obj=run_diagnostic(a.data_root,a.mode,cfg(a.config),progress_path=a.progress_json,artifact_registry=a.artifact_registry,backend=a.backend,answer_mode=a.answer_mode,checkpoint_dir=a.checkpoint_dir)
    elif a.cmd=='parser-smoke':obj=run_parser_contract_smoke(a.data_root,a.parser_mode,cfg(a.config),artifact_registry=a.artifact_registry)
    else:obj=run_parser_robustness(a.data_root,a.parser_mode,cfg(a.config),artifact_registry=a.artifact_registry,answer_mode=a.answer_mode,progress_path=a.progress_json,checkpoint_dir=a.checkpoint_dir,bootstrap_reps=a.bootstrap_reps)
    if getattr(a,'output',None) and a.cmd not in {'run','runtime-artifacts','artifacts','evaluate'}:write_json(a.output,obj)
    print(json.dumps(obj,ensure_ascii=False,indent=2,default=str))
if __name__=='__main__':main()
