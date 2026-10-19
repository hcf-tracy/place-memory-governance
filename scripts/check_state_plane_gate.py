#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from hf13.io import read_jsonl,write_json
from hf13.evaluator import state_plane_invariance,full_backend_invariance,observation_invariance
from hf13.artifact_registry import RuntimeArtifactRegistry
CONDS=('native','css','qasc','place')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model-root',required=True);ap.add_argument('--surface',choices=['primary','closed_loop'],required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
    mr=Path(a.model_root);r=mr/a.surface;rows={}
    for c in CONDS:
        p=r/c/'trace.jsonl'
        if not p.exists():raise SystemExit(f'missing state-plane trace: {p}')
        rows[c]=read_jsonl(p)
    regp=mr/'runtime_artifacts/runtime_artifact_registry.json';reg=RuntimeArtifactRegistry.load(regp) if regp.exists() else None
    inv=state_plane_invariance(rows,ROOT/'data/longitudinal',reg);full=full_backend_invariance(rows);obs=observation_invariance(rows)
    checks={}
    for name in ('native_qasc','css_place'):
        x=inv[name];checks[f'{name}_semantic_state_equal']=x['equality_rate']==1.0 and x['equal_count']==153 and not x['mismatches'] and not x['missing'] and x['post_event_state']['equal_count']==105 and x['pre_query_state']['equal_count']==48
    passed=all(checks.values())
    obj={'schema_version':'hf13-live-semantic-state-plane-hard-gate-v2','surface':a.surface,'passed':passed,'paper_table_eligible':passed,'checks':checks,'semantic_state_plane_invariance':inv,'full_backend_equality_diagnostic':full,'observation_ledger_equality_diagnostic':obs,'rule':'Paper aggregation requires semantic beta(X) equality at all 105 post-event + 48 pre-query checkpoint instances for Native↔QASC and CSS↔PLACE. Full-backend and branch-observation equality are diagnostic only.'}
    write_json(a.output,obj);print(json.dumps(obj,indent=2));raise SystemExit(0 if passed else 12)
if __name__=='__main__':main()
