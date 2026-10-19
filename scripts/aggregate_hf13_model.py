#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from hf13.io import read_json,write_json
from hf13.stats import aggregate
from hf13.status import merge_statuses
CONDS=('native','css','qasc','place')
def status_of(o): return o.get('status') or (o.get('evaluation') or {}).get('status') or 'UNKNOWN'
def evaluation_payload(o):
    """Accept both legacy wrapped and structural-factorial flat evaluations."""
    e=o.get('evaluation') if isinstance(o,dict) else None
    return e if isinstance(e,dict) else o
def load_inputs(model_root):
    """Load live model outputs without assuming a historical wrapper schema."""
    r=Path(model_root)
    pe={c:read_json(r/f'primary/{c}/evaluation.json') for c in CONDS}
    pr={c:read_json(r/f'pairs/{c}/result.json')['rows'] for c in CONDS}
    ur={c:read_json(r/f'utility/{c}/result.json')['rows'] for c in CONDS}
    fr={c:read_json(r/f'fault/{c}/result.json')['rows'] for c in CONDS}
    ce={c:evaluation_payload(read_json(r/f'closed_loop/{c}/result.json')) for c in CONDS}
    return pe,pr,ur,fr,ce
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model-root',required=True);ap.add_argument('--output',required=True);ap.add_argument('--bootstrap-reps',type=int,default=4000);a=ap.parse_args();r=Path(a.model_root)
 pe,pr,ur,fr,ce=load_inputs(r)
 obj=aggregate(pe,pr,ur,fr,ce,bootstrap_reps=a.bootstrap_reps)
 statuses=[]
 for c in CONDS:
  statuses += [status_of(pe[c]), status_of(read_json(r/f'pairs/{c}/result.json')), status_of(read_json(r/f'utility/{c}/result.json')), status_of(read_json(r/f'fault/{c}/result.json')), status_of(read_json(r/f'closed_loop/{c}/result.json'))]
 obj['component_statuses']=statuses
 obj['status']=merge_statuses(statuses)
 write_json(a.output,obj);print(json.dumps(obj,indent=2))
if __name__=='__main__':main()
