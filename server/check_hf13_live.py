#!/usr/bin/env python3
from __future__ import annotations
import argparse,importlib.metadata,importlib.util,json,platform,tempfile,sys
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from hf13.config_runtime import load_config

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--output');a=ap.parse_args();issues=[];warnings=[];steps={}
 py=platform.python_version();steps['python']=py
 if tuple(map(int,py.split('.')[:2]))!=(3,11):issues.append(f'Python 3.11 required; got {py}')
 try:mv=importlib.metadata.version('mem0ai')
 except importlib.metadata.PackageNotFoundError:mv=None
 steps['mem0ai']=mv
 if mv!='1.0.11':issues.append(f'mem0ai==1.0.11 required; got {mv}')
 for m in ('mem0','yaml','requests','pytest','matplotlib','networkx','ollama'):
  ok=bool(importlib.util.find_spec(m));steps[f'import_{m}']=ok
  if not ok:issues.append(f'missing Python module: {m}')
 cfg=load_config(a.config);mcfg=cfg.get('model') or {};model=mcfg.get('model_id');base=mcfg.get('base_url','http://127.0.0.1:11434').rstrip('/');embed=((((cfg.get('memory') or {}).get('mem0_config') or {}).get('embedder') or {}).get('config') or {}).get('model','nomic-embed-text')
 try:
  r=requests.get(base+'/api/tags',timeout=10);r.raise_for_status();names=[x.get('name') or x.get('model') for x in r.json().get('models',[])];steps['ollama_models']=names
  if not any(str(model).split(':')[0] in str(x) for x in names):issues.append(f'Ollama model not installed: {model}')
  if not any(str(embed).split(':')[0] in str(x) for x in names):issues.append(f'Ollama embedding model not installed: {embed}')
 except Exception as e:issues.append(f'Ollama tags failed at {base}: {e}')
 if not issues:
  try:
   from hf13.backend import make_store
   from hf13.css import CSSKernel
   from hf13.llm import ollama_chat,get_last_diagnostics
   txt=ollama_chat(base,model,[{'role':'user','content':'Reply with the single token OK.'}],temperature=0,max_tokens=16,timeout=int(mcfg.get('timeout_seconds',180)));steps['ollama_chat_nonempty']=bool(txt);steps['ollama_chat_diag']=get_last_diagnostics()
   with tempfile.TemporaryDirectory(prefix='hf13_live_preflight_') as td:
    s=make_store('mem0',cfg,result_root=td,namespace='hf13-preflight')
    try:
     d=s.native_add('My airport transfer time is 01:18 PM.',user_id='native_user',metadata={},role='user',max_retries=1);steps['native_write']=d.get('diagnostics')
     nst=(d.get('diagnostics') or {}).get('status')
     if nst=='model_operation_error':warnings.append('Native Mem0/Qwen preflight produced a model-action parse/action error; backend stayed usable and this is treated as an evaluated-system outcome, not infrastructure failure.')
     elif nst=='recovered':warnings.append('Native preflight required a verified bounded retry; backend remained usable.')
     elif nst!='clean':issues.append(f'Native preflight infrastructure status was not usable: {nst}')
     # CSS/PLACE reliability path: raw utterances are inert infer=False observations;
     # CSS alone decides governed state and realization.
     od=s.add_observation('The airline changed the preflight departure from 09:00 to 11:00.',user_id='css_user',observation_id='user:preflight_t1',role='user');steps['css_observation']=od.get('diagnostics')
     if not (od.get('diagnostics') or {}).get('verified'):issues.append('CSS infer=False observation preflight did not verify physical storage')
     k=CSSKernel();ev={'event_id':'preflight_e1','turn_id':'preflight_t1','target':{'entity':'preflight_trip','predicate':'departure_time'},'assertion':{'value':'11:00'},'intent':{'replace_current_value':True},'scope':{'trip':'preflight'},'provenance':{'source_type':'user_report'},'epistemic':{'status':'confirmed'}};k.apply(ev)
     rec=s.reconcile_managed(k.snapshot(),user_id='css_user');steps['css_reconcile']=rec
     if rec.get('status')!='clean':warnings.append(f"CSS deterministic realization required verified recovery: {rec.get('status')}")
     sr=s.search('What is the preflight departure?',user_id='css_user',limit=10);steps['css_native_retrieval_count']=len(sr)
     proj=s.project_managed(user_id='css_user');steps['managed_readback_count']=len(proj.get('facts') or []);steps['managed_mismatches']=proj.get('mismatches') or []
     if proj.get('mismatches'):issues.append('CSS physical certification mismatch')
     if not any(x.get('entity')=='preflight_trip' and x.get('value')=='11:00' for x in proj.get('facts') or []):issues.append('CSS managed readback missing expected fact')
    finally:s.close()
  except Exception as e:issues.append(f'live Mem0/CSS preflight failed: {type(e).__name__}: {e}')
 out={'schema_version':'hf13-runtime-hardened-live-preflight-v2','config':a.config,'model':model,'passed':not issues,'issues':issues,'warnings':warnings,'steps':steps}
 if a.output:Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps(out,indent=2));raise SystemExit(0 if out['passed'] else 1)
if __name__=='__main__':main()
