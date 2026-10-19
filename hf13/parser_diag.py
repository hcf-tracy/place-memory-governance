from __future__ import annotations
import copy,json,re,tempfile,shutil,time,hashlib,os
from pathlib import Path
from typing import Any
from .io import read_jsonl,write_jsonl,write_json
from .schema import validate_trusted_event,validate_trusted_query
from .llm import ollama_chat,get_last_diagnostics
from .status import from_diagnostics,merge_statuses
# Causal-pair target phrases are part of the unpublished benchmark, not this code.
# Private evaluation users may provide the original mapping as local data.
def _private_pair_schema():
    from pathlib import Path
    import json, os
    p=Path(os.environ.get('HF13_PRIVATE_PAIR_SCHEMA','data/longitudinal/parser_diagnostic/pair_target_schema.json'))
    if not p.is_file():
        raise FileNotFoundError(f'Private causal-pair mapping is not distributed: {p}')
    return json.loads(p.read_text(encoding='utf-8'))
from .governance_schema import canonicalize_legacy_intent, validate_runtime_governance

# Deliberately conservative deterministic parser. It is a robustness baseline,
# not an oracle: every one of the 302 turns is processed and false positives /
# false negatives are scored after all predictions exist.
def deterministic_event(utterance:str, turn_id:str):
    raise RuntimeError('Original benchmark-specific deterministic parser is not distributed in this code-only release; use the LLM parser mode with privately provisioned data.')


def deterministic_query(utterance:str, turn_id:str):
    raise RuntimeError('Original benchmark-specific deterministic parser is not distributed in this code-only release; use the LLM parser mode with privately provisioned data.')


def _extract_json(text:str):
    m=re.search(r'\{.*\}',text,re.S)
    if not m:raise ValueError('LLM parser returned no JSON object')
    return json.loads(m.group(0))

PARSER_LABELS={'none','event','query','both'}
PARSER_EVENT_KEYS={'event_id','turn_id','target','assertion','governance','scope','provenance','epistemic','temporal','depends_on'}
PARSER_QUERY_KEYS={'query_id','turn_id','target','query_intent','scope','as_of','selectors'}

def _strict_event(e,turn_id):
    if not isinstance(e,dict):raise ValueError('event must be object or null')
    extra=set(e)-PARSER_EVENT_KEYS
    if extra:raise ValueError(f'event has noncanonical keys: {sorted(extra)}')
    if e.get('turn_id')!=turn_id or e.get('event_id')!=f'parsed_{turn_id}':raise ValueError('event_id/turn_id must match the requested turn exactly')
    validate_trusted_event(e);return e

def _strict_query(q,turn_id):
    if not isinstance(q,dict):raise ValueError('query must be object or null')
    extra=set(q)-PARSER_QUERY_KEYS
    if extra:raise ValueError(f'query has noncanonical keys: {sorted(extra)}')
    if q.get('turn_id')!=turn_id or q.get('query_id')!=f'parsed_q_{turn_id}':raise ValueError('query_id/turn_id must match the requested turn exactly')
    validate_trusted_query(q);return q

def _validate_parser_envelope(obj,turn_id):
    if not isinstance(obj,dict) or set(obj)!={'label','event','query'}:raise ValueError('parser output must have exactly label,event,query')
    label=str(obj.get('label') or '')
    if label not in PARSER_LABELS:raise ValueError(f'invalid parser label: {label!r}')
    e=obj.get('event');q=obj.get('query');actual=_label(bool(e),bool(q))
    if actual!=label:raise ValueError(f'label/object inconsistency: label={label}, objects imply {actual}')
    if e is not None:e=_strict_event(e,turn_id)
    if q is not None:q=_strict_query(q,turn_id)
    return label,e,q

def llm_parse_turn(utterance:str,turn_id:str,cfg:dict):
    m=cfg.get('model') or {}
    sys=f"""Parse exactly one raw user turn into canonical USER-LEVEL semantics. Output one JSON object with EXACTLY these top-level keys: label, event, query.
label must be one of: none, event, query, both.
Consistency is mandatory: none=>event=null/query=null; event=>event object/query=null; query=>event=null/query object; both=>both objects.
A non-null event must use only canonical keys event_id, turn_id, target, assertion, governance, scope, provenance, epistemic, temporal, depends_on; event_id must be parsed_{turn_id} and turn_id must be {turn_id}. target requires entity and predicate. governance must contain exactly these fixed general fields: update,current_use,future_use,retention,data_access,action_authority,applicability,authority_preference,dependency_recompute_of,completion,collection_mutation. Do not emit bespoke per-turn flags, CSS operations, or lifecycle/resulting-state labels.
A non-null query must use only canonical keys query_id, turn_id, target, query_intent, scope, as_of, selectors; query_id must be parsed_q_{turn_id} and turn_id must be {turn_id}. Do not emit required/forbidden answers, eligible/ineligible facts, or evaluator gold.
Use only meaning stated by the user. Do not invent omitted fields. Output JSON only."""
    text=ollama_chat(m.get('base_url','http://127.0.0.1:11434'),m.get('model_id','qwen2.5:3b'),[{'role':'system','content':sys},{'role':'user','content':utterance}],temperature=0,max_tokens=768,timeout=int(m.get('timeout_seconds',180)),max_retries=2)
    diag=get_last_diagnostics()
    try:
        obj=_extract_json(text);label,e,q=_validate_parser_envelope(obj,turn_id)
        return e,q,{**diag,'parse_valid':True,'parse_error':None,'parsed_label':label}
    except Exception as ex:
        return None,None,{**diag,'parse_valid':False,'parse_error':f'{type(ex).__name__}: {ex}','parsed_label':None}

def _leaves(x,p=''):
    out=set()
    if isinstance(x,dict):
        for k,v in x.items():
            if k in {'event_id','query_id','turn_id','surface_forms'}:continue
            out|=_leaves(v,f'{p}.{k}' if p else k)
    elif isinstance(x,list):
        for v in x:out|=_leaves(v,p)
    else:out.add((p,json.dumps(x,sort_keys=True,ensure_ascii=False)))
    return out

def _f1(tp,fp,fn):
    return (2*tp/(2*tp+fp+fn)) if (2*tp+fp+fn) else None

def detection_metrics(pairs):
    tp=fp=fn=tn=0
    for pred,gold in pairs:
        p=pred is not None;g=gold is not None
        if p and g:tp+=1
        elif p and not g:fp+=1
        elif not p and g:fn+=1
        else:tn+=1
    return {'tp':tp,'fp':fp,'fn':fn,'tn':tn,'f1':_f1(tp,fp,fn) if (tp+fp+fn) else None}

def field_micro_metrics(items):
    pset=set();gset=set()
    for obs_id,pred,gold in items:
        pset|={(str(obs_id),path,val) for path,val in _leaves(pred or {})}
        gset|={(str(obs_id),path,val) for path,val in _leaves(gold or {})}
    tp=len(pset&gset);fp=len(pset-gset);fn=len(gset-pset)
    return {'tp':tp,'fp':fp,'fn':fn,'predicted_field_atoms':len(pset),'gold_field_atoms':len(gset),'f1':_f1(tp,fp,fn) if (pset or gset) else None}

def field_f1(pred,gold):
    return field_micro_metrics([('one',pred,gold)])['f1']

def _label(e,q):return 'both' if e and q else 'event' if e else 'query' if q else 'none'

def _pair_schema(text:str):
    low=str(text).casefold();matches=[(label,schema) for label,schema in _private_pair_schema().items() if label in low]
    if not matches:return None,None
    label,schema=max(matches,key=lambda z:len(z[0]));return label,copy.deepcopy(schema)

def _pair_mechanism(text:str)->str:
    u=str(text).casefold()
    if 'different scope' in u or 'current scope only' in u:return 'scope'
    if 'task-restricted historical' in u or 'current allowed' in u:return 'permission'
    if 'assistant previously guessed' in u or 'i now state the current' in u:return 'authority'
    if 'upstream change' in u or 'recomputed current' in u:return 'dependency'
    if 'hypothetical branch' in u or 'confirmed current' in u:return 'epistemic'
    return 'supersession'

def _pair_is_old(text:str)->bool:
    u=str(text).casefold();return any(x in u for x in ('earlier,','for a different scope','task-restricted historical','assistant previously guessed','before the upstream change','in a hypothetical branch'))

def _pair_value(text:str):
    raw=str(text).strip();u=raw.casefold()
    if 'task-restricted historical' in u:
        x=raw[u.index(' was ')+5:];return x.split(';',1)[0].strip().rstrip('.')
    for m in (' myself: ',' might have been ',' is now ',' is ',' was '):
        pos=u.rfind(m)
        if pos>=0:return raw[pos+len(m):].strip().rstrip('.').strip()
    return None

def deterministic_pair_event(utterance:str,case_id:str):
    label,sch=_pair_schema(utterance)
    if not sch:return None
    old=_pair_is_old(utterance);mech=_pair_mechanism(utterance);value=_pair_value(utterance)
    if value is None:return None
    predicate=sch['old_predicate'] if old else sch['current_predicate'];scope=copy.deepcopy(sch['old_scope'] if old else sch['current_scope'])
    intent={'record_for_planning':True};prov='user_report';ep='confirmed'
    if mech=='supersession' and not old:intent={'replace_current_value':True}
    elif mech=='permission' and old:intent={'allowed_for_current_task':False,'allowed_for_future_planning':False}
    elif mech=='authority' and old:prov='assistant_generated'
    elif mech=='dependency':intent={'record_for_planning':True} if old else {'replace_current_value':True}
    elif mech=='epistemic' and old:intent={'hypothetical_only':True};ep='hypothetical'
    e={'event_id':f'parsed_pair_{case_id}_{"old" if old else "current"}','turn_id':f'parsed_pair_{case_id}_{"old" if old else "current"}','target':{'entity':sch['entity'],'predicate':predicate},'assertion':{'value':value},'governance':canonicalize_legacy_intent(intent,{'epistemic':{'status':'confirmed'}}),'scope':scope,'provenance':{'source_type':prov},'epistemic':{'status':ep}}
    if mech=='dependency' and old:e['depends_on']=[{'entity':sch['entity'],'predicate':'upstream','source_sequence':1}]
    try:validate_trusted_event(e);return e
    except Exception:return None

def deterministic_pair_query(utterance:str,case_id:str):
    label,sch=_pair_schema(utterance)
    if not sch:return None
    q={'query_id':f'parsed_pair_q_{case_id}','turn_id':f'parsed_pair_q_{case_id}','target':{'entity':sch['entity'],'predicate':sch['query_predicate']},'query_intent':{'temporal_view':'current'},'scope':copy.deepcopy(sch['current_scope'])}
    try:validate_trusted_query(q);return q
    except Exception:return None

def _strict_pair_event(e,case_id,role_name):
    tid=f'parsed_pair_{case_id}_{role_name}'
    if not isinstance(e,dict):raise ValueError('pair event must be object')
    extra=set(e)-PARSER_EVENT_KEYS
    if extra:raise ValueError(f'pair event has noncanonical keys: {sorted(extra)}')
    if e.get('turn_id')!=tid or e.get('event_id')!=tid:raise ValueError('pair event id/turn id mismatch')
    validate_trusted_event(e);return e

def llm_parse_pair_event(utterance:str,case_id:str,cfg:dict):
    label,sch=_pair_schema(utterance)
    if not sch:return None,{'status':'clean','parse_valid':False,'parse_error':'target label not recognized'}
    old=_pair_is_old(utterance);role_name='old' if old else 'current';tid=f'parsed_pair_{case_id}_{role_name}';m=cfg.get('model') or {}
    prompt=(f'Parse one causal-pair user sentence into canonical USER-LEVEL semantics. The public ontology maps target phrase \"{label}\" to entity {sch["entity"]}. '
            f'Output exactly {{"label":"event","event":<canonical event>,"query":null}}. The event must have event_id={tid}, turn_id={tid}; use only canonical event keys and the fixed general governance object; do not invent bespoke intent flags. '
            'Never emit eligibility, gold answers, CSS operations, lifecycle/result states, or hidden pair labels. Output JSON only.')
    text=ollama_chat(m.get('base_url','http://127.0.0.1:11434'),m.get('model_id','qwen2.5:3b'),[{'role':'system','content':prompt},{'role':'user','content':utterance}],temperature=0,max_tokens=512,timeout=int(m.get('timeout_seconds',180)),max_retries=2);diag=get_last_diagnostics()
    try:
        o=_extract_json(text)
        if not isinstance(o,dict) or set(o)!={'label','event','query'} or o.get('label')!='event' or o.get('query') is not None:raise ValueError('pair event envelope invalid')
        e=_strict_pair_event(o.get('event'),case_id,role_name);return e,{**diag,'parse_valid':True,'parse_error':None}
    except Exception as ex:return None,{**diag,'parse_valid':False,'parse_error':f'{type(ex).__name__}: {ex}'}

def llm_parse_pair_query(utterance:str,case_id:str,cfg:dict):
    label,sch=_pair_schema(utterance)
    if not sch:return None,{'status':'clean','parse_valid':False,'parse_error':'target label not recognized'}
    tid=f'parsed_pair_q_{case_id}';m=cfg.get('model') or {}
    prompt=(f'Parse one causal-pair query into canonical USER-LEVEL query semantics. The public ontology maps target phrase \"{label}\" to entity {sch["entity"]}. '
            f'Output exactly {{"label":"query","event":null,"query":<canonical query>}}. The query must have query_id={tid}, turn_id={tid}; use only canonical query keys and user-stated meaning. '
            'Never output required/forbidden answers, eligibility, lifecycle gold, or evaluator labels. Output JSON only.')
    text=ollama_chat(m.get('base_url','http://127.0.0.1:11434'),m.get('model_id','qwen2.5:3b'),[{'role':'system','content':prompt},{'role':'user','content':utterance}],temperature=0,max_tokens=384,timeout=int(m.get('timeout_seconds',180)),max_retries=2);diag=get_last_diagnostics()
    try:
        o=_extract_json(text)
        if not isinstance(o,dict) or set(o)!={'label','event','query'} or o.get('label')!='query' or o.get('event') is not None:raise ValueError('pair query envelope invalid')
        q=o.get('query')
        if not isinstance(q,dict):raise ValueError('pair query must be object')
        extra=set(q)-PARSER_QUERY_KEYS
        if extra:raise ValueError(f'pair query has noncanonical keys: {sorted(extra)}')
        if q.get('query_id')!=tid or q.get('turn_id')!=tid:raise ValueError('pair query id/turn id mismatch')
        validate_trusted_query(q);return q,{**diag,'parse_valid':True,'parse_error':None}
    except Exception as ex:return None,{**diag,'parse_valid':False,'parse_error':f'{type(ex).__name__}: {ex}'}

PARSER_CHECKPOINT_SCHEMA='hf13-parser-durable-resume-v1'

def _sha256_file(path):
    p=Path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None

def _registry_fingerprint(registry):
    if registry is None:return None
    if isinstance(registry,(str,Path)):
        p=Path(registry);return _sha256_file(p) if p.exists() else hashlib.sha256(str(p).encode()).hexdigest()
    obj=getattr(registry,'obj',registry)
    try:raw=json.dumps(obj,sort_keys=True,ensure_ascii=False,default=str).encode()
    except Exception:raw=repr(obj).encode()
    return hashlib.sha256(raw).hexdigest()

def _checkpoint_identity(root,mode,cfg,registry):
    root=Path(root)
    package_manifest=Path(__file__).resolve().parents[1]/'MANIFEST.sha256'
    payload={'schema':PARSER_CHECKPOINT_SCHEMA,'mode':mode,'cfg':cfg or {},'registry_sha256':_registry_fingerprint(registry),'package_manifest_sha256':_sha256_file(package_manifest),'files':{}}
    for rel in ('trajectory_raw.jsonl','trusted_events.jsonl','trusted_queries.jsonl','parser_diagnostic/manifest.jsonl','parser_diagnostic/pair_manifest.jsonl'):
        payload['files'][rel]=_sha256_file(root/rel)
    raw=json.dumps(payload,sort_keys=True,ensure_ascii=False,default=str).encode();return {'identity':payload,'fingerprint':hashlib.sha256(raw).hexdigest()}

def _load_jsonl_tolerant(path):
    p=Path(path);out=[]
    if not p.exists():return out
    lines=p.read_text(encoding='utf-8').splitlines();truncated_tail=False
    for i,line in enumerate(lines):
        if not line.strip():continue
        try:out.append(json.loads(line))
        except Exception:
            if i==len(lines)-1:truncated_tail=True;break
            raise
    if truncated_tail:write_jsonl(p,out)  # discard only an incomplete final fsync record
    return out

def _append_durable(path,row):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a',encoding='utf-8') as f:
        f.write(json.dumps(row,ensure_ascii=False,sort_keys=True,default=str)+'\n');f.flush();os.fsync(f.fileno())

def _prepare_checkpoint_dir(checkpoint_dir,root,mode,cfg,registry):
    if checkpoint_dir is None:return None
    d=Path(checkpoint_dir);d.mkdir(parents=True,exist_ok=True);meta=d/'checkpoint_meta.json';ident=_checkpoint_identity(root,mode,cfg,registry)
    if meta.exists():
        old=json.loads(meta.read_text())
        if old.get('fingerprint')!=ident['fingerprint']:
            raise RuntimeError(f'parser resume checkpoint identity mismatch at {d}; refuse to mix model/data/package configurations')
    else:write_json(meta,{'schema_version':PARSER_CHECKPOINT_SCHEMA,**ident})
    return d

def _load_turn_prediction_checkpoint(path,manifest):
    expected={x['turn_id'] for x in manifest};rows=_load_jsonl_tolerant(path);by={};kept=[]
    for r in rows:
        tid=r.get('turn_id')
        if tid not in expected:raise RuntimeError(f'parser checkpoint contains unexpected turn_id: {tid}')
        # A provider/infrastructure failure is not a completed parser observation.
        # Resume must retry it instead of permanently caching the failure.
        if str((r.get('diagnostic') or {}).get('status') or 'clean')=='failed':continue
        by[tid]=r;kept.append(r)
    if path and len(kept)!=len(rows):write_jsonl(path,kept)
    return by

def _load_pair_prediction_checkpoint(path,manifest):
    expected={x['case_id'] for x in manifest};rows=_load_jsonl_tolerant(path);by={};kept=[]
    for r in rows:
        cid=r.get('case_id')
        if cid not in expected:raise RuntimeError(f'parser-pair checkpoint contains unexpected case_id: {cid}')
        if any(str((d or {}).get('status') or 'clean')=='failed' for d in (r.get('diagnostics') or [])):continue
        by[cid]=r;kept.append(r)
    if path and len(kept)!=len(rows):write_jsonl(path,kept)
    return by

def run_pair_parser_extension(data_root='data/longitudinal',mode='deterministic',cfg=None,backend='reference',answer_mode='reference',progress_callback=None,checkpoint_dir=None):
    """48-pair parser extension used only to estimate parser-induced ΔOLR.

    Raw pair utterances are parsed before hidden pair events/query/gold are loaded.
    Durable checkpoints contain parser outputs only, never hidden targets. Pair
    downstream cases are also checkpointed so a crash resumes at the first
    unfinished case rather than repeating completed model calls.
    """
    from .pairs import run_pair,summarize as pair_summary
    root=Path(data_root);manifest=read_jsonl(root/'parser_diagnostic/pair_manifest.jsonl');cp=Path(checkpoint_dir)/'pair_extension' if checkpoint_dir else None
    pred_path=(cp/'pair_predictions.jsonl') if cp else None;down_path=(cp/'pair_downstream.jsonl') if cp else None
    pred_by=_load_pair_prediction_checkpoint(pred_path,manifest) if pred_path else {};resumed_predictions=len(pred_by)
    for x in manifest:
        case=x['case_id']
        if case in pred_by:
            if progress_callback:progress_callback('pair_parse',len(pred_by),len(manifest),case)
            continue
        a0=x['world_a_sentences'][0];b0=x['world_b_sentences'][0];cur=x['world_a_sentences'][1]
        if x['world_b_sentences'][1]!=cur:raise ValueError(f'{case}: current pair sentence differs across worlds')
        if mode=='deterministic':
            ea=deterministic_pair_event(a0,case);eb=deterministic_pair_event(b0,case);ec=deterministic_pair_event(cur,case);q=deterministic_pair_query(x['query'],case);ds=[{'status':'clean','parse_valid':True}]*4
        else:
            try:ea,d1=llm_parse_pair_event(a0,case,cfg or {})
            except Exception as ex:ea=None;d1={'status':'failed','parse_valid':False,'parse_error':f'{type(ex).__name__}: {ex}'}
            try:eb,d2=llm_parse_pair_event(b0,case,cfg or {})
            except Exception as ex:eb=None;d2={'status':'failed','parse_valid':False,'parse_error':f'{type(ex).__name__}: {ex}'}
            try:ec,d3=llm_parse_pair_event(cur,case,cfg or {})
            except Exception as ex:ec=None;d3={'status':'failed','parse_valid':False,'parse_error':f'{type(ex).__name__}: {ex}'}
            try:q,d4=llm_parse_pair_query(x['query'],case,cfg or {})
            except Exception as ex:q=None;d4={'status':'failed','parse_valid':False,'parse_error':f'{type(ex).__name__}: {ex}'}
            ds=[d1,d2,d3,d4]
        row={'case_id':case,'event_a_old':ea,'event_b_old':eb,'event_current':ec,'query':q,'diagnostics':ds}
        pred_by[case]=row
        if pred_path:_append_durable(pred_path,row)
        if progress_callback:progress_callback('pair_parse',len(pred_by),len(manifest),case)
    pred=[pred_by[x['case_id']] for x in manifest];diags=[{'case_id':r['case_id'],**d} for r in pred for d in (r.get('diagnostics') or [])]

    # Hidden targets are opened only after every pair prediction is durable.
    hidden=read_jsonl(root/'hidden/causal_pairs.jsonl');hb={x['opaque_case_id']:x for x in hidden};pb={x['case_id']:x for x in pred}
    saved_down=_load_jsonl_tolerant(down_path) if down_path else [];completed={};kept_down=[]
    for r in saved_down:
        if (r.get('trusted_row') or {}).get('status')=='FAILED' or (r.get('parsed_row') or {}).get('status')=='FAILED':continue
        completed[r['case_id']]=r;kept_down.append(r)
    if down_path and len(kept_down)!=len(saved_down):write_jsonl(down_path,kept_down)
    resumed_downstream=len(completed)
    for x in manifest:
        case=x['case_id']
        if case in completed:
            if progress_callback:progress_callback('pair_downstream',len(completed),len(manifest),case)
            continue
        h=hb[case];pp=pb[case];trusted_row=run_pair(h,'place',backend=backend,cfg=cfg,answer_mode=answer_mode,result_root='results/hf13_pair_parser_trusted')
        z=copy.deepcopy(h);z['events_a']=[pp['event_a_old'],copy.deepcopy(pp['event_current'])];z['events_b']=[pp['event_b_old'],copy.deepcopy(pp['event_current'])];z['query_semantics']=copy.deepcopy(pp['query']);z['parsed_mode']=True;z['event_sentence_count']=2
        parsed_row=run_pair(z,'place',backend=backend,cfg=cfg,answer_mode=answer_mode,result_root='results/hf13_pair_parser_parsed')
        rr={'case_id':case,'trusted_row':trusted_row,'parsed_row':parsed_row};completed[case]=rr
        if down_path:_append_durable(down_path,rr)
        if progress_callback:progress_callback('pair_downstream',len(completed),len(manifest),case)
    trusted_rows=[completed[x['case_id']]['trusted_row'] for x in manifest];parsed_rows=[completed[x['case_id']]['parsed_row'] for x in manifest]

    event_items=[];query_items=[];event_det_pairs=[];query_det_pairs=[]
    for case in [x['case_id'] for x in manifest]:
        h=hb[case];pp=pb[case]
        triples=[('a_old',pp['event_a_old'],h['events_a'][0]),('b_old',pp['event_b_old'],h['events_b'][0]),('current',pp['event_current'],h['events_a'][1])]
        for suffix,ppred,ggold in triples:event_items.append((f'{case}:{suffix}',ppred,ggold));event_det_pairs.append((ppred,ggold))
        query_items.append((case,pp['query'],h['query_semantics']));query_det_pairs.append((pp['query'],h['query_semantics']))
    tb={x['pair_id']:x for x in trusted_rows};pr={x['pair_id']:x for x in parsed_rows};common=[k for k in tb if tb[k]['eligible'] and pr[k]['eligible']]
    trusted_common=sum(tb[k]['olr_event'] for k in common)/len(common) if common else None;parsed_common=sum(pr[k]['olr_event'] for k in common)/len(common) if common else None;delta=(parsed_common-trusted_common) if common else None
    status=merge_statuses([from_diagnostics(diags)]+[x.get('status') for x in trusted_rows+parsed_rows]);em=field_micro_metrics(event_items);qm=field_micro_metrics(query_items);ed=detection_metrics(event_det_pairs);qd=detection_metrics(query_det_pairs);invalid=sum(d.get('parse_valid') is False for d in diags)
    return {'mode':mode,'pair_case_count':len(manifest),'predictions_completed_before_hidden_targets':len(pred)==len(manifest),'event_detection_f1':ed['f1'],'query_detection_f1':qd['f1'],'event_detection_counts':ed,'query_detection_counts':qd,'event_field_f1':em['f1'],'query_field_f1':qm['f1'],'event_field_counts':em,'query_field_counts':qm,'invalid_parse_count':invalid,'trusted_summary':pair_summary(trusted_rows),'parsed_summary':pair_summary(parsed_rows),'common_eligible_n':len(common),'trusted_common_olr':trusted_common,'parsed_common_olr':parsed_common,'delta_olr':delta,'eligibility_rate_parsed':sum(x['eligible'] for x in parsed_rows)/len(parsed_rows),'state_mediated_n_parsed':sum(x['state_mediated_divergence'] for x in parsed_rows),'equal_but_wrong_n_parsed':sum(x['equal_but_wrong'] for x in parsed_rows),'provider_diagnostics':diags,'status':status,'parsed_rows':parsed_rows,'resume':{'resumed_pair_predictions':resumed_predictions,'new_pair_predictions':len(manifest)-resumed_predictions,'resumed_pair_downstream_cases':resumed_downstream,'new_pair_downstream_cases':len(manifest)-resumed_downstream}}

def _downstream(root:Path,events,queries,artifact_registry=None,backend="reference",answer_mode="reference",cfg=None,checkpoint_dir=None):
    """Run parser-induced downstream diagnostics with stage-level durable resume.

    Only scientifically complete stage checkpoints are reusable. Provider-failed
    stages are rerun on resume. Parser-induced absence of a Fault target is a
    measured inapplicability; standalone trusted Fault remains strict.
    """
    from .runner import run
    from .evaluator import evaluate_rows
    from .utility import run_utility,summarize as util_summary
    from .fault import run_fault,summarize as fault_summary
    from .closed_loop import run_closed_loop
    cp=Path(checkpoint_dir)/'downstream' if checkpoint_dir else None
    if cp:cp.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='hf13_parser_') as td:
        op=Path(td)/'operational';op.mkdir(parents=True)
        for name in ('trajectory_raw.jsonl','conversation_boundaries.json','runtime_bindings.jsonl','turn_roles.jsonl'):
            shutil.copy2(root/name,op/name)
        write_jsonl(op/'trusted_events.jsonl',events);write_jsonl(op/'trusted_queries.jsonl',queries)

        prow=(cp/'primary_rows.jsonl') if cp else None;pev=(cp/'primary_evaluation.json') if cp else None;reuse_primary=False
        if prow and pev and prow.exists() and pev.exists():
            cand=json.loads(pev.read_text());reuse_primary=cand.get('status')!='FAILED'
            if reuse_primary:rows=read_jsonl(prow);ev=cand
        if not reuse_primary:
            rows=run('place',data_root=op,backend=backend,cfg=cfg,answer_mode=answer_mode,run_id='parser-primary',artifact_registry=artifact_registry);ev=evaluate_rows(rows,root,artifact_registry=artifact_registry)
            if cp:write_jsonl(prow,rows);write_json(pev,ev)

        up=(cp/'utility.json') if cp else None;uwrap=None
        if up and up.exists():
            cand=json.loads(up.read_text());uwrap=cand if cand.get('status')!='FAILED' else None
        if uwrap is None:
            urows=run_utility(rows,'place',data_root=root,answer_mode=answer_mode,cfg=cfg,artifact_registry=artifact_registry);util=util_summary(urows);ustatus=merge_statuses([r.get('status') for r in urows]);uwrap={'summary':util,'status':ustatus}
            if up:write_json(up,uwrap)
        util=uwrap['summary'];ustatus=uwrap['status']

        fp=(cp/'fault.json') if cp else None;fwrap=None
        if fp and fp.exists():
            cand=json.loads(fp.read_text());fwrap=cand if cand.get('status')!='FAILED' else None
        if fwrap is None:
            frows=run_fault(rows,'place',data_root=root,runtime_data_root=op,answer_mode=answer_mode,cfg=cfg,artifact_registry=artifact_registry,missing_target_policy='record_ineligible');fault=fault_summary(frows);fstatus=merge_statuses([r.get('status') for r in frows]);fwrap={'summary':fault,'status':fstatus}
            if fp:write_json(fp,fwrap)
        fault=fwrap['summary'];fstatus=fwrap['status']

        cp_cl=(cp/'closed_loop_evaluation.json') if cp else None;clev=None
        if cp_cl and cp_cl.exists():
            cand=json.loads(cp_cl.read_text());clev=cand if cand.get('status')!='FAILED' else None
        if clev is None:
            cl=run_closed_loop('place',data_root=op,backend=backend,cfg=cfg,answer_mode=answer_mode,run_id='parser-closed',artifact_registry=artifact_registry,evaluator_root=root);clev=evaluate_rows(cl['rows'],root,artifact_registry=artifact_registry)
            if cp_cl:write_json(cp_cl,clev)
    stage_statuses={'primary':ev.get('status'),'utility':ustatus,'fault':fstatus,'closed_loop':clev.get('status')};status=merge_statuses(stage_statuses.values())
    return {'primary':ev,'utility':util,'fault':fault,'closed_loop':clev,'stage_statuses':stage_statuses,'status':status}

def _trusted_baseline(root,artifact_registry,backend,answer_mode,cfg):
    from .runner import run
    from .evaluator import evaluate_rows
    rows=run('place',data_root=root,backend=backend,cfg=cfg,answer_mode=answer_mode,run_id='parser-trusted-baseline',artifact_registry=artifact_registry)
    return evaluate_rows(rows,root,artifact_registry=artifact_registry)

def _deltas(parsed,trusted,pair_extension=None):
    p=parsed['primary'];t=trusted;px=pair_extension or {}
    return {
      'delta_state_f1':p['state_f1_mean']-t['state_f1_mean'],
      'delta_exposure':p['unauthorized_support_exposure_rate']-t['unauthorized_support_exposure_rate'],
      'delta_gvr':p['gvr_mean']-t['gvr_mean'],
      'delta_answer_f1':p['answer_f1_mean']-t['answer_f1_mean'],
      'delta_jaur':p['jaur_mean']-t['jaur_mean'],
      'delta_olr':px.get('delta_olr'),
      'olr_common_eligible_n':px.get('common_eligible_n'),
      'olr_parsed_eligibility_rate':px.get('eligibility_rate_parsed'),
      'olr_rule':'ΔOLR uses only the 48 causal pair IDs eligible under both trusted and parsed semantics; parser-induced eligibility loss is reported separately.'
    }

def run_diagnostic(data_root='data/longitudinal',mode='deterministic',cfg=None,progress_path=None,artifact_registry=None,artifact_cache=None,backend='reference',answer_mode='reference',trusted_baseline=None,checkpoint_dir=None):
    registry=artifact_registry if artifact_registry is not None else artifact_cache;root=Path(data_root);manifest=read_jsonl(root/'parser_diagnostic/manifest.jsonl');pp=Path(progress_path) if progress_path else None;start=time.time();cp=_prepare_checkpoint_dir(checkpoint_dir,root,mode,cfg or {},registry)
    pred_path=(cp/'turn_predictions.jsonl') if cp else None;pred_by=_load_turn_prediction_checkpoint(pred_path,manifest) if pred_path else {};resumed_predictions=len(pred_by)
    def prog(stage,done,current=None,status=None,**extra):
        if not pp:return
        elapsed=max(.001,time.time()-start);new_done=max(0,done-resumed_predictions);rate=(new_done/elapsed if new_done else None);eta=((len(manifest)-done)/rate if rate and done<len(manifest) else 0 if done>=len(manifest) else None)
        write_json(pp,{'schema_version':'hf13-parser-progress-resumable-v2','mode':mode,'stage':stage,'done':done,'total':len(manifest),'current_turn_id':current,'rate_new_items_s':rate,'eta_s':eta,'status':status,'resumed_predictions':resumed_predictions,**extra})
    prog('parsing',len(pred_by))
    for x in manifest:
        tid=x['turn_id']
        if tid in pred_by:
            prog('parsing',len(pred_by),tid);continue
        if mode=='deterministic':e=deterministic_event(x['utterance'],tid);q=deterministic_query(x['utterance'],tid);d={'status':'clean','parse_valid':True,'parse_error':None}
        else:
            try:e,q,d=llm_parse_turn(x['utterance'],tid,cfg or {})
            except Exception as ex:d={'status':'failed','errors':[f'{type(ex).__name__}: {ex}'],'parse_valid':False,'parse_error':f'{type(ex).__name__}: {ex}'};e=q=None
        row={'turn_id':tid,'event':e,'query':q,'label':_label(e,q),'diagnostic':d};pred_by[tid]=row
        if pred_path:_append_durable(pred_path,row)
        prog('parsing',len(pred_by),tid)
    pred=[{k:v for k,v in pred_by[x['turn_id']].items() if k!='diagnostic'} for x in manifest];diags=[pred_by[x['turn_id']].get('diagnostic') or {'status':'clean'} for x in manifest]

    # Hidden parser gold is loaded only after all 302 predictions are durably complete.
    gold={x['turn_id']:x for x in read_jsonl(root/'hidden/parser_gold.jsonl')};pb={x['turn_id']:x for x in pred};event_items=[(t,pb[t]['event'],g.get('event')) for t,g in gold.items()];query_items=[(t,pb[t]['query'],g.get('query')) for t,g in gold.items()];event_det=detection_metrics([(pb[t]['event'],g.get('event')) for t,g in gold.items()]);query_det=detection_metrics([(pb[t]['query'],g.get('query')) for t,g in gold.items()]);event_field=field_micro_metrics(event_items);query_field=field_micro_metrics(query_items);label_acc=sum(pb[t]['label']==g['label'] for t,g in gold.items())/len(gold);pe=[x['event'] for x in pred if x['event']];pq=[x['query'] for x in pred if x['query']];invalid_count=sum(d.get('parse_valid') is False for d in diags)
    prog('downstream',len(manifest),resumed_stage=bool(cp and (cp/'downstream').exists()));downstream=_downstream(root,pe,pq,artifact_registry=registry,backend=backend,answer_mode=answer_mode,cfg=cfg,checkpoint_dir=cp)
    if trusted_baseline is None:trusted=_trusted_baseline(root,registry,backend,answer_mode,cfg)
    elif isinstance(trusted_baseline,(str,Path)):trusted=json.loads(Path(trusted_baseline).read_text())
    else:trusted=trusted_baseline
    prog('pair_parser_extension',len(manifest),current='48 causal pairs')
    pair_started=time.time()
    def pair_progress(stage,done,total,current):
        if not pp:return
        elapsed=max(.001,time.time()-pair_started);rate=done/elapsed if done else None;eta=((total-done)/rate if rate and done<total else 0 if done>=total else None)
        write_json(pp,{'schema_version':'hf13-parser-progress-resumable-v2','mode':mode,'stage':stage,'done':len(manifest),'total':len(manifest),'pair_done':done,'pair_total':total,'current_pair_case':current,'pair_rate_cases_s':rate,'pair_eta_s':eta,'status':None,'resumed_predictions':resumed_predictions})
    pair_ext=run_pair_parser_extension(root,mode,cfg=cfg,backend=backend,answer_mode=answer_mode,progress_callback=pair_progress,checkpoint_dir=cp)
    deltas=_deltas(downstream,trusted,pair_ext);all_diags=diags+list(pair_ext.get('provider_diagnostics') or []);status=merge_statuses([from_diagnostics(all_diags),downstream.get('status'),pair_ext.get('status'),trusted.get('status')]);prog('complete',len(manifest),status=status,pair_done=48,pair_total=48)
    return {'mode':mode,'primary_table_eligible':False,'parser_turn_count':len(manifest),'all_302_turns_processed':len(pred)==302,'targets_loaded_only_after_parser_execution':True,'event_detection_f1':event_det['f1'],'query_detection_f1':query_det['f1'],'event_detection_counts':event_det,'query_detection_counts':query_det,'event_field_f1':event_field['f1'],'query_field_f1':query_field['f1'],'event_field_counts':event_field,'query_field_counts':query_field,'label_accuracy':label_acc,'predicted_event_count':len(pe),'predicted_query_count':len(pq),'gold_event_count':sum(g.get('event') is not None for g in gold.values()),'gold_query_count':sum(g.get('query') is not None for g in gold.values()),'invalid_parse_count':invalid_count,'invalid_parse_rate':invalid_count/len(manifest),'provider_diagnostics':all_diags,'status':status,'trusted_place_baseline':{k:trusted[k] for k in ('state_f1_mean','unauthorized_support_exposure_rate','gvr_mean','answer_f1_mean','jaur_mean')},'downstream':downstream,'pair_parser_extension':pair_ext,'deltas':deltas,'predictions':pred,'resume':{'checkpoint_schema':PARSER_CHECKPOINT_SCHEMA,'resumed_turn_predictions':resumed_predictions,'new_turn_predictions':len(manifest)-resumed_predictions,'pair_extension':pair_ext.get('resume')}}

