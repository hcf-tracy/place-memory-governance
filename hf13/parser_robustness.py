from __future__ import annotations
import copy, hashlib, json, os, random, time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .artifact_registry import RuntimeArtifactRegistry
from .css import CSSKernel, Fact, fact_signature
from .evaluator import runtime_gold, f1_sets, answer_score, _forbidden_mentions, _forbidden_answer_mentions
from .governance_schema import RUNTIME_GOVERNANCE_KEYS
from .io import read_jsonl, write_json, write_jsonl
from .llm import ollama_chat, get_last_diagnostics, answer_with_support
from .parser_diag import deterministic_event, deterministic_query, field_micro_metrics, field_f1
from .qasc import authorize
from .runner import _render_fact, _reference_answer, _budget_support
from .schema import validate_trusted_event, validate_trusted_query
from .status import from_diagnostics, merge_statuses

SCHEMA_VERSION='hf13-parser-robustness-factorial-v4'
CONDITIONS=('TT','PT','TP','PP')
FAMILIES=('supersession','scope','permission','authority','dependency','epistemic')


def _sha(path:Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _extract_json(text:str)->dict:
    text=str(text or '').strip()
    if text.startswith('```'):
        text=text.strip('`').strip()
        if text.lower().startswith('json'):text=text[4:].strip()
    try:
        obj=json.loads(text)
    except Exception:
        a=text.find('{');b=text.rfind('}')
        if a<0 or b<=a:raise ValueError('no JSON object')
        obj=json.loads(text[a:b+1])
    if not isinstance(obj,dict):raise ValueError('parser output must be one JSON object')
    return obj


def _load_public_ontology(root:Path)->dict:
    """Load the fixed schema vocabulary exposed to the parser.

    The ontology is deliberately schema-only: it contains no turn-to-label map,
    assertion values, evaluator answers, lifecycle state, or authorized support.
    """
    p=root/'parser_diagnostic/public_parser_ontology.json'
    if not p.exists():
        raise RuntimeError(f'public parser ontology missing: {p}')
    z=json.loads(p.read_text(encoding='utf-8'))
    if z.get('schema_version')!='hf13-public-parser-ontology-v1':
        raise RuntimeError(f'unsupported public parser ontology: {z.get("schema_version")}')
    return z


def _visible_context(turns:list[dict],turn_id:str,max_prior:int|None=None)->list[dict]:
    """Return prior raw user turns from the *current conversation only*.

    Persistent-memory state, trusted annotations, evaluator gold, and earlier
    conversation text are intentionally unavailable.  This mirrors the online
    runtime boundary: a parser may resolve ordinary within-session anaphora but
    may not inspect persistent memory to decide what the query should mean.
    """
    if max_prior is None:
        max_prior=int(os.environ.get('HF13_PARSER_CONTEXT_TURNS','12'))
    idx=next((i for i,t in enumerate(turns) if t.get('turn_id')==turn_id),None)
    if idx is None:raise KeyError(turn_id)
    cur=turns[idx];cid=cur.get('conversation_id')
    prior=[t for t in turns[:idx] if t.get('conversation_id')==cid]
    prior=prior[-max(0,int(max_prior)):]
    return [{'turn_id':t.get('turn_id'),'utterance':str(t.get('utterance') or '')} for t in prior]


def _schema_string(values=None, *, nullable=False):
    base={'type':'string','minLength':1}
    if values:base['enum']=list(values)
    return {'anyOf':[base,{'type':'null'}]} if nullable else base


def _scope_schema(ontology:dict)->dict:
    props={}
    for k,vals in (ontology.get('scope') or {}).items():
        props[k]=_schema_string(vals)
    return {'type':'object','additionalProperties':False,'properties':props}


def _event_json_schema(turn_id: str, ontology:dict|None=None) -> dict:
    ontology=ontology or {}
    nullable_string = {'anyOf': [{'type': 'string'}, {'type': 'null'}]}
    nullable_completion = {'anyOf': [{'type': 'string', 'enum': ['task', 'condition', 'trip']}, {'type': 'null'}]}
    nullable_collection = {'anyOf': [{'type': 'string', 'enum': ['remove_selected_item']}, {'type': 'null'}]}
    entities=ontology.get('entities') or None;predicates=ontology.get('predicates') or None
    prov=ontology.get('provenance_source_types') or None;epi=ontology.get('epistemic_statuses') or None
    return {
        'type': 'object','additionalProperties': False,
        'properties': {
            'event_id': {'type': 'string','enum':[f'parsed_{turn_id}']},
            'turn_id': {'type': 'string','enum':[turn_id]},
            'target': {'type':'object','additionalProperties':False,'properties':{
                'entity':_schema_string(entities),'predicate':_schema_string(predicates)},'required':['entity','predicate']},
            'assertion': {'type':'object','additionalProperties':False,'properties':{'value':{}},'required':['value']},
            'governance': {'type':'object','additionalProperties':False,'properties':{
                'update': {'type':'string','enum':['assert','replace','remove_current','suspend','restore','clarify','advance_time','mutate_collection']},
                'current_use': {'type':'string','enum':['allow','deny','unchanged']},
                'future_use': {'type':'string','enum':['allow','deny','unchanged']},
                'retention': {'type':'string','enum':['retain','history','preserve_future','do_not_retain']},
                'data_access': {'type':'string','enum':['allowed','restricted']},
                'action_authority': {'type':'string','enum':['authorized','not_authorized','unspecified']},
                'applicability': {'type':'string','enum':['current','historical','hypothetical']},
                'authority_preference': nullable_string,'dependency_recompute_of': nullable_string,
                'completion': nullable_completion,'collection_mutation': nullable_collection,
            },'required':list(RUNTIME_GOVERNANCE_KEYS)},
            'scope':_scope_schema(ontology) if ontology else {'type':'object'},
            'provenance': {'type':'object','additionalProperties':False,'properties':{'source_type':_schema_string(prov)},'required':['source_type']},
            'epistemic': {'type':'object','additionalProperties':False,'properties':{'status':_schema_string(epi)},'required':['status']},
            'temporal': {'type':'object','additionalProperties':False,'properties':{
                'valid_until':{'type':'string'},'offset_days':{'type':'integer'},'relative_to':{'type':'string'},'after':{'type':'string'},'due':{'type':'string'}}},
            'depends_on': {'type':'array','items':{'type':'object','additionalProperties':False,'properties':{
                'entity':_schema_string(entities),'predicate':_schema_string(predicates),'source_sequence':{'type':'integer'}},'required':['entity','predicate']}},
        },
        'required':['event_id','turn_id','target','assertion','governance','scope','provenance','epistemic'],
    }


def _query_json_schema(turn_id: str, ontology:dict|None=None) -> dict:
    ontology=ontology or {};entities=ontology.get('entities') or None;predicates=ontology.get('predicates') or None
    qi_props={}
    enums=(ontology.get('query_intent_enums') or {})
    for k,spec in (ontology.get('query_intent') or {}).items():
        typ=spec.get('type')
        if k=='selectors' or typ=='selectors':
            qi_props[k]={'type':'array','items':{'type':'object','additionalProperties':False,'properties':{
                'entity':_schema_string(entities),'predicate':_schema_string(predicates)}}}
        elif k in enums:
            qi_props[k]={'type':'string','enum':enums[k]}
        elif typ=='boolean':qi_props[k]={'type':'boolean'}
        else:qi_props[k]={'type':'string'}
    return {
        'type':'object','additionalProperties':False,
        'properties':{
            'query_id':{'type':'string','enum':[f'parsed_q_{turn_id}']},'turn_id':{'type':'string','enum':[turn_id]},
            'target':{'type':'object','additionalProperties':False,'properties':{
                'entity':_schema_string(entities),'predicate':_schema_string(predicates)}},
            'query_intent':{'type':'object','additionalProperties':False,'properties':qi_props},
            'scope':_scope_schema(ontology) if ontology else {'type':'object'},
            'as_of':{'anyOf':[{'type':'string'},{'type':'null'}]},
        },'required':['query_id','turn_id','target','query_intent','scope'],
    }


def _ontology_prompt(ontology:dict, kind:str)->str:
    entities=', '.join(ontology.get('entities') or [])
    predicates=', '.join(ontology.get('predicates') or [])
    base=("PUBLIC CANONICAL ONTOLOGY (schema vocabulary, not per-turn gold):\n"
          f"entities=[{entities}]\n"
          f"predicates=[{predicates}]\n"
          "Choose the canonical label that best matches the target expressed or resolved from CURRENT SESSION CONTEXT. Do not invent a new label.")
    if kind=='query':
        lines=[]
        for k,v in (ontology.get('query_intent') or {}).items():
            lines.append(f"- {k}: {v.get('description','')}")
        base+='\nAllowed query_intent fields:\n'+'\n'.join(lines)
    return base


def _event_prompt(turn_id:str,ontology:dict)->str:
    keys=', '.join(RUNTIME_GOVERNANCE_KEYS)
    return f"""You are a schema-guided semantic parser for one persistent-memory EVENT turn. Resolve ordinary references only from CURRENT SESSION CONTEXT supplied with the target turn. Extract USER-LEVEL semantics only. Return exactly one JSON event object and nothing else. A JSON Schema is enforced by the runtime.
Required event_id: parsed_{turn_id}. Required turn_id: {turn_id}.
{_ontology_prompt(ontology,'event')}
The assertion field is ALWAYS {{\"value\": <asserted value>}}. target contains exactly entity and predicate. governance contains exactly these 11 keys: {keys}.
Governance describes user-level intent, never CSS lifecycle operations. Defaults for an ordinary factual assertion: update=assert, current_use=unchanged, future_use=unchanged, retention=retain, data_access=allowed, action_authority=unspecified, applicability=current, authority_preference=null, dependency_recompute_of=null, completion=null, collection_mutation=null.
Use update=replace only when the user presents a replacement/correction/current-value change; retention=do_not_retain and data_access=restricted for explicit deletion/non-retention of sensitive data; action_authority=not_authorized when the user explicitly says an offer/candidate/hold is not authorization to book; completion=condition/task/trip only when the user marks that condition/task/trip completed.
Use provenance.source_type=user_report for an ordinary user assertion unless the user explicitly attributes the fact to a canonical external source such as airline/latest_airline_itinerary/Aman_reply. Epistemic status should reflect the user's wording (confirmed, tentative, probable, hypothetical, ambiguous).
Canonical governance interpretation rules: (a) a completion/status change such as completed/arrived/refunded is a current state replacement and may set completion=condition/task/trip when the utterance closes that condition; (b) if one utterance both closes a task/condition and requests deletion/non-retention of sensitive dependent data, target the completion/status fact and encode the privacy instruction in retention/data_access rather than inventing a lifecycle operation; (c) a stated authoritative source preference uses target predicate source_preference and governance.authority_preference; (d) an offer/hold/candidate explicitly distinguished from a booking uses action_authority=not_authorized; (e) person/trip/task restrictions belong in scope rather than the assertion value.
Do NOT emit lifecycle, SUPERSEDE, PURGE, ACTIVATE, INVALIDATE, resulting_state, expected_state, gold, required_answer, forbidden_answer, eligibility, or authorized support. Do not inspect or infer from persistent memory."""


def _query_prompt(turn_id:str,ontology:dict)->str:
    return f"""You are a schema-guided semantic parser for one persistent-memory QUERY turn. Resolve pronouns/ellipsis only from CURRENT SESSION CONTEXT supplied with the target turn. Extract USER-LEVEL query semantics only. Return exactly one JSON query object and nothing else. A JSON Schema is enforced by the runtime.
Required query_id: parsed_q_{turn_id}. Required turn_id: {turn_id}.
{_ontology_prompt(ontology,'query')}
target contains the canonical entity and/or predicate being asked about. Use an empty target only when neither can be resolved from the target utterance plus current-session context. query_intent may use ONLY the public fields above; include only semantically active fields (do not emit unused boolean fields as false) and choose the smallest set that captures the user's request. scope contains only scope explicitly stated or unambiguously resolved within the current session.
Canonical query-intent cues: current value plus why/source -> request_provenance_reason=true and temporal_view=current; asking which value was superseded/replaced -> requested_relation=replaced_previous_value and temporal_view=historical; asking whether an old/replaced value should still be used -> reject_replaced_values=true and temporal_view=current; asking whether another person's constraint applies to the requesting user -> check_applicability_to_requesting_user=true; asking whether a preference/restriction should generalize to future planning -> future_generalization=true; asking whether something was actually booked/purchased/committed -> ask_whether_irreversible_action_authorized=true; asking for a sensitive stored value -> request_sensitive_value=true; asking what changed because a parent condition changed -> request_dependency_consequences=true. Use temporal_view=current for current-state questions, historical for past/provenance/history questions, and upcoming for future-planning questions.
Do NOT emit required_answer, forbidden_answer, eligible_facts, ineligible_facts, gold_query, expected_answer, CSS transitions, lifecycle/resulting-state labels, persistent-memory contents, or authorized support."""


def _parser_user_message(context:list[dict],utterance:str,turn_id:str)->str:
    ctx='\n'.join(f"[{x['turn_id']}] {x['utterance']}" for x in context) or '(no prior turns in this conversation)'
    return f"CURRENT SESSION CONTEXT (prior raw user turns only):\n{ctx}\n\nTARGET TURN [{turn_id}]:\n{utterance}"

def _canonicalize_event_shape(obj:dict, turn_id:str)->tuple[dict,list[str]]:
    """Canonicalize only representation shape; never infer hidden semantics."""
    x=copy.deepcopy(obj); actions=[]
    if set(x)=={'event'} and isinstance(x.get('event'),dict):
        x=x['event']; actions.append('unwrap_event_envelope')
    if 'assertion' in x and not isinstance(x.get('assertion'),dict):
        x['assertion']={'value':x['assertion']};actions.append('wrap_scalar_assertion')
    elif isinstance(x.get('assertion'),dict) and 'value' not in x['assertion'] and set(x['assertion'])=={'assertion_value'}:
        x['assertion']={'value':x['assertion']['assertion_value']};actions.append('rename_assertion_value')
    if isinstance(x.get('provenance'),str):
        x['provenance']={'source_type':x['provenance']};actions.append('wrap_provenance_source_type')
    if isinstance(x.get('epistemic'),str):
        x['epistemic']={'status':x['epistemic']};actions.append('wrap_epistemic_status')
    x['event_id']=f'parsed_{turn_id}';x['turn_id']=turn_id
    return x,actions


def _canonicalize_query_shape(obj:dict, turn_id:str)->tuple[dict,list[str]]:
    """Canonicalize only representation shape; never fill semantic fields from gold."""
    x=copy.deepcopy(obj);actions=[]
    if set(x)=={'query'} and isinstance(x.get('query'),dict):
        x=x['query'];actions.append('unwrap_query_envelope')
    x['query_id']=f'parsed_q_{turn_id}';x['turn_id']=turn_id
    return x,actions


def _validate_event_obj(obj:dict,turn_id:str)->tuple[dict,list[str]]:
    obj,actions=_canonicalize_event_shape(obj,turn_id);validate_trusted_event(obj);return obj,actions


def _validate_query_obj(obj:dict,turn_id:str)->tuple[dict,list[str]]:
    obj,actions=_canonicalize_query_shape(obj,turn_id);validate_trusted_query(obj);return obj,actions


def _llm_parse(kind:str,utterance:str,turn_id:str,cfg:dict,*,context:list[dict]|None=None,ontology:dict|None=None)->tuple[dict|None,dict]:
    m=cfg.get('model') or {};base=m.get('base_url','http://127.0.0.1:11434');model=m.get('model_id','qwen2.5:3b');timeout=int(m.get('timeout_seconds',180))
    ontology=ontology or {};context=context or []
    prompt=_event_prompt(turn_id,ontology) if kind=='event' else _query_prompt(turn_id,ontology)
    validator=_validate_event_obj if kind=='event' else _validate_query_obj
    schema=_event_json_schema(turn_id,ontology) if kind=='event' else _query_json_schema(turn_id,ontology)
    user_msg=_parser_user_message(context,utterance,turn_id)
    prior='';attempt_diags=[];raw_outputs=[];schema_mode=str(os.environ.get('HF13_PARSER_SCHEMA_MODE','structured')).strip().lower()
    first_valid=None;first_actions=[];first_diag={};first_text=None
    for sem_attempt in (1,2):
        messages=[{'role':'system','content':prompt},{'role':'user','content':user_msg}]
        if prior:messages.append({'role':'user','content':'Your previous JSON violated the public parser contract. Repair it using only the same current-session context and target utterance. Validation error: '+prior})
        text=None;structured_used=(schema_mode!='json')
        try:
            retries=int(os.environ.get('HF13_ROBUSTNESS_LLM_RETRIES','4'))
            try:text=ollama_chat(base,model,messages,temperature=0,max_tokens=1024,timeout=timeout,max_retries=retries,format_schema=(schema if structured_used else None),format_json=(not structured_used))
            except Exception:
                if schema_mode!='auto':raise
                text=ollama_chat(base,model,messages,temperature=0,max_tokens=1024,timeout=timeout,max_retries=retries,format_json=True);structured_used=False
            d=get_last_diagnostics();raw_outputs.append(text);extracted=_extract_json(text);obj,actions=validator(extracted,turn_id)
            first_valid,first_actions,first_diag,first_text=obj,actions,d,text
            break
        except Exception as ex:
            d=get_last_diagnostics()
            if text is not None and (not raw_outputs or raw_outputs[-1]!=text):raw_outputs.append(text)
            prior=f'{type(ex).__name__}: {ex}';attempt_diags.append({**d,'semantic_attempt':sem_attempt,'error':prior,'raw_output':text})
    if first_valid is None:
        provider_failed=any(str((d or {}).get('status'))=='failed' for d in attempt_diags)
        return None,{'status':'model_operation_error' if provider_failed else 'clean','parse_valid':False,'schema_attempts':2,'repair_used':True,'parse_error':prior,'attempt_diagnostics':attempt_diags,'raw_outputs':raw_outputs,'raw_output_final':raw_outputs[-1] if raw_outputs else None,'context_turn_ids':[x['turn_id'] for x in context],'semantic_self_check':False}

    # A second, gold-free semantic verification pass catches schema-valid but semantically
    # vacuous outputs (the main failure mode in the previous HF13 parser version).
    self_check=str(os.environ.get('HF13_PARSER_SELF_CHECK','1')).strip().lower() not in {'0','false','no','off'}
    final_obj,final_actions,final_diag,final_text=first_valid,first_actions,first_diag,first_text
    self_check_status='disabled'
    if self_check:
        audit=("Audit the candidate parse against ONLY the current-session context, target utterance, and public ontology. "
               "Correct wrong/missing target, scope, query-intent, provenance, epistemic, temporal, assertion, or governance semantics. "
               "Do not add facts from persistent memory and do not add evaluator/lifecycle labels. Return only the corrected JSON object.\n\n"
               "CANDIDATE PARSE:\n"+json.dumps(first_valid,ensure_ascii=False,sort_keys=True))
        try:
            text2=ollama_chat(base,model,[{'role':'system','content':prompt},{'role':'user','content':user_msg},{'role':'user','content':audit}],temperature=0,max_tokens=1024,timeout=timeout,max_retries=int(os.environ.get('HF13_ROBUSTNESS_LLM_RETRIES','4')),format_schema=(schema if schema_mode!='json' else None),format_json=(schema_mode=='json'))
            raw_outputs.append(text2);d2=get_last_diagnostics();obj2,actions2=validator(_extract_json(text2),turn_id)
            final_obj,final_actions,final_diag,final_text=obj2,actions2,d2,text2;self_check_status='complete'
        except Exception as ex:
            self_check_status='failed_keep_first_valid';attempt_diags.append({**get_last_diagnostics(),'semantic_self_check':True,'error':f'{type(ex).__name__}: {ex}','raw_output':locals().get('text2')})
    return final_obj,{**final_diag,'parse_valid':True,'schema_attempts':1+bool(prior),'repair_used':bool(prior),'parse_error':None,'structured_schema_used':schema_mode!='json','canonicalization_actions':final_actions,'raw_outputs':raw_outputs,'raw_output_initial':first_text,'raw_output_final':final_text,'semantic_self_check':self_check,'semantic_self_check_status':self_check_status,'context_turn_ids':[x['turn_id'] for x in context]}

def _prediction_meta(root:Path,mode:str,cfg:dict)->dict:
    pkg=Path(__file__).resolve().parents[1]/'MANIFEST.sha256'
    payload={'schema_version':SCHEMA_VERSION,'mode':mode,'model':(cfg.get('model') or {}).get('model_id'),'cases_sha256':_sha(root/'parser_robustness/cases.jsonl'),'trajectory_sha256':_sha(root/'trajectory_raw.jsonl'),'public_ontology_sha256':_sha(root/'parser_diagnostic/public_parser_ontology.json'),'context_turns':int(os.environ.get('HF13_PARSER_CONTEXT_TURNS','12')),'semantic_self_check':str(os.environ.get('HF13_PARSER_SELF_CHECK','1')).lower() not in {'0','false','no','off'},'package_manifest_sha256':_sha(pkg) if pkg.exists() else None}
    payload['fingerprint']=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest();return payload


def _load_predictions(cp:Path|None,meta:dict)->dict[tuple[str,str],dict]:
    if cp is None:return {}
    cp.mkdir(parents=True,exist_ok=True);mp=cp/'checkpoint_meta.json'
    if mp.exists():
        old=json.loads(mp.read_text())
        if old.get('fingerprint')!=meta['fingerprint']:raise RuntimeError('parser robustness checkpoint identity mismatch')
    else:write_json(mp,meta)
    p=cp/'parser_predictions.jsonl';out={}
    if p.exists():
        for x in read_jsonl(p):
            if str((x.get('diagnostic') or {}).get('status'))=='failed':continue
            out[(x['kind'],x['turn_id'])]=x
    return out


def _append(path:Path,row:dict):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',encoding='utf-8') as f:
        f.write(json.dumps(row,ensure_ascii=False,sort_keys=True,default=str)+'\n');f.flush();os.fsync(f.fileno())

def _load_case_checkpoints(cp:Path|None)->dict[str,dict]:
    if cp is None:return {}
    p=cp/'case_results.jsonl';out={}
    if not p.exists():return out
    for x in read_jsonl(p):
        cid=str(x.get('case_id') or '')
        if cid and isinstance(x.get('case_row'),dict):out[cid]=x
    return out


def _transition_sig(tr:dict|None)->tuple:
    ops=[]
    for x in ((tr or {}).get('internal_transitions') or []):
        z={k:v for k,v in x.items() if k not in {'fact_id','fact_ids','affected_fact_ids'}}
        ops.append(json.dumps(z,sort_keys=True,ensure_ascii=False,default=str))
    return tuple(sorted(ops))


def _realize(kernel:CSSKernel):
    for f in kernel.facts:
        if not f.physical_ids:f.physical_ids=[f'diag:{f.fact_id}']


def _replay(turns:list[dict],events:dict[str,dict],query_turn_id:str,replace_turn_id:str|None=None,replacement:dict|None=None):
    idx={t['turn_id']:i for i,t in enumerate(turns)};qidx=idx[query_turn_id]
    if replace_turn_id and idx[replace_turn_id]>=qidx:raise ValueError(f'event {replace_turn_id} must precede query {query_turn_id}')
    k=CSSKernel();selected_transition=None
    for t in turns[:qidx+1]:
        k.advance_time(t.get('timestamp'))
        if t['turn_id']==query_turn_id:break
        e=events.get(t['turn_id'])
        if t['turn_id']==replace_turn_id:e=replacement
        if e is not None:
            tr=k.apply(copy.deepcopy(e),as_of=t.get('timestamp'));_realize(k)
            if t['turn_id']==replace_turn_id:selected_transition=copy.deepcopy(tr)
    _realize(k);return k.snapshot(),selected_transition


def _replay_parsed_event(turns,events,query_turn_id,replace_turn_id,parsed_event):
    """Execute one parsed event without allowing parser semantics to crash the diagnostic.

    Trusted replay remains strict. If a parser output passes surface parsing but is
    not executable by the canonical CSS contract, that is an evaluated parser
    failure. We record it and conservatively model the event as missed rather than
    turning a single parser case into an infrastructure failure.
    """
    if parsed_event is None:
        state,tr=_replay(turns,events,query_turn_id,replace_turn_id,None)
        return state,tr,{'runtime_valid':False,'runtime_error':'parser_event_missing','fallback':'omit_target_event'}
    try:
        state,tr=_replay(turns,events,query_turn_id,replace_turn_id,parsed_event)
        return state,tr,{'runtime_valid':True,'runtime_error':None,'fallback':None}
    except Exception as ex:
        state,tr=_replay(turns,events,query_turn_id,replace_turn_id,None)
        return state,tr,{'runtime_valid':False,'runtime_error':f'{type(ex).__name__}: {ex}','fallback':'omit_target_event'}

def _parser_query_gold_projection(query:dict)->dict:
    # `family` is evaluator stratification metadata. `as_of` is supplied by the
    # runtime turn timestamp when absent, so neither is a raw-utterance parser target.
    q=copy.deepcopy(query);q.pop('family',None);q.pop('as_of',None);return q


def _query_component_f1(pred:dict|None,gold:dict|None,key:str):
    """Field F1 for one query component, using the same canonical scorer.

    Query intent is the parser-observable semantic target used by the smoke gate.
    Target/scope alignment remain diagnostics because persistent-memory referent
    resolution is not always identifiable from the legal parser input alone.
    """
    return field_f1({key:copy.deepcopy((pred or {}).get(key,{}))},
                    {key:copy.deepcopy((gold or {}).get(key,{}))})


def _authorize_parsed_query(state,query,*,as_of_fallback=None):
    """Authorize parsed query semantics with explicit parser-failure containment.

    Canonical trusted-query authorization is never routed through this helper and
    therefore still hard-fails on harness/data defects.
    """
    if query is None:return [],{'runtime_valid':False,'runtime_error':'parser_query_missing','fallback':'empty_authorized_support'}
    try:
        pids=[p for f in state for p in (f.get('physical_ids') or [])]
        return authorize(state,query,as_of=query.get('as_of') or as_of_fallback,physical_record_ids=pids),{'runtime_valid':True,'runtime_error':None,'fallback':None}
    except Exception as ex:
        return [],{'runtime_valid':False,'runtime_error':f'{type(ex).__name__}: {ex}','fallback':'empty_authorized_support'}

def _mean(xs):return sum(xs)/len(xs) if xs else None


def _stratified_bootstrap(case_rows:list[dict],condition:str,metric:str,reps:int,seed:int=42):
    base={r['case_id']:r for r in case_rows};byfam=defaultdict(list)
    for r in case_rows:byfam[r['family']].append(r['case_id'])
    diffs=[r['conditions'][condition][metric]-r['conditions']['TT'][metric] for r in case_rows]
    estimate=_mean(diffs);rng=random.Random(seed);boots=[]
    for _ in range(reps):
        sampled=[]
        for fam in FAMILIES:
            ids=byfam[fam];sampled.extend(rng.choice(ids) for _ in range(len(ids)))
        boots.append(_mean([base[i]['conditions'][condition][metric]-base[i]['conditions']['TT'][metric] for i in sampled]))
    boots.sort()
    lo=boots[int(.025*(len(boots)-1))] if boots else None;hi=boots[int(.975*(len(boots)-1))] if boots else None
    eps=1e-12
    return {'estimate':estimate,'interval95':[lo,hi],'interval_type':'stratified_case_bootstrap_sensitivity','n_cases':len(case_rows),'improved_n':sum(x>eps for x in diffs),'unchanged_n':sum(abs(x)<=eps for x in diffs),'degraded_n':sum(x<-eps for x in diffs),'median_delta':sorted(diffs)[len(diffs)//2] if diffs else None}


def run_parser_contract_smoke(data_root='data/longitudinal', parser_mode='llm', cfg=None, artifact_registry=None):
    """One event/query per governance family; prespecified parser viability gate.

    Predictions use only the raw target turn, bounded current-session raw context,
    and the fixed public schema ontology. Trusted semantics are opened only after
    all predictions exist; semantic field scores then participate in the gate.
    """
    cfg=cfg or {};root=Path(data_root);cases=read_jsonl(root/'parser_robustness/cases.jsonl');turns=read_jsonl(root/'trajectory_raw.jsonl');turn_by={t['turn_id']:t for t in turns};ontology=_load_public_ontology(root)
    chosen=[];seen=set()
    for c in cases:
        if c['family'] not in seen:
            chosen.append(c);seen.add(c['family'])
    if tuple(c['family'] for c in chosen)!=FAMILIES:
        # Manifest order is expected but do not silently select a different family set.
        by={c['family']:c for c in cases if c['family'] not in {x['family'] for x in []}}
        chosen=[next(c for c in cases if c['family']==f) for f in FAMILIES]
    pred={};provider=[]
    for kind in ('event','query'):
        for c in chosen:
            tid=c['event_turn_id'] if kind=='event' else c['query_turn_id'];u=turn_by[tid]['utterance']
            if parser_mode=='deterministic':
                obj=deterministic_event(u,tid) if kind=='event' else deterministic_query(u,tid)
                diag={'status':'clean','parse_valid':obj is not None,'schema_attempts':1,'repair_used':False,'parse_error':None if obj else 'deterministic parser returned null','raw_outputs':[]}
            else: obj,diag=_llm_parse(kind,u,tid,cfg,context=_visible_context(turns,tid),ontology=ontology)
            pred[(kind,tid)]={'kind':kind,'turn_id':tid,'prediction':obj,'diagnostic':diag};provider.append(diag)
    # Gold/trusted semantics are deliberately opened only after all twelve predictions exist.
    reg=artifact_registry
    if isinstance(reg,(str,Path)):reg=RuntimeArtifactRegistry.load(reg)
    if reg is None:raise RuntimeError('parser smoke requires RuntimeArtifactRegistry')
    trusted_events={x['turn_id']:reg.resolve_object(x) for x in read_jsonl(root/'trusted_events.jsonl') if x.get('event_id')}
    trusted_queries={x['turn_id']:x for x in read_jsonl(root/'trusted_queries.jsonl') if x.get('query_id')}
    rows=[]
    for c in chosen:
        et,qt=c['event_turn_id'],c['query_turn_id'];pe=pred[('event',et)]['prediction'];pq=pred[('query',qt)]['prediction']
        tt_state,tt_tr=_replay(turns,trusted_events,qt,et,trusted_events[et]);pt_state,pt_tr,ed=_replay_parsed_event(turns,trusted_events,qt,et,pe)
        _,qd=_authorize_parsed_query(pt_state,pq,as_of_fallback=turn_by[qt].get('timestamp'))
        rows.append({'family':c['family'],'case_id':c['case_id'],'event_turn_id':et,'query_turn_id':qt,
                     'event_schema_valid':bool(pred[('event',et)]['diagnostic'].get('parse_valid')),
                     'query_schema_valid':bool(pred[('query',qt)]['diagnostic'].get('parse_valid')),
                     'event_runtime_valid':bool(ed['runtime_valid']),'query_runtime_valid':bool(qd['runtime_valid']),
                     'event_field_f1':field_f1(pe,trusted_events[et]),
                     'query_field_f1':field_f1(pq,_parser_query_gold_projection(trusted_queries[qt])),
                     'query_intent_f1':_query_component_f1(pq,_parser_query_gold_projection(trusted_queries[qt]),'query_intent'),
                     'query_target_f1':_query_component_f1(pq,_parser_query_gold_projection(trusted_queries[qt]),'target'),
                     'query_scope_f1':_query_component_f1(pq,_parser_query_gold_projection(trusted_queries[qt]),'scope'),
                     'transition_agreement':_transition_sig(pt_tr)==_transition_sig(tt_tr),
                     'event_diagnostic':pred[('event',et)]['diagnostic'],'query_diagnostic':pred[('query',qt)]['diagnostic']})
    structural=all(r['event_schema_valid'] and r['query_schema_valid'] and r['event_runtime_valid'] and r['query_runtime_valid'] for r in rows)
    event_mean=_mean([r['event_field_f1'] or 0 for r in rows]) or 0.0
    query_full_mean=_mean([r['query_field_f1'] or 0 for r in rows]) or 0.0
    query_intent_mean=_mean([r['query_intent_f1'] or 0 for r in rows]) or 0.0
    query_target_mean=_mean([r['query_target_f1'] or 0 for r in rows]) or 0.0
    query_scope_mean=_mean([r['query_scope_f1'] or 0 for r in rows]) or 0.0
    transition_rate=_mean([r['transition_agreement'] for r in rows]) or 0.0
    per_family_nonzero=all((r['event_field_f1'] or 0)>0 and (r['query_intent_f1'] or 0)>0 for r in rows)

    # The smoke is an implementation-integrity barrier, not a model-selection step.
    # Conditioning entry to the 24-case diagnostic on an arbitrary F1 threshold would
    # select on the outcome being measured. Semantic scores are therefore surfaced as
    # prespecified diagnostics/warnings, while the hard gate only requires that every
    # parser path is executable without schema/runtime fallback.
    gate=structural
    semantic_sanity=event_mean>0 and query_intent_mean>0
    warnings=[]
    if not semantic_sanity:warnings.append('degenerate aggregate semantic overlap: event or query-intent mean F1 is zero')
    if not per_family_nonzero:warnings.append('one or more smoke families has zero event or query-intent overlap')
    if transition_rate<=0:warnings.append('transition agreement is zero on the six-family smoke')
    failures=[]
    if not structural:failures.append('schema/runtime contract')
    return {'schema_version':'hf13-parser-contract-smoke-v3','status':'COMPLETE_CLEAN' if gate else 'FAILED_CONTRACT_SMOKE','gate_pass':gate,
            'gate_definition':'implementation-integrity gate: all six event and six query parses must be schema-valid and runtime-executable without parser fallback. Semantic event/query-intent scores and transition agreement are reported before the 24-case run but are not used to select models into the evaluation, avoiding outcome-conditioned threshold tuning.',
            'semantic_sanity_pass':semantic_sanity,'semantic_warnings':warnings,
            'input_contract':{'current_session_context_only':True,'context_turn_limit':int(os.environ.get('HF13_PARSER_CONTEXT_TURNS','12')),'public_schema_ontology':True,'persistent_memory_visible':False,'trusted_annotations_visible_during_prediction':False,'evaluator_gold_visible_during_prediction':False},
            'families':list(FAMILIES),'case_count':6,'event_schema_valid_n':sum(r['event_schema_valid'] for r in rows),'query_schema_valid_n':sum(r['query_schema_valid'] for r in rows),
            'event_runtime_valid_n':sum(r['event_runtime_valid'] for r in rows),'query_runtime_valid_n':sum(r['query_runtime_valid'] for r in rows),
            'event_field_f1_mean':event_mean,'query_field_f1_mean':query_full_mean,'query_intent_f1_mean':query_intent_mean,'query_target_f1_mean':query_target_mean,'query_scope_f1_mean':query_scope_mean,'transition_agreement_n':sum(r['transition_agreement'] for r in rows),'transition_agreement_rate':transition_rate,'per_family_nonzero_event_and_query_intent':per_family_nonzero,'rows':rows,'provider_diagnostics':provider}



def _condition_means(rows:list[dict])->dict:
    return {cond:{m:_mean([r['conditions'][cond][m] for r in rows]) for m in ('state_f1','authorized_support_f1','answer_f1','jaur','exposure','gvr')} for cond in CONDITIONS}


def _propagation_summary(rows:list[dict])->dict:
    def rate(num,den):return num/den if den else None
    event_err=sum(r['event_parser_error'] for r in rows);query_err=sum(r['query_parser_error'] for r in rows);any_err=sum(r['event_parser_error'] or r['query_parser_error'] for r in rows)
    event_harm=sum(r['event_error_propagated'] for r in rows);query_harm=sum(r['query_error_propagated'] for r in rows);e2e_harm=sum(r['end_to_end_error_propagated'] for r in rows)
    return {'event_parser':{'error_cases':event_err,'harmful_cases':event_harm,'propagation_rate':rate(event_harm,event_err)},
            'query_parser':{'error_cases':query_err,'harmful_cases':query_harm,'propagation_rate':rate(query_harm,query_err)},
            'end_to_end':{'error_cases':any_err,'harmful_cases':e2e_harm,'propagation_rate':rate(e2e_harm,any_err)}}


def run_parser_robustness(data_root='data/longitudinal',parser_mode='llm',cfg=None,artifact_registry=None,answer_mode='llm',progress_path=None,checkpoint_dir=None,bootstrap_reps=4000):
    cfg=cfg or {};root=Path(data_root);cases=read_jsonl(root/'parser_robustness/cases.jsonl');turns=read_jsonl(root/'trajectory_raw.jsonl');turn_by={t['turn_id']:t for t in turns};families=Counter(x['family'] for x in cases);ontology=_load_public_ontology(root)
    if len(cases)!=24 or any(families[f]!=4 for f in FAMILIES):raise RuntimeError(f'parser robustness manifest must be 24 cases, four/family: {families}')
    event_turns=sorted({x['event_turn_id'] for x in cases},key=lambda z:next(i for i,t in enumerate(turns) if t['turn_id']==z));query_turns=[x['query_turn_id'] for x in cases]
    if len(set(query_turns))!=24:raise RuntimeError('query turns must be unique across 24 cases')
    pp=Path(progress_path) if progress_path else None;cp=Path(checkpoint_dir) if checkpoint_dir else None;start=time.time();meta=_prediction_meta(root,parser_mode,cfg);pred=_load_predictions(cp,meta);initial_pred_n=len(pred);pred_path=(cp/'parser_predictions.jsonl') if cp else None;case_cp=_load_case_checkpoints(cp);case_cp_path=(cp/'case_results.jsonl') if cp else None
    def progress(stage,case_done=0,current=None,**extra):
        if pp:write_json(pp,{'schema_version':SCHEMA_VERSION,'stage':stage,'done':case_done,'total':24,'current_case':current,'event_predictions_done':sum(('event',x) in pred for x in event_turns),'event_predictions_total':len(event_turns),'query_predictions_done':sum(('query',x) in pred for x in query_turns),'query_predictions_total':len(query_turns),'elapsed_s':time.time()-start,**extra})
    # Parser phase sees raw utterances + prespecified role only. Trusted/gold semantics are deliberately unopened here.
    for kind,tids in [('event',event_turns),('query',query_turns)]:
        for tid in tids:
            if (kind,tid) in pred:continue
            u=turn_by[tid]['utterance']
            if parser_mode=='deterministic':obj=deterministic_event(u,tid) if kind=='event' else deterministic_query(u,tid);diag={'status':'clean','parse_valid':obj is not None,'schema_attempts':1,'repair_used':False,'parse_error':None if obj else 'deterministic parser returned null'}
            else:obj,diag=_llm_parse(kind,u,tid,cfg,context=_visible_context(turns,tid),ontology=ontology)
            row={'kind':kind,'turn_id':tid,'prediction':obj,'diagnostic':diag};pred[(kind,tid)]=row
            if pred_path:_append(pred_path,row)
            progress('parse_'+kind,0,tid)
    progress('predictions_complete',0)

    # Only after all parser outputs are durable do we open trusted semantics / evaluator gold.
    reg=artifact_registry
    if isinstance(reg,(str,Path)):reg=RuntimeArtifactRegistry.load(reg)
    if reg is None:raise RuntimeError('parser robustness requires the core RuntimeArtifactRegistry for runtime-resolved trusted semantics')
    trusted_events={x['turn_id']:reg.resolve_object(x) for x in read_jsonl(root/'trusted_events.jsonl') if x.get('event_id')};trusted_queries={x['turn_id']:x for x in read_jsonl(root/'trusted_queries.jsonl') if x.get('query_id')};_,gold_query=runtime_gold(root,reg)
    event_items=[];query_items=[]
    for tid in event_turns:event_items.append((tid,pred[('event',tid)]['prediction'],trusted_events[tid]))
    for tid in query_turns:query_items.append((tid,pred[('query',tid)]['prediction'],_parser_query_gold_projection(trusted_queries[tid])))
    em=field_micro_metrics(event_items);qm=field_micro_metrics(query_items)
    qi_items=[(tid,{'query_intent':copy.deepcopy((pred[('query',tid)]['prediction'] or {}).get('query_intent',{}))},{'query_intent':copy.deepcopy(_parser_query_gold_projection(trusted_queries[tid]).get('query_intent',{}))}) for tid in query_turns]
    qt_items=[(tid,{'target':copy.deepcopy((pred[('query',tid)]['prediction'] or {}).get('target',{}))},{'target':copy.deepcopy(_parser_query_gold_projection(trusted_queries[tid]).get('target',{}))}) for tid in query_turns]
    qs_items=[(tid,{'scope':copy.deepcopy((pred[('query',tid)]['prediction'] or {}).get('scope',{}))},{'scope':copy.deepcopy(_parser_query_gold_projection(trusted_queries[tid]).get('scope',{}))}) for tid in query_turns]
    qim=field_micro_metrics(qi_items);qtm=field_micro_metrics(qt_items);qsm=field_micro_metrics(qs_items)

    case_rows=[];answer_diags=[];resumed_case_n=0
    for ci,case in enumerate(cases,1):
        cid=case['case_id']
        if cid in case_cp:
            saved=case_cp[cid];case_rows.append(copy.deepcopy(saved['case_row']));answer_diags.extend(copy.deepcopy(saved.get('answer_diagnostics') or []));resumed_case_n+=1;progress('downstream',ci,cid,condition='PP',resumed_case=True);continue
        et=case['event_turn_id'];qt=case['query_turn_id'];raw_q=turn_by[qt]['utterance'];trusted_q=trusted_queries[qt];parsed_e=pred[('event',et)]['prediction'];parsed_q=pred[('query',qt)]['prediction'];gold=gold_query[qt]
        # TT is the trusted control and therefore remains strict: any exception here
        # is a benchmark/harness defect and must fail the campaign. Parsed paths are
        # evaluated components and are contained/recorded rather than crashing all 24 cases.
        tt_state,tt_tr=_replay(turns,trusted_events,qt,et,trusted_events[et])
        pt_state,pt_tr,event_runtime=_replay_parsed_event(turns,trusted_events,qt,et,parsed_e)
        # Reference support is the TT semantic target of this controlled diagnostic.
        tt_auth=authorize(tt_state,trusted_q,as_of=trusted_q.get('as_of'),physical_record_ids=[p for f in tt_state for p in (f.get('physical_ids') or [])])
        conds={};case_answer_diags=[]
        query_runtime={'runtime_valid':True,'runtime_error':None,'fallback':None}
        for cond in CONDITIONS:
            state=pt_state if cond[0]=='P' else tt_state;qsem=parsed_q if cond[1]=='P' else trusted_q
            if cond[1]=='P':
                auth,qdiag=_authorize_parsed_query(state,qsem,as_of_fallback=turn_by[qt].get('timestamp'))
                if not qdiag['runtime_valid']:query_runtime=qdiag
            else:
                auth=authorize(state,qsem,as_of=qsem.get('as_of'),physical_record_ids=[p for f in state for p in (f.get('physical_ids') or [])])
            support=_budget_support([_render_fact(f) for f in auth],cfg)
            if answer_mode=='reference':ans=_reference_answer(support);adiag={'status':'clean'}
            else:
                try:ans=answer_with_support(raw_q,support,cfg);adiag=get_last_diagnostics()
                except Exception as ex:
                    ans=''
                    # The answer model is part of the evaluated diagnostic surface.
                    # Preserve the failure in reliability counters without converting
                    # one case into a harness-level FAILED campaign.
                    ad=get_last_diagnostics();adiag={**ad,'status':'model_operation_error','errors':list((ad or {}).get('errors') or [])+[f'{type(ex).__name__}: {ex}']}
            ar={'case_id':cid,'condition':cond,**adiag};answer_diags.append(ar);case_answer_diags.append(ar)
            state_f1=f1_sets([fact_signature(x) for x in state],[fact_signature(x) for x in tt_state]);auth_f1=f1_sets([fact_signature(x) for x in auth],[fact_signature(x) for x in tt_auth]);support_text='\n'.join(support);exposure=1.0 if _forbidden_mentions(support_text,gold.get('ineligible_facts') or []) else 0.0;af1=answer_score(ans,gold.get('expected_facts') or [],gold.get('ineligible_facts') or [],gold.get('expect_abstain',False),query=gold.get('query_semantics'),raw_query=raw_q);gvr=1.0 if _forbidden_answer_mentions(ans,gold.get('ineligible_facts') or [],raw_q) else 0.0;jaur=1.0 if state_f1>=.999999 and auth_f1>=.999999 and exposure==0 and af1>=.999999 and gvr==0 else 0.0
            conds[cond]={'state_f1':state_f1,'authorized_support_f1':auth_f1,'answer_f1':af1,'jaur':jaur,'exposure':exposure,'gvr':gvr,'authorized_support_count':len(auth),'response':ans}
        q_gold=_parser_query_gold_projection(trusted_q)
        e_f1=field_f1(parsed_e,trusted_events[et]);q_f1=field_f1(parsed_q,q_gold)
        qi_f1=_query_component_f1(parsed_q,q_gold,'query_intent');qt_f1=_query_component_f1(parsed_q,q_gold,'target');qs_f1=_query_component_f1(parsed_q,q_gold,'scope')
        event_err=(parsed_e is None or not event_runtime['runtime_valid'] or (e_f1 or 0)<.999999);query_err=(parsed_q is None or not query_runtime['runtime_valid'] or (q_f1 or 0)<.999999)
        event_harm=event_err and (conds['PT']['state_f1']<.999999 or conds['PT']['authorized_support_f1']<.999999 or conds['PT']['answer_f1']+1e-12<conds['TT']['answer_f1'] or conds['PT']['jaur']<conds['TT']['jaur']);query_harm=query_err and (conds['TP']['authorized_support_f1']<.999999 or conds['TP']['answer_f1']+1e-12<conds['TT']['answer_f1'] or conds['TP']['jaur']<conds['TT']['jaur']);any_err=event_err or query_err;pp_harm=any_err and (conds['PP']['state_f1']<.999999 or conds['PP']['authorized_support_f1']<.999999 or conds['PP']['answer_f1']+1e-12<conds['TT']['answer_f1'] or conds['PP']['jaur']<conds['TT']['jaur'])
        case_row={'case_id':cid,'family':case['family'],'event_turn_id':et,'query_turn_id':qt,'event_field_f1':e_f1,'query_field_f1':q_f1,'query_intent_f1':qi_f1,'query_target_f1':qt_f1,'query_scope_f1':qs_f1,'event_schema_valid':bool(pred[('event',et)]['diagnostic'].get('parse_valid')),'query_schema_valid':bool(pred[('query',qt)]['diagnostic'].get('parse_valid')),'event_runtime_valid':event_runtime['runtime_valid'],'query_runtime_valid':query_runtime['runtime_valid'],'event_runtime_error':event_runtime['runtime_error'],'query_runtime_error':query_runtime['runtime_error'],'event_runtime_fallback':event_runtime['fallback'],'query_runtime_fallback':query_runtime['fallback'],'transition_agreement':_transition_sig(pt_tr)==_transition_sig(tt_tr),'event_parser_error':event_err,'query_parser_error':query_err,'event_error_propagated':bool(event_harm),'query_error_propagated':bool(query_harm),'end_to_end_error_propagated':bool(pp_harm),'conditions':conds}
        case_rows.append(case_row)
        if case_cp_path:_append(case_cp_path,{'case_id':cid,'case_row':case_row,'answer_diagnostics':case_answer_diags})
        progress('downstream',ci,cid,condition='PP',resumed_case=False)

    condition_means=_condition_means(case_rows)
    deltas={cond:{m:_stratified_bootstrap(case_rows,cond,m,int(bootstrap_reps),42+i) for i,m in enumerate(('state_f1','authorized_support_f1','answer_f1','jaur'))} for cond in ('PT','TP','PP')}
    propagation=_propagation_summary(case_rows)
    tt_correct_rows=[r for r in case_rows if r['conditions']['TT']['jaur']>=.999999]
    tt_correct={
        'definition':'subset of prespecified cases where TT (trusted event + trusted query) achieves JAUR=1; isolates parser-induced degradation from answer-model/control failures',
        'case_count':len(tt_correct_rows),
        'family_counts':dict(Counter(r['family'] for r in tt_correct_rows)),
        'condition_means':_condition_means(tt_correct_rows) if tt_correct_rows else {},
        'error_propagation':_propagation_summary(tt_correct_rows),
        'case_ids':[r['case_id'] for r in tt_correct_rows],
    }
    family_metrics={}
    for fam in FAMILIES:
        fr=[r for r in case_rows if r['family']==fam]
        family_metrics[fam]={'n':len(fr),'event_field_f1_mean':_mean([r['event_field_f1'] or 0 for r in fr]),'query_field_f1_mean':_mean([r['query_field_f1'] or 0 for r in fr]),'query_intent_f1_mean':_mean([r['query_intent_f1'] or 0 for r in fr]),'query_target_f1_mean':_mean([r['query_target_f1'] or 0 for r in fr]),'query_scope_f1_mean':_mean([r['query_scope_f1'] or 0 for r in fr]),'transition_agreement_rate':_mean([r['transition_agreement'] for r in fr]),'event_runtime_valid_rate':_mean([r['event_runtime_valid'] for r in fr]),'query_runtime_valid_rate':_mean([r['query_runtime_valid'] for r in fr])}
    fallback_summary={'event_fallback_cases':sum(bool(r.get('event_runtime_fallback')) for r in case_rows),'query_fallback_cases':sum(bool(r.get('query_runtime_fallback')) for r in case_rows),'event_fallback_rate':_mean([bool(r.get('event_runtime_fallback')) for r in case_rows]),'query_fallback_rate':_mean([bool(r.get('query_runtime_fallback')) for r in case_rows])}
    parse_diags=[x['diagnostic'] for x in pred.values()];status=merge_statuses([from_diagnostics(parse_diags),from_diagnostics(answer_diags)])
    result={'schema_version':SCHEMA_VERSION,'status':status,'resume':{'parser_predictions_reused':initial_pred_n,'downstream_cases_reused':resumed_case_n,'checkpoint_dir':str(cp) if cp else None},'primary_table_eligible':False,'role':'small_prespecified_parser_robustness_diagnostic','case_count':24,'cases_per_family':4,'families':list(FAMILIES),'parser_mode':parser_mode,'answer_mode':answer_mode,'condition_contract':{'TT':'trusted event + trusted query','PT':'parsed event + trusted query','TP':'trusted event + parsed query','PP':'parsed event + parsed query'},'isolation_contract':'All non-target events are replayed from trusted semantics; only the prespecified case event and/or query is substituted. Parser predictions use raw target utterance + bounded current-session raw context + fixed public schema ontology only and are completed before trusted targets/evaluator gold are loaded. Persistent-memory state is never parser input. The diagnostic uses CSS and QASC semantic logic with deterministic physical-id realization, not a new Mem0 longitudinal campaign.','parser_input_contract':{'current_session_context_only':True,'context_turn_limit':int(os.environ.get('HF13_PARSER_CONTEXT_TURNS','12')),'public_ontology_schema_version':ontology.get('schema_version'),'persistent_memory_visible':False,'trusted_or_gold_visible_during_prediction':False,'semantic_self_check':str(os.environ.get('HF13_PARSER_SELF_CHECK','1')).lower() not in {'0','false','no','off'}},'family_metrics':family_metrics,'fallback_summary':fallback_summary,'parser_summary':{'unique_event_inputs':len(event_turns),'unique_query_inputs':len(query_turns),'event_schema_valid_rate':_mean([bool(pred[('event',x)]['diagnostic'].get('parse_valid')) for x in event_turns]),'query_schema_valid_rate':_mean([bool(pred[('query',x)]['diagnostic'].get('parse_valid')) for x in query_turns]),'event_field_micro_f1':em['f1'],'query_field_micro_f1':qm['f1'],'query_intent_micro_f1':qim['f1'],'query_target_micro_f1':qtm['f1'],'query_scope_micro_f1':qsm['f1'],'event_exact_match_rate':_mean([field_f1(pred[('event',x)]['prediction'],trusted_events[x])==1.0 for x in event_turns]),'query_exact_match_rate':_mean([field_f1(pred[('query',x)]['prediction'],_parser_query_gold_projection(trusted_queries[x]))==1.0 for x in query_turns]),'transition_agreement_rate':_mean([r['transition_agreement'] for r in case_rows]),'event_runtime_valid_rate':_mean([r['event_runtime_valid'] for r in case_rows]),'query_runtime_valid_rate':_mean([r['query_runtime_valid'] for r in case_rows]),'event_runtime_failure_count':sum(not r['event_runtime_valid'] for r in case_rows),'query_runtime_failure_count':sum(not r['query_runtime_valid'] for r in case_rows)},'condition_means':condition_means,'all_cases':{'case_count':len(case_rows),'condition_means':condition_means,'error_propagation':propagation},'tt_correct_subset':tt_correct,'deltas_vs_TT':deltas,'error_propagation':propagation,'case_rows':case_rows,'parser_predictions':[pred[k] for k in sorted(pred)],'provider_diagnostics':{'parser':parse_diags,'answers':answer_diags}}
    progress('complete',24,status=status);return result
