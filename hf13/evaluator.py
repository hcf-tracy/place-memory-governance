from __future__ import annotations
import json,copy
from collections import Counter
from pathlib import Path
from .dataset import EvaluatorData
from .io import read_jsonl,write_json
from .css import fact_signature,state_signature,_norm_semantic,canon
from .status import from_diagnostics
from .artifact_registry import RuntimeArtifactRegistry
from .oracle import OracleEngine
from .authorization import oracle_authorize
from .schema import validate_trusted_event,validate_trusted_query
from .evaluator_projection import EvaluatorPhysicalProjector

def f1_sets(pred,gold):
    p=Counter(pred);g=Counter(gold);np=sum(p.values());ng=sum(g.values())
    if not np and not ng:return 1.0
    if not np or not ng:return 0.0
    tp=sum((p&g).values());pr=tp/np;rc=tp/ng
    return 2*pr*rc/(pr+rc) if pr+rc else 0.0

def _reg(x):
    if x is None:return None
    if isinstance(x,RuntimeArtifactRegistry):return x
    if isinstance(x,(str,Path)):return RuntimeArtifactRegistry.load(x)
    if isinstance(x,dict):return RuntimeArtifactRegistry(x)
    raise TypeError(type(x))
def _resolve(x,reg):return reg.resolve_object(x) if reg else x
def _semkey(f):return fact_signature(f)
def _slot(f):return (str(f.get('entity')),str(f.get('predicate')),canon(f.get('scope') or {}))
def _value_variants(v):
    if isinstance(v,dict) and v.get('artifact_id'):
        vals=[str(v.get('artifact_id'))]
        if v.get('text'):vals.append(str(v['text']))
        return vals
    if isinstance(v,dict):return [str(x) for x in v.values() if not isinstance(x,(dict,list))]
    if isinstance(v,list):
        out=[]
        for x in v:out.extend(_value_variants(x))
        return out
    return [str(v)]
def _mentioned(text,v):
    a=str(text or '').casefold();vals=[x.casefold() for x in _value_variants(v) if len(str(x).strip())>=1]
    return bool(vals and any(x in a for x in vals))
def _support_text(q):return '\n'.join(str(x) for x in ((q or {}).get('model_visible_support') or []))
def _required_visible(q,expected):
    if not expected:return True
    s=_support_text(q);return all(_mentioned(s,g.get('value')) for g in expected)
def _forbidden_mentions(text,ineligible):return [g for g in ineligible if _mentioned(text,g.get('value'))]
def _entity_surface(entity):return str(entity or '').replace('_',' ').casefold().strip()
def _forbidden_answer_mentions(text,ineligible,raw_query=''):
    """Detect forbidden answer disclosure without penalizing the user's query.

    If a forbidden value was supplied verbatim by the current user query, merely
    repeating or negating that value is not memory leakage. A new entity--value
    association can still count when the entity itself was not present in the
    query and is introduced by the answer.
    """
    out=[];qt=str(raw_query or '');qlo=qt.casefold();alo=str(text or '').casefold()
    for g in ineligible:
        if not _mentioned(text,g.get('value')):continue
        if _mentioned(qt,g.get('value')):
            ent=_entity_surface(g.get('entity'))
            if ent and ent not in qlo and ent in alo:out.append(g)
            continue
        out.append(g)
    return out

def _answer_contract(query,expected):
    if expected:return 'fact_values'
    qi=(query or {}).get('query_intent') or {}
    if qi.get('requires_clarification'):return 'clarification_required'
    if qi.get('request_sensitive_value'):return 'sensitive_unavailable'
    if qi.get('check_archival_retention'):return 'retention_exclusion'
    return 'negative_inactive_or_unknown'

def _has_any(text,phrases):
    low=str(text or '').casefold();return any(p in low for p in phrases)
def _authorized_contract(actual,expected,ineligible):
    A=Counter(_semkey(x) for x in actual);E=Counter(_semkey(x) for x in expected);I={_semkey(x) for x in ineligible}
    required_ok=all(A[k]>=n for k,n in E.items());forbidden_ok=not any(A[k]>0 for k in I)
    return bool(required_ok and forbidden_ok)

def _target_related(f,q):
    target=q.get('target') or {};selectors=(q.get('query_intent') or {}).get('selectors') or []
    if selectors:
        ok=False
        for sel in selectors:
            if sel.get('entity') and f.get('entity')!=sel['entity']:continue
            if sel.get('predicate') and f.get('predicate')!=sel['predicate']:continue
            ok=True;break
        if not ok:return False
    if target.get('entity') and f.get('entity')!=target['entity']:return False
    if target.get('predicate') and f.get('predicate')!=target['predicate']:return False
    return True

def runtime_gold(data_root='data/longitudinal',artifact_registry=None):
    """Evaluator-only runtime-resolved hidden oracle.

    Static gold may contain binding_id placeholders. This function resolves trusted
    events against the actual shared RuntimeArtifactRegistry *inside the evaluator*
    and independently replays OracleEngine. Runtime components never receive it.
    """
    reg=_reg(artifact_registry);root=Path(data_root)
    if reg is None:
        ev=EvaluatorData(root);return ev.gold_state,ev.gold_query
    turns=read_jsonl(root/'trajectory_raw.jsonl');events=[reg.resolve_object(x) for x in read_jsonl(root/'trusted_events.jsonl')];queries=read_jsonl(root/'trusted_queries.jsonl')
    # Static files were validated before resolution. Runtime copies may legitimately
    # contain artifact_id values produced by RuntimeArtifactRegistry.
    [validate_trusted_query(x) for x in queries]
    eb={x['turn_id']:x for x in events};qb={x['turn_id']:x for x in queries};eng=OracleEngine();gs=[];gq={}
    for t in turns:
        eng.advance_time(t['timestamp']);q=qb.get(t['turn_id'])
        if q:
            pre=eng.snapshot();expected=oracle_authorize(pre,q);allowed={_semkey(x) for x in expected}
            related=[copy.deepcopy(f) for f in pre if _target_related(f,q)]
            # Purged facts remain evaluator-only tombstones: they are absent from
            # current state but remain forbidden disclosure targets. Runtime
            # components never see this evaluator history.
            related += [copy.deepcopy(f) for f in eng.purged_snapshot() if _target_related(f,q)]
            uniq={_semkey(f):f for f in related};inel=[f for k,f in uniq.items() if k not in allowed]
            gq[t['turn_id']]={'query_id':q['query_id'],'turn_id':q['turn_id'],'family':q['family'],'query_semantics':copy.deepcopy(q),'expected_facts':expected,'ineligible_facts':inel,'expected_state':pre,'required_answer_values':[copy.deepcopy(x.get('value')) for x in expected],'forbidden_answer_values':[copy.deepcopy(x.get('value')) for x in inel],'expect_abstain':len(expected)==0,'answer_contract':_answer_contract(q,expected)}
        e=eb.get(t['turn_id'])
        if e:
            eng.apply(e,as_of=t['timestamp']);gs.append({'checkpoint_id':f"state_{e['event_id']}",'after_turn_id':t['turn_id'],'event_id':e['event_id'],'facts':eng.snapshot()})
    return gs,gq

def _evaluator_semantic_state(row,condition,projector,*,through_turn_id=None,include_through=True):
    if condition in {'native','qasc'}:
        records=(row.get('physical_backend_records') or [])
        return projector.project(records,through_turn_id=through_turn_id or row.get('turn_id'),include_through=include_through)['facts'] if projector else []
    return row.get('physical_state') or row.get('physical_facts') or []

def state_scores(rows,gold_state,condition,projector=None):
    by={r['turn_id']:r for r in rows};out=[]
    for cp in gold_state:
        r=by[cp['after_turn_id']];pred=_evaluator_semantic_state(r,condition,projector,through_turn_id=cp['after_turn_id'],include_through=True);gold=cp.get('facts') or []
        out.append({'checkpoint_id':cp['checkpoint_id'],'turn_id':cp['after_turn_id'],'conversation_id':r.get('conversation_id'),'state_f1':f1_sets([_semkey(x) for x in pred],[_semkey(x) for x in gold]),'gold_count':len(gold),'pred_count':len(pred)})
    return out

def answer_score(response,expected,ineligible,expect_abstain=False,*,query=None,raw_query=''):
    # Forbidden disclosure is always disqualifying, including an answer that
    # combines an abstention phrase with a secret value.
    if _forbidden_answer_mentions(response,ineligible,raw_query):return 0.0
    if not expected:
        mode=_answer_contract(query,expected)
        if mode=='clarification_required':
            return 1.0 if _has_any(response,('clarif','which ','which one','specify','more detail','need confirmation','need you to','what information','what exactly','are you referring')) else 0.0
        if mode=='sensitive_unavailable':
            return 1.0 if _has_any(response,('insufficient_memory','insufficient memory','do not have',"don't have",'not stored','no longer stored','removed','deleted','forgot','forgotten','not retained','not available','need to enter','provide them again','provide the numbers again')) else 0.0
        if mode=='retention_exclusion':
            return 1.0 if _has_any(response,('not include','do not include',"won't include",'will not include','exclude','not retained','not stored','removed','deleted','not present')) else 0.0
        return 1.0 if _has_any(response,('insufficient_memory','insufficient memory','no ','not ','never','none','did not',"didn't",'does not',"doesn't",'cannot',"can't",'not authorized','not allowed','no longer','expired','unavailable','cancelled','canceled','completed','historical','hypothetical','not applicable','should not','would not','need confirmation','need your approval')) else 0.0
    tp=sum(_mentioned(response,g.get('value')) for g in expected);fp=len(_forbidden_answer_mentions(response,ineligible,raw_query));precision=tp/(tp+fp) if (tp+fp) else 0.0;recall=tp/len(expected)
    return 2*precision*recall/(precision+recall) if precision+recall else 0.0

def evaluate_rows(rows,data_root='data/longitudinal',artifact_registry=None):
    reg=_reg(artifact_registry);gold_state,gold_query=runtime_gold(data_root,reg);by={r['turn_id']:r for r in rows};condition=rows[0]['condition'] if rows else None;projector=EvaluatorPhysicalProjector(data_root,reg) if condition in {'native','qasc'} else None;ss=state_scores(rows,gold_state,condition,projector);qs=[]
    for tid,g in gold_query.items():
        r=by[tid];q=r.get('query') or {};physical=_evaluator_semantic_state({'turn_id':tid,'physical_backend_records':q.get('physical_backend_records') or [],'physical_state':q.get('physical_state') or []},condition,projector,through_turn_id=tid,include_through=False);expected=g.get('expected_facts') or [];ineligible=g.get('ineligible_facts') or [];expected_state=g.get('expected_state') or []
        query_state_f1=f1_sets([_semkey(x) for x in physical],[_semkey(x) for x in expected_state]);actual_auth=q.get('authorized_support') or []
        visible=_support_text(q);exposed=_forbidden_mentions(visible,ineligible);exposure=1.0 if exposed else 0.0;required_visible=_required_visible(q,expected);sra=1.0 if required_visible else 0.0
        if condition=='qasc':auth_contract=bool(required_visible and exposure==0.0)
        elif condition=='place':auth_contract=_authorized_contract(actual_auth,expected,ineligible)
        else:auth_contract=None
        response=q.get('response') if 'response' in q else q.get('answer');raw_query=r.get('raw_utterance') or '';af1=answer_score(response,expected,ineligible,g.get('expect_abstain',False),query=g.get('query_semantics'),raw_query=raw_query);gvr=1.0 if _forbidden_answer_mentions(response,ineligible,raw_query) else 0.0
        support_correct=bool(required_visible and exposure==0.0 and (auth_contract is not False))
        jaur=1.0 if query_state_f1>=.999999 and support_correct and sra==1.0 and gvr==0.0 and af1>=.999999 else 0.0
        qs.append({'turn_id':tid,'conversation_id':r.get('conversation_id'),'family':g.get('family'),'query_state_f1':query_state_f1,'query_contract':auth_contract,'unauthorized_support_exposure':exposure,'exposed_forbidden_count':len(exposed),'answer_f1':af1,'gvr':gvr,'needed_sra':sra,'support_correct':support_correct,'jaur':jaur,'expected_count':len(expected),'authorized_count':len(actual_auth),'model_visible_support_count':len(q.get('model_visible_support') or [])})
    fam={}
    for f in sorted({x['family'] for x in qs}):
        xs=[x for x in qs if x['family']==f];fam[f]={'n':len(xs),'query_state_f1':sum(x['query_state_f1'] for x in xs)/len(xs),'exposure':sum(x['unauthorized_support_exposure'] for x in xs)/len(xs),'gvr':sum(x['gvr'] for x in xs)/len(xs),'answer_f1':sum(x['answer_f1'] for x in xs)/len(xs),'needed_sra':sum(x['needed_sra'] for x in xs)/len(xs),'jaur':sum(x['jaur'] for x in xs)/len(xs),'query_contract_rate':(sum(bool(x['query_contract']) for x in xs)/len(xs) if condition in {'qasc','place'} else None)}
    diags=[d for r in rows for d in (r.get('provider_diagnostics') or [])];status=from_diagnostics(diags)
    recovered_turns=sum(any(d.get('status')=='recovered' for d in (r.get('provider_diagnostics') or [])) for r in rows);model_error_turns=sum(any(d.get('status')=='model_operation_error' for d in (r.get('provider_diagnostics') or [])) for r in rows);failed_turns=sum(str(r.get('turn_status') or '')=='FAILED' or any(d.get('status')=='failed' for d in (r.get('provider_diagnostics') or [])) for r in rows)
    return {'condition':condition,'row_count':len(rows),'query_count':len(qs),'state_checkpoint_count':len(ss),'state_f1_mean':sum(x['state_f1'] for x in ss)/len(ss),'query_state_f1_mean':sum(x['query_state_f1'] for x in qs)/len(qs),'unauthorized_support_exposure_rate':sum(x['unauthorized_support_exposure'] for x in qs)/len(qs),'gvr_mean':sum(x['gvr'] for x in qs)/len(qs),'answer_f1_mean':sum(x['answer_f1'] for x in qs)/len(qs),'needed_sra_mean':sum(x['needed_sra'] for x in qs)/len(qs),'jaur_mean':sum(x['jaur'] for x in qs)/len(qs),'query_contract_rate':(sum(bool(x['query_contract']) for x in qs)/len(qs) if condition in {'qasc','place'} else None),'family_metrics':fam,'state_checkpoints':ss,'queries':qs,'provider_recovered_error_count':sum(d.get('status')=='recovered' for d in diags),'provider_recovered_turn_count':recovered_turns,'model_operation_error_count':sum(d.get('status')=='model_operation_error' for d in diags),'model_operation_error_turn_count':model_error_turns,'failed_turn_count':failed_turns,'status':status}

def evaluate(result_path,data_root='data/longitudinal',output=None,artifact_registry=None):
    out=evaluate_rows(read_jsonl(result_path),data_root,artifact_registry=artifact_registry)
    if output:write_json(output,out)
    return out

def _physical_backend_signature(row):
    recs=row.get('physical_backend_records') or []
    return sorted((str(x.get('text') or x.get('memory') or ''),canon(x.get('metadata') or {})) for x in recs)

def full_backend_invariance(rows_by_condition):
    """Diagnostic-only equality of raw physical backend snapshots."""
    def by(rows):return {r['turn_id']:_physical_backend_signature(r) for r in rows}
    out={}
    for a,b,name in [('native','qasc','native_qasc'),('css','place','css_place')]:
        A=by(rows_by_condition[a]);B=by(rows_by_condition[b]);keys=sorted(set(A)|set(B));eq=[k in A and k in B and A[k]==B[k] for k in keys]
        out[name]={'compared_turns':len(keys),'equal_count':sum(eq),'equality_rate':sum(eq)/len(keys) if keys else None,'mismatches':[k for k,e in zip(keys,eq) if not e],'diagnostic_only':True,'basis':'raw_physical_backend_text_and_metadata'}
    return out

def observation_invariance(rows_by_condition):
    """Diagnostic-only equality of branch-local assistant/context ledgers."""
    def sig(r):return (str(r.get('assistant_response') or ''),str(r.get('observation_ledger_digest') or ''))
    def by(rows):return {r['turn_id']:sig(r) for r in rows}
    out={};applicable=any(bool(r.get('closed_loop')) for rows in rows_by_condition.values() for r in rows)
    for a,b,name in [('native','qasc','native_qasc'),('css','place','css_place')]:
        if not applicable:
            out[name]={'applicable':False,'compared_turns':0,'equal_count':None,'equality_rate':None,'mismatches':[],'diagnostic_only':True,'basis':'branch_local_non_authoritative_observation_context_ledger'};continue
        A=by(rows_by_condition[a]);B=by(rows_by_condition[b]);keys=sorted(set(A)|set(B));eq=[k in A and k in B and A[k]==B[k] for k in keys]
        out[name]={'applicable':True,'compared_turns':len(keys),'equal_count':sum(eq),'equality_rate':sum(eq)/len(keys) if keys else None,'mismatches':[k for k,e in zip(keys,eq) if not e],'diagnostic_only':True,'basis':'branch_local_non_authoritative_observation_context_ledger'}
    return out

def state_plane_invariance(rows_by_condition,data_root='data/longitudinal',artifact_registry=None):
    """Hard semantic state-plane equality using the same beta(X) as State F1.

    Exactly 105 post-event State checkpoint instances and 48 pre-query checkpoint
    instances are compared. A turn carrying both roles is intentionally counted
    twice because the pre-query and post-event states are different estimands.
    """
    root=Path(data_root);roles={x['turn_id']:set(x['turn_role']) for x in read_jsonl(root/'turn_roles.jsonl')};state_tids=[t for t,r in roles.items() if 'state_checkpoint' in r];query_tids=[t for t,r in roles.items() if 'query_checkpoint' in r]
    by={c:{r['turn_id']:r for r in rows_by_condition[c]} for c in rows_by_condition};reg=_reg(artifact_registry);projector=EvaluatorPhysicalProjector(root,reg)
    def sem(condition,tid,kind):
        r=by[condition][tid]
        if kind=='post_event_state':
            if condition in {'native','qasc'}:
                return projector.project(r.get('physical_backend_records') or [],through_turn_id=tid,include_through=True)['facts']
            return r.get('physical_state') or r.get('physical_facts') or []
        q=r.get('query') or {}
        if condition in {'native','qasc'}:
            return projector.project(q.get('physical_backend_records') or [],through_turn_id=tid,include_through=False)['facts']
        return q.get('physical_state') or []
    out={}
    for a,b,name in [('native','qasc','native_qasc'),('css','place','css_place')]:
        inst=[]
        for tid in state_tids:inst.append(('post_event_state',tid))
        for tid in query_tids:inst.append(('pre_query_state',tid))
        mism=[];missing=[];post_eq=0;pre_eq=0
        beta_projection_count=0;raw_identity_shortcuts=0
        for kind,tid in inst:
            if tid not in by[a] or tid not in by[b]:missing.append({'checkpoint_type':kind,'turn_id':tid});continue
            # For Native/QASC, exact shared producer snapshots imply equal beta(X)
            # by determinism. Only a raw snapshot divergence needs two expensive
            # evaluator projections. CSS/PLACE expose managed semantic state directly.
            if a=='native' and b=='qasc':
                ra=by[a][tid];rb=by[b][tid]
                rec_a=(ra.get('physical_backend_records') or []) if kind=='post_event_state' else ((ra.get('query') or {}).get('physical_backend_records') or [])
                rec_b=(rb.get('physical_backend_records') or []) if kind=='post_event_state' else ((rb.get('query') or {}).get('physical_backend_records') or [])
                raw_a=sorted((str(x.get('physical_id') or ''),str(x.get('text') or x.get('memory') or ''),canon(x.get('metadata') or {})) for x in rec_a)
                raw_b=sorted((str(x.get('physical_id') or ''),str(x.get('text') or x.get('memory') or ''),canon(x.get('metadata') or {})) for x in rec_b)
                if raw_a==raw_b:
                    eq=True;raw_identity_shortcuts+=1
                else:
                    eq=state_signature(sem(a,tid,kind))==state_signature(sem(b,tid,kind));beta_projection_count+=2
            else:
                eq=state_signature(sem(a,tid,kind))==state_signature(sem(b,tid,kind))
            if eq:
                if kind=='post_event_state':post_eq+=1
                else:pre_eq+=1
            else:mism.append({'checkpoint_type':kind,'turn_id':tid})
        compared=len(inst);equal=post_eq+pre_eq
        out[name]={'basis':'semantic_physical_state_beta_X_same_as_state_f1','post_event_state':{'compared_checkpoints':len(state_tids),'equal_count':post_eq,'equality_rate':post_eq/len(state_tids) if state_tids else None},'pre_query_state':{'compared_checkpoints':len(query_tids),'equal_count':pre_eq,'equality_rate':pre_eq/len(query_tids) if query_tids else None},'compared_checkpoint_instances':compared,'equal_count':equal,'equality_rate':equal/compared if compared else None,'mismatches':mism,'missing':missing,'hard_blocker':True,'beta_projection_count':beta_projection_count,'raw_shared_snapshot_identity_shortcuts':raw_identity_shortcuts}
    return {'schema_version':'hf13-semantic-state-plane-invariance-v2','expected_checkpoint_instances':{'post_event_state':len(state_tids),'pre_query_state':len(query_tids),'combined':len(state_tids)+len(query_tids)},**out}

def invariance(rows_by_condition,data_root='data/longitudinal',artifact_registry=None):
    """Backward-compatible alias for the semantic hard state-plane invariant."""
    return state_plane_invariance(rows_by_condition,data_root,artifact_registry)
