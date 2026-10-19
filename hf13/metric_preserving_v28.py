from __future__ import annotations
import copy,re
from collections import defaultdict
from .answer_evaluator_strict_general import scalar_match, norm_text, _sentences
from .semantic_measurement_v26 import SemanticEvidenceResolver, value_match
from .css import canon

VERSION='2.8-metric-preserving'


def _factsame(a,b):
    return (a.get('entity')==b.get('entity') and a.get('predicate')==b.get('predicate') and
            canon(a.get('scope') or {})==canon(b.get('scope') or {}) and canon(a.get('value'))==canon(b.get('value')))


def _explicit_authorized_item(item, authorized, resolver, forbidden=None):
    """True only when the *support item itself* is an explicitly authorized unit.

    This prevents a stale-looking scalar embedded in a separately authorized compound
    artifact from being counted as exposure merely by bag-of-text matching.  It does
    not suppress an item whose primary authorized fact is itself the forbidden fact.
    """
    ni=norm_text(item)
    for a in authorized or []:
        if isinstance(a,str):
            if norm_text(a)==ni:return True
            continue
        if not isinstance(a,dict):continue
        tx=a.get('text')
        if tx is not None and norm_text(tx)==ni:
            return True
        ent,pred=a.get('entity'),a.get('predicate')
        if ent and pred:
            prefix=norm_text(f'{ent}.{pred}')
            if ni.startswith(prefix) and resolver.fact_evidence(item,a):
                if forbidden is None or not _factsame(a,forbidden):
                    return True
    return False


def support_metrics_v28(resolver:SemanticEvidenceResolver,q,expected,ineligible):
    q=q or {};items=[str(x) for x in (q.get('model_visible_support') or [])];authorized=q.get('authorized_support') or []
    req_hits=[any(resolver.fact_evidence(item,f) for item in items) for f in expected]
    exposed=[];detail=[]
    for f in ineligible:
        raw_hits=[];counted=[];suppressed=[]
        for i,item in enumerate(items):
            if not resolver.fact_evidence(item,f):continue
            raw_hits.append(i)
            if _explicit_authorized_item(item,authorized,resolver,forbidden=f):suppressed.append(i)
            else:counted.append(i)
        if counted:
            exposed.append(f);detail.append({'entity':f.get('entity'),'predicate':f.get('predicate'),'value':f.get('value'),'support_item_indexes':counted,'raw_match_indexes':raw_hits,'authorized_container_suppressed_indexes':suppressed})
    return {'needed_sra':1.0 if (not expected or all(req_hits)) else 0.0,'required_hits':req_hits,'exposure':1.0 if exposed else 0.0,'exposed_facts':exposed,'exposure_match_detail':detail,'support_items':len(items)}



_PAIR_FUNC={'the','a','an','is','are','was','were','to','for','of','and','at','now','currently'}
_TIME_RE=re.compile(r'^(?:[01]?\d|2[0-3]):[0-5]\d$')
_NUM_RE=re.compile(r'^[-+]?\d+(?:\.\d+)?$')


def _ordered_tokens(textn,tokens,max_gap=8):
    words=textn.split();pos=-1
    for j,tok in enumerate(tokens):
        found=None
        start=pos+1
        stop=len(words) if j==0 else min(len(words),start+max_gap)
        for i in range(start,stop):
            if words[i]==tok:found=i;break
        if found is None:return False
        pos=found
    return True


def pair_value_match(text,value):
    """Representation-aware match for the *same pair value*, preserving polarity/order."""
    if scalar_match(text,value):return True
    nv=norm_text(value);nt=norm_text(text);toks=nv.split()
    if not toks:return False
    # Numeric/time composites may naturally insert descriptive words between a
    # semantic label and the scalar ("arrival estimate is now at 17:00").
    scalar_idx=[i for i,t in enumerate(toks) if _TIME_RE.match(t) or _NUM_RE.match(t)]
    if scalar_idx:
        i=scalar_idx[-1];scalar=toks[i];labels=[t for t in toks[:i] if t not in _PAIR_FUNC]
        for sent in _sentences(text):
            if scalar_match(sent,scalar) and _ordered_tokens(norm_text(sent),labels,max_gap=7):return True
        return False
    # Non-numeric phrases retain order and only permit function-word insertion.
    # This prevents Plan-B from matching Plan-A and prevents a negated current
    # phrase from matching a positive historical phrase merely by token overlap.
    parts=[]
    for j,t in enumerate(toks):
        if j:parts.append(r'(?:\s+(?:the|a|an|is|are|was|were|to|for|of|and|at|now|currently)){0,3}\s+')
        parts.append(re.escape(t))
    return bool(re.search(r'(?<!\w)'+''.join(parts)+r'(?!\w)',nt))

_HISTORY_CUES=('earlier','previous','previously','old ','before','formerly','superseded','historical','used to')

def _old_value_match(answer,old,current):
    if not pair_value_match(answer,old):return False
    # Narrow polarity exception: a negative current value such as "no
    # restriction" naturally contains the positive topic noun phrase
    # "raw-seafood restriction".  Suppress that embedded topic mention unless
    # it is separately historical.  Do not apply this to ordinary A/B values
    # such as window-vs-aisle seats.
    if norm_text(current).startswith('no ') and pair_value_match(answer,current):
        seen=False
        for sent in _sentences(answer):
            if not pair_value_match(sent,old):continue
            seen=True
            if not pair_value_match(sent,current):return True
            ns=norm_text(sent)
            if any(c in ns for c in _HISTORY_CUES):return True
        if seen:return False
    return True

def pair_outcome(answer,pair):
    current=str(pair['gold_authorized'][0]['value']);iv=pair.get('pair_intervention') or {}
    olds=[str(x) for x in (iv.get('world_a'),iv.get('world_b')) if x is not None and str(x)!=current]
    labels=[]
    if pair_value_match(answer,current):labels.append('CURRENT')
    for v in sorted(set(olds)):
        if _old_value_match(answer,v,current):labels.append('OLD:'+v)
    na=norm_text(answer)
    if 'insufficient memory' in na or 'insufficient_memory' in str(answer).casefold():labels.append('INSUFFICIENT')
    return tuple(labels or ['OTHER'])


def rescore_pair_row(row,pair):
    r=copy.deepcopy(row);oa=pair_outcome(r.get('answer_a',''),pair);ob=pair_outcome(r.get('answer_b',''),pair)
    eligible=bool(r.get('eligible'));semdiv=oa!=ob;old=any(str(x).startswith('OLD:') for x in oa+ob)
    cur=pair['gold_authorized'][0]['value'];ca=pair_value_match(r.get('answer_a',''),cur);cb=pair_value_match(r.get('answer_b',''),cur)
    needed=float(r.get('needed_sra',0.0));rawdiv=bool(r.get('raw_answer_diverged'))
    r.update({'outcome_a':oa,'outcome_b':ob,'semantic_outcome_diverged':semdiv,'olr_event':bool(eligible and semdiv),'gvr_event':bool(old),'ipdr_event':bool(eligible and rawdiv and not semdiv),'jaur':1.0 if eligible and not semdiv and needed>=.999999 and ca and cb and not old else 0.0,'v28_current_correct_a':bool(ca),'v28_current_correct_b':bool(cb)})
    return r


def pair_summary(rows):
    fam={};groups=defaultdict(list)
    for r in rows:groups[r['family']].append(r)
    def sm(xs):
        n=len(xs);el=[x for x in xs if x.get('eligible')]
        return {'n':n,'eligible_n':len(el),'olr':sum(bool(x.get('olr_event')) for x in el)/len(el) if el else None,'state_mediated_n':sum(bool(x.get('state_mediated_divergence')) for x in xs),'equal_but_wrong_n':sum(bool(x.get('equal_but_wrong')) for x in xs),'ipdr_n':sum(bool(x.get('ipdr_event')) for x in xs),'needed_sra':sum(float(x.get('needed_sra',0)) for x in xs)/n if n else None,'gvr':sum(bool(x.get('gvr_event')) for x in xs)/n if n else None,'jaur':sum(float(x.get('jaur',0)) for x in xs)/n if n else None}
    for k,xs in groups.items():fam[k]=sm(xs)
    allm=sm(rows);allm['family_metrics']=fam;vals=[x['olr'] for x in fam.values() if x['olr'] is not None];allm['macro_family_olr']=sum(vals)/len(vals) if vals else None
    return allm
