from __future__ import annotations

"""Frozen-trace answer-level metric validation for HF13.

This module is intentionally post-hoc and read-only.  It never calls Mem0, CSS,
QASC, Ollama, or any write path.  All scores are computed from persisted Primary
traces/evaluations plus evaluator-only gold resolved through the campaign's
RuntimeArtifactRegistry.

Conventional text metrics use fixed *canonical semantic references* constructed
from gold required facts / answer contracts.  These are not human reference
answers, and outputs label them accordingly.  An optional human-reference JSONL
can be supplied by the CLI in future without changing the frozen experiment.
"""

import json, math, re, string, statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .evaluator import (
    runtime_gold, _mentioned, _forbidden_answer_mentions, answer_score,
)
from .io import read_jsonl

CONDITIONS = ("native", "css", "qasc", "place")

# ------------------------------ text normalization ---------------------------
_ARTICLES = re.compile(r"\b(a|an|the)\b", flags=re.I)
_WS = re.compile(r"\s+")


def normalize_answer(s: str) -> str:
    s = str(s or "").lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = _ARTICLES.sub(" ", s)
    return _WS.sub(" ", s).strip()


def _tokens(s: str) -> list[str]:
    n = normalize_answer(s)
    return n.split() if n else []


def normalized_em(pred: str, ref: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(ref))


def token_prf(pred: str, ref: str) -> tuple[float, float, float]:
    p, r = _tokens(pred), _tokens(ref)
    if not p and not r:
        return 1.0, 1.0, 1.0
    if not p or not r:
        return 0.0, 0.0, 0.0
    common = Counter(p) & Counter(r)
    tp = sum(common.values())
    pr, rc = tp / len(p), tp / len(r)
    f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
    return pr, rc, f1


def _lcs_len(a: list[str], b: list[str]) -> int:
    # O(min(n,m)) memory dynamic program.
    if len(a) < len(b):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def rouge_l_prf(pred: str, ref: str) -> tuple[float, float, float]:
    p, r = _tokens(pred), _tokens(ref)
    if not p and not r:
        return 1.0, 1.0, 1.0
    if not p or not r:
        return 0.0, 0.0, 0.0
    l = _lcs_len(p, r)
    pr, rc = l / len(p), l / len(r)
    f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
    return pr, rc, f1


def _char_ngrams(s: str, n: int) -> Counter:
    x = normalize_answer(s).replace(" ", "")
    if len(x) < n:
        return Counter()
    return Counter(x[i:i+n] for i in range(len(x)-n+1))


def chrf(pred: str, ref: str, max_n: int = 6, beta: float = 2.0) -> float:
    ps, rs = [], []
    for n in range(1, max_n + 1):
        p, r = _char_ngrams(pred, n), _char_ngrams(ref, n)
        np_, nr = sum(p.values()), sum(r.values())
        if np_ == 0 and nr == 0:
            ps.append(1.0); rs.append(1.0); continue
        if np_ == 0 or nr == 0:
            ps.append(0.0); rs.append(0.0); continue
        tp = sum((p & r).values())
        ps.append(tp / np_); rs.append(tp / nr)
    pr, rc = float(np.mean(ps)), float(np.mean(rs))
    b2 = beta * beta
    return (1+b2) * pr * rc / (b2 * pr + rc) if (b2 * pr + rc) else 0.0


def _word_ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)) if len(tokens) >= n else Counter()


def sentence_bleu4(pred: str, ref: str) -> float:
    """Supplementary BLEU-4 with add-one smoothing for short QA answers."""
    p, r = _tokens(pred), _tokens(ref)
    if not p:
        return 1.0 if not r else 0.0
    logs = []
    for n in range(1, 5):
        pg, rg = _word_ngrams(p, n), _word_ngrams(r, n)
        matches = sum((pg & rg).values())
        total = sum(pg.values())
        # Add-one smoothing is explicit and deterministic.
        logs.append(math.log((matches + 1.0) / (total + 1.0)))
    bp = 1.0 if len(p) > len(r) else math.exp(1.0 - (len(r) / max(len(p), 1)))
    return float(bp * math.exp(sum(logs) / 4.0))


# --------------------------- canonical gold references -----------------------
def _value_text(v: Any) -> str:
    if isinstance(v, dict) and v.get("text"):
        return str(v["text"])
    if isinstance(v, dict):
        return "; ".join(f"{str(k).replace('_',' ')} { _value_text(x) }" for k, x in sorted(v.items()))
    if isinstance(v, list):
        return ", ".join(_value_text(x) for x in v)
    return str(v).replace("_", " ")


def canonical_references(gold: dict) -> list[str]:
    expected = gold.get("expected_facts") or []
    if expected:
        values = [_value_text(f.get("value")) for f in expected]
        values_only = "; ".join(values)
        semantic = "; ".join(
            f"{str(f.get('entity') or '').replace('_',' ')} {str(f.get('predicate') or '').replace('_',' ')}: {_value_text(f.get('value'))}"
            for f in expected
        )
        return list(dict.fromkeys([values_only, semantic]))
    contract = gold.get("answer_contract") or "negative_inactive_or_unknown"
    refs = {
        "clarification_required": [
            "I need clarification.", "Please clarify which item you mean.",
        ],
        "sensitive_unavailable": [
            "I do not have that information.",
            "The sensitive information is no longer stored.",
            "That information has been removed.",
        ],
        "retention_exclusion": [
            "I will not retain or include that information.",
            "That information is excluded from the archive.",
        ],
        "negative_inactive_or_unknown": [
            "No.", "There is no current authorized fact.",
            "I do not have a current applicable fact.",
        ],
    }
    return refs.get(contract, refs["negative_inactive_or_unknown"])


def _best_over_refs(pred: str, refs: list[str], scorer):
    vals = [scorer(pred, r) for r in refs]
    if not vals:
        return None
    if isinstance(vals[0], tuple):
        # select reference maximizing F1 (last component), retaining its P/R.
        return max(vals, key=lambda x: x[-1])
    return max(vals)


# -------------------------- benchmark-aware diagnostics ----------------------
_ABSTAIN_CUES = (
    "insufficient memory", "insufficient_memory", "do not have", "don't have",
    "not stored", "no longer stored", "removed", "deleted", "forgot", "forgotten",
    "not retained", "not available", "need clarification", "please clarify", "which ",
    "specify", "no ", "not ", "never", "none", "did not", "didn't", "does not",
    "doesn't", "cannot", "can't", "not authorized", "not allowed", "no longer",
    "expired", "unavailable", "cancelled", "canceled", "completed", "historical",
    "hypothetical", "not applicable", "should not", "would not", "need confirmation",
    "need your approval", "will not include", "won't include", "exclude",
)


def predicted_abstention(response: str, expected: list[dict]) -> bool:
    low = str(response or "").casefold()
    cue = any(x in low for x in _ABSTAIN_CUES)
    has_required = any(_mentioned(response, f.get("value")) for f in expected)
    return bool(cue and not has_required)


def semantic_value_metrics(response: str, gold: dict, raw_query: str) -> dict:
    expected = gold.get("expected_facts") or []
    ineligible = gold.get("ineligible_facts") or []
    forbidden = _forbidden_answer_mentions(response, ineligible, raw_query)
    if not expected:
        return {
            "answer_value_em": None,
            "answer_value_precision": None,
            "answer_value_recall": None,
            "answer_value_f1": None,
        }
    tp = sum(bool(_mentioned(response, f.get("value"))) for f in expected)
    fp = len(forbidden)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / len(expected)
    f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0
    em = float(tp == len(expected) and fp == 0)
    return {
        "answer_value_em": em,
        "answer_value_precision": precision,
        "answer_value_recall": recall,
        "answer_value_f1": f1,
    }


def _support_text(q: dict) -> str:
    return "\n".join(str(x) for x in ((q or {}).get("model_visible_support") or []))


def grounding_metrics(response: str, q: dict, gold: dict) -> dict:
    expected = gold.get("expected_facts") or []
    ineligible = gold.get("ineligible_facts") or []
    support = _support_text(q)
    universe = expected + ineligible
    mentioned = [f for f in universe if _mentioned(response, f.get("value"))]
    grounded = [f for f in mentioned if _mentioned(support, f.get("value"))]
    if mentioned:
        grounded_rate = len(grounded) / len(mentioned)
        grounded_answer = float(len(grounded) == len(mentioned))
    else:
        grounded_rate = 1.0 if not expected and predicted_abstention(response, expected) else None
        grounded_answer = grounded_rate

    # Fixed evaluator-defined denominator: every condition is scored against the
    # same gold-required facts for a query.  This prevents a condition from looking
    # artificially strong merely because it exposed fewer required facts.
    required_n = len(expected)
    visible_required = [f for f in expected if _mentioned(support, f.get("value"))]
    used_required = [f for f in expected if _mentioned(support, f.get("value")) and _mentioned(response, f.get("value"))]
    visible_n = len(visible_required); used_n = len(used_required)
    visibility_recall = (visible_n / required_n) if required_n else None
    support_to_answer_recall = (used_n / required_n) if required_n else None
    full_required = float(used_n == required_n) if required_n else None
    conditional_visible_use = (used_n / visible_n) if visible_n else None
    return {
        "benchmark_grounded_answer": grounded_answer,
        "benchmark_grounded_value_rate": grounded_rate,
        "gold_required_support_count": required_n,
        "visible_required_count": visible_n,
        "used_visible_required_count": used_n,
        "required_support_visibility_recall": visibility_recall,
        "required_support_to_answer_recall": support_to_answer_recall,
        "full_required_support_to_answer": full_required,
        # Retained only as a diagnostic of answer-model use conditional on retrieval.
        # It is not used for between-condition claims because its denominator varies.
        "conditional_visible_support_utilization": conditional_visible_use,
    }


# ------------------------------ standard statistics --------------------------
def _rankdata(x: list[float]) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    i = 0
    while i < len(a):
        j = i + 1
        while j < len(a) and a[order[j]] == a[order[i]]:
            j += 1
        rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = rank
        i = j
    return ranks


def pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2:
        return None
    a, b = np.asarray(x, float), np.asarray(y, float)
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2:
        return None
    return pearson(_rankdata(x).tolist(), _rankdata(y).tolist())


def _percentile(v: list[float], q: float) -> float | None:
    return float(np.percentile(v, q)) if v else None


def block_bootstrap_corr(rows: list[dict], xk: str, yk: str, *, method: str, reps: int = 2000, seed: int = 42) -> dict:
    valid = [r for r in rows if r.get(xk) is not None and r.get(yk) is not None]
    fn = spearman if method == "spearman" else pearson
    point = fn([r[xk] for r in valid], [r[yk] for r in valid])
    blocks = sorted({r.get("conversation_id") for r in valid if r.get("conversation_id") is not None})
    rng = np.random.default_rng(seed); vals=[]
    if len(blocks) >= 2:
        by = {b:[r for r in valid if r.get("conversation_id")==b] for b in blocks}
        for _ in range(reps):
            chosen = rng.choice(blocks, size=len(blocks), replace=True)
            sample=[]
            for b in chosen: sample.extend(by[b])
            z=fn([r[xk] for r in sample],[r[yk] for r in sample])
            if z is not None and math.isfinite(z): vals.append(z)
    return {
        "estimate": point, "n": len(valid), "cluster_count": len(blocks),
        "interval95": [_percentile(vals,2.5), _percentile(vals,97.5)],
        "interval_type": "conversation_block_bootstrap_sensitivity",
        "bootstrap_reps_requested": reps, "bootstrap_reps_valid": len(vals),
    }


def roc_auc(y: list[int], s: list[float]) -> float | None:
    if len(set(y)) < 2:
        return None
    pos=[v for v,t in zip(s,y) if t==1];neg=[v for v,t in zip(s,y) if t==0]
    wins=0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p>n else (0.5 if p==n else 0.0)
    return wins/(len(pos)*len(neg))


def average_precision(y: list[int], s: list[float]) -> float | None:
    npos=sum(y)
    if npos==0:return None
    order=sorted(range(len(y)),key=lambda i:s[i],reverse=True)
    hit=0;total=0.0
    for rank,i in enumerate(order,1):
        if y[i]:
            hit+=1;total+=hit/rank
    return total/npos


def _sigmoid(z):
    z=np.clip(z,-35,35);return 1.0/(1.0+np.exp(-z))


def fit_logit(X: np.ndarray, y: np.ndarray, max_iter: int = 100, ridge: float = 1e-8) -> dict:
    X=np.asarray(X,float);y=np.asarray(y,float)
    X1=np.column_stack([np.ones(len(X)),X]);b=np.zeros(X1.shape[1])
    for _ in range(max_iter):
        p=_sigmoid(X1@b);w=np.clip(p*(1-p),1e-8,None)
        grad=X1.T@(y-p)-ridge*np.r_[0.0,b[1:]]
        H=-(X1.T@(X1*w[:,None]))-ridge*np.diag(np.r_[0.0,np.ones(len(b)-1)])
        try:step=np.linalg.solve(H,grad)
        except np.linalg.LinAlgError:step=np.linalg.pinv(H)@grad
        nb=b-step
        if np.max(np.abs(nb-b))<1e-8:b=nb;break
        b=nb
    p=np.clip(_sigmoid(X1@b),1e-12,1-1e-12)
    ll=float(np.sum(y*np.log(p)+(1-y)*np.log(1-p)))
    return {"coef":b,"ll":ll,"pred":p}


def predict_logit(coef: np.ndarray, X: np.ndarray) -> np.ndarray:
    return _sigmoid(np.column_stack([np.ones(len(X)),X])@coef)


def _null_ll(y: np.ndarray) -> float:
    p=float(np.clip(np.mean(y),1e-12,1-1e-12))
    return float(np.sum(y*np.log(p)+(1-y)*np.log(1-p)))


def _chi2_sf(x: float, df: int) -> float | None:
    try:
        from scipy.stats import chi2
        return float(chi2.sf(x,df))
    except Exception:
        return None


def incremental_validity(rows: list[dict], *, semantic_key: str = "bertscore_f1") -> dict:
    # Incremental validity requires an criterion independent of HF13's own semantic
    # evaluator.  We therefore require blinded human correctness labels here.
    valid=[r for r in rows if r.get("human_answer_correct") is not None and r.get(semantic_key) is not None]
    if not valid:
        return {"status":"NOT_AVAILABLE","reason":f"Independent human correctness labels and {semantic_key} are required. Canonical-gold runs are descriptive only; Answer Value EM is not used as an external criterion because it shares HF13 semantic gold."}
    specs={
        "M1_semantic_only":[semantic_key],
        "M2_semantic_plus_state_exposure":[semantic_key,"query_state_f1","unauthorized_support_exposure"],
        "M3_semantic_plus_jaur":[semantic_key,"jaur"],
    }
    y=np.asarray([int(bool(r["human_answer_correct"])) for r in valid],int)
    if len(set(y.tolist()))<2:
        return {"status":"NOT_AVAILABLE","reason":"criterion has one class"}
    full={};cv={}
    for name,keys in specs.items():
        X=np.asarray([[float(r[k]) for k in keys] for r in valid],float)
        mu=X.mean(0);sd=X.std(0);sd=np.where(sd==0,1,sd);Xs=(X-mu)/sd
        fit=fit_logit(Xs,y);k=Xs.shape[1]+1;ll0=_null_ll(y)
        full[name]={
            "predictors":keys,"n":len(y),"log_likelihood":fit["ll"],
            "mcfadden_pseudo_r2":1-fit["ll"]/ll0 if ll0 else None,
            "AIC":2*k-2*fit["ll"],"BIC":k*math.log(len(y))-2*fit["ll"],
            "AUROC_apparent":roc_auc(y.tolist(),fit["pred"].tolist()),
            "AUPRC_apparent":average_precision(y.tolist(),fit["pred"].tolist()),
        }
        # Leave-one-conversation-out CV prevents query rows in the same session
        # from appearing in both training and validation folds.
        conv=sorted({r["conversation_id"] for r in valid});yp=[];sp=[]
        for hold in conv:
            tr=[i for i,r in enumerate(valid) if r["conversation_id"]!=hold]
            te=[i for i,r in enumerate(valid) if r["conversation_id"]==hold]
            if not tr or not te or len(set(y[tr].tolist()))<2:continue
            Xm=X[tr].mean(0);Xsdev=X[tr].std(0);Xsdev=np.where(Xsdev==0,1,Xsdev)
            fittr=fit_logit((X[tr]-Xm)/Xsdev,y[tr])
            pp=predict_logit(fittr["coef"],(X[te]-Xm)/Xsdev)
            yp.extend(y[te].tolist());sp.extend(pp.tolist())
        cv[name]={"fold_unit":"conversation","folds":len(conv),"n_predicted":len(yp),"AUROC":roc_auc(yp,sp),"AUPRC":average_precision(yp,sp)}
    base=full["M1_semantic_only"]
    for name in ("M2_semantic_plus_state_exposure","M3_semantic_plus_jaur"):
        alt=full[name];lr=max(0.0,2*(alt["log_likelihood"]-base["log_likelihood"]));df=len(alt["predictors"])-len(base["predictors"])
        alt["vs_M1"]={"delta_mcfadden_pseudo_r2":alt["mcfadden_pseudo_r2"]-base["mcfadden_pseudo_r2"],"delta_AIC":alt["AIC"]-base["AIC"],"delta_BIC":alt["BIC"]-base["BIC"],"likelihood_ratio_statistic":lr,"df":df,"p":_chi2_sf(lr,df) if df>0 else None,"nested_comparison":True}
    return {"status":"COMPLETE","criterion":"blinded human answer correctness","semantic_predictor":semantic_key,"full_data_fit":full,"leave_one_conversation_out":cv}


def paired_effect(rows: list[dict], metric: str, a: str, b: str, *, reps=2000, seed=42) -> dict:
    # Effect is a-b, paired on identical model/query.
    by={(r["condition"],r["turn_id"]):r for r in rows}
    tids=sorted(set(r["turn_id"] for r in rows if r["condition"]==a)&set(r["turn_id"] for r in rows if r["condition"]==b))
    pairs=[]
    for t in tids:
        ra,rb=by[(a,t)],by[(b,t)]
        if ra.get(metric) is None or rb.get(metric) is None:continue
        pairs.append((t,ra["conversation_id"],float(ra[metric])-float(rb[metric])))
    point=float(np.mean([x[2] for x in pairs])) if pairs else None
    blocks=sorted({x[1] for x in pairs});byb={z:[x for x in pairs if x[1]==z] for z in blocks};rng=np.random.default_rng(seed);vals=[]
    if blocks:
        for _ in range(reps):
            chosen=rng.choice(blocks,size=len(blocks),replace=True);sample=[]
            for z in chosen:sample.extend(byb[z])
            if sample:vals.append(float(np.mean([x[2] for x in sample])))
    return {"contrast":f"{a.upper()} - {b.upper()}","metric":metric,"estimate":point,"paired_query_n":len(pairs),"cluster_count":len(blocks),"interval95":[_percentile(vals,2.5),_percentile(vals,97.5)],"interval_type":"paired_conversation_block_bootstrap_sensitivity"}


# -------------------------------- data loading -------------------------------
def load_model_rows(model_root: str | Path, data_root: str | Path, external_references: dict|None=None, human_labels: dict|None=None) -> tuple[str,list[dict],dict]:
    root=Path(model_root);model=root.name.lower();reg=root/'runtime_artifacts/runtime_artifact_registry.json'
    if not reg.exists():raise FileNotFoundError(f"missing RuntimeArtifactRegistry: {reg}")
    _,gq=runtime_gold(data_root,reg)
    out=[]
    for cond in CONDITIONS:
        tp=root/'primary'/cond/'trace.jsonl';ep=root/'primary'/cond/'evaluation.json'
        if not tp.exists() or not ep.exists():raise FileNotFoundError(f"missing frozen Primary outputs for {cond}: {tp} / {ep}")
        traces={r['turn_id']:r for r in read_jsonl(tp)};ev=json.load(open(ep));eq={q['turn_id']:q for q in ev['queries']}
        for tid,gold in gq.items():
            if tid not in traces or tid not in eq:continue
            tr=traces[tid];q=tr.get('query') or {};resp=str(q.get('response') or tr.get('assistant_response') or '')
            ext=(external_references or {}).get((model,tid))
            refs=list(ext) if ext else canonical_references(gold)
            tpr=_best_over_refs(resp,refs,token_prf);rpr=_best_over_refs(resp,refs,rouge_l_prf)
            row={
                "model":model,"condition":cond,"turn_id":tid,"conversation_id":tr.get('conversation_id'),
                "family":eq[tid].get('family'),"raw_query":tr.get('raw_utterance'),"response":resp,
                "canonical_references":refs,"canonical_reference_type":("external_reference_answers" if ext else "evaluator_gold_semantic_templates_not_human_references"),
                "normalized_em":_best_over_refs(resp,refs,normalized_em),
                "token_precision":tpr[0],"token_recall":tpr[1],"token_f1":tpr[2],
                "rouge_l_precision":rpr[0],"rouge_l_recall":rpr[1],"rouge_l_f1":rpr[2],
                "chrf":_best_over_refs(resp,refs,chrf),"bleu4":_best_over_refs(resp,refs,sentence_bleu4),
                "bertscore_precision":None,"bertscore_recall":None,"bertscore_f1":None,
                "our_answer_f1":eq[tid]['answer_f1'],"jaur":eq[tid]['jaur'],"query_state_f1":eq[tid]['query_state_f1'],
                "unauthorized_support_exposure":eq[tid]['unauthorized_support_exposure'],"gvr":eq[tid]['gvr'],
                "gold_answerable":bool(gold.get('expected_facts')),
                "predicted_abstention":predicted_abstention(resp,gold.get('expected_facts') or []),
                "answer_contract":gold.get('answer_contract'),
            }
            hl=(human_labels or {}).get((model,cond,tid),{})
            row['human_answer_correct']=hl.get('answer_correct')
            row['human_governance_appropriate']=hl.get('governance_appropriate')
            if row['human_answer_correct'] is not None and row['human_governance_appropriate'] is not None:
                row['human_joint_correct_authorized']=float(bool(row['human_answer_correct']) and bool(row['human_governance_appropriate']))
            else: row['human_joint_correct_authorized']=None
            row.update(semantic_value_metrics(resp,gold,tr.get('raw_utterance') or ''))
            row.update(grounding_metrics(resp,q,gold))
            out.append(row)
    return model,out,{"runtime_artifact_registry":str(reg),"query_gold_n":len(gq)}


def add_bertscore(rows: list[dict], *, model_type: str="roberta-large", device: str|None=None, batch_size:int=16) -> dict:
    try:
        from bert_score import score as bert_score
    except Exception as e:
        return {"status":"NOT_AVAILABLE","reason":f"bert_score import failed: {type(e).__name__}: {e}","model_type":model_type}
    # Score every candidate-reference combination, then select the reference with
    # maximal BERTScore F1 for that candidate (standard multi-reference logic).
    cands=[];refs=[];owners=[]
    for i,r in enumerate(rows):
        for ref in r['canonical_references']:
            cands.append(r['response']);refs.append(ref);owners.append(i)
    try:
        P,R,F=bert_score(cands,refs,model_type=model_type,device=device,batch_size=batch_size,verbose=False,rescale_with_baseline=False)
        best={}
        for j,i in enumerate(owners):
            z=(float(P[j]),float(R[j]),float(F[j]))
            if i not in best or z[2]>best[i][2]:best[i]=z
        for i,z in best.items():
            rows[i]['bertscore_precision'],rows[i]['bertscore_recall'],rows[i]['bertscore_f1']=z
        return {"status":"COMPLETE","model_type":model_type,"device":device,"scored_candidate_reference_pairs":len(cands),"rows":len(rows)}
    except Exception as e:
        return {"status":"FAILED","reason":f"BERTScore execution failed: {type(e).__name__}: {e}","model_type":model_type,"device":device}


def abstention_summary(rows:list[dict])->dict:
    # Positive class = query should not disclose/return a positive memory fact.
    y=[not r['gold_answerable'] for r in rows];p=[r['predicted_abstention'] for r in rows]
    tp=sum(a and b for a,b in zip(y,p));fp=sum((not a) and b for a,b in zip(y,p));fn=sum(a and (not b) for a,b in zip(y,p));tn=sum((not a) and (not b) for a,b in zip(y,p))
    pr=tp/(tp+fp) if tp+fp else 0.0;rc=tp/(tp+fn) if tp+fn else 0.0;f1=2*pr*rc/(pr+rc) if pr+rc else 0.0
    acc=(tp+tn)/len(y) if y else None;tpr=rc;tnr=tn/(tn+fp) if tn+fp else 0.0
    return {"positive_class":"should_abstain_or_return_no_positive_memory_fact","n":len(y),"tp":tp,"fp":fp,"fn":fn,"tn":tn,"precision":pr,"recall":rc,"f1":f1,"accuracy":acc,"balanced_accuracy":(tpr+tnr)/2}


def aggregate_rows(rows:list[dict])->dict:
    keys=["token_precision","token_recall","token_f1","rouge_l_f1","chrf","bleu4","bertscore_f1","answer_value_em","answer_value_precision","answer_value_recall","answer_value_f1","benchmark_grounded_answer","benchmark_grounded_value_rate","required_support_visibility_recall","required_support_to_answer_recall","full_required_support_to_answer","conditional_visible_support_utilization","our_answer_f1","jaur"]
    out={}
    for cond in CONDITIONS:
        xs=[r for r in rows if r['condition']==cond];m={}
        for k in keys:
            vs=[float(r[k]) for r in xs if r.get(k) is not None]
            m[k]={"mean":float(np.mean(vs)) if vs else None,"n":len(vs)}
        m['abstention']=abstention_summary(xs);out[cond]=m
    return out


def validation_statistics(rows:list[dict], *, bootstrap_reps:int=2000)->dict:
    conventional=["token_f1","rouge_l_f1","bertscore_f1"]
    corr={}
    for ours in ("our_answer_f1","jaur"):
        for other in conventional:
            corr[f"{ours}__vs__{other}"]={
                "spearman":block_bootstrap_corr(rows,ours,other,method='spearman',reps=bootstrap_reps),
                "pearson":block_bootstrap_corr(rows,ours,other,method='pearson',reps=bootstrap_reps),
            }
    # Criterion validity is external only when blinded human labels exist.
    human=[r for r in rows if r.get('human_answer_correct') is not None]
    if human and len(set(int(bool(r['human_answer_correct'])) for r in human))>1:
        hy=[int(bool(r['human_answer_correct'])) for r in human]
        criterion={
            "our_answer_f1_to_human_correct":{"n":len(human),"AUROC":roc_auc(hy,[r['our_answer_f1'] for r in human]),"AUPRC":average_precision(hy,[r['our_answer_f1'] for r in human])},
        }
        joint=[r for r in human if r.get('human_joint_correct_authorized') is not None]
        jy=[int(bool(r['human_joint_correct_authorized'])) for r in joint]
        criterion['jaur_to_human_joint_correct_authorized']={"n":len(joint),"AUROC":roc_auc(jy,[r['jaur'] for r in joint]) if len(set(jy))>1 else None,"AUPRC":average_precision(jy,[r['jaur'] for r in joint]) if len(set(jy))>1 else None}
        criterion['status']='EXTERNAL_HUMAN_CRITERION'
    else:
        criterion={"status":"NOT_AVAILABLE","reason":"No independent blinded human labels supplied. Answer Value EM is reported as an internal factual diagnostic but is not used to claim external criterion validity because it shares evaluator gold with HF13 Answer F1."}
    effects={}
    for metric in ("our_answer_f1","token_f1","rouge_l_f1","bertscore_f1"):
        effects[metric]=[
            paired_effect(rows,metric,'place','native',reps=bootstrap_reps),
            paired_effect(rows,metric,'css','native',reps=bootstrap_reps),
            paired_effect(rows,metric,'place','css',reps=bootstrap_reps),
            paired_effect(rows,metric,'qasc','native',reps=bootstrap_reps),
        ]
    return {"convergent_validity":corr,"criterion_validity":criterion,"effect_size_agreement":effects,"incremental_validity":incremental_validity(rows)}
