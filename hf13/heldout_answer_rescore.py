from __future__ import annotations

"""Read-only equivalence-aware answer rescoring for frozen naturalistic held-outs.

This module deliberately does *not* replace the preregistered HF13 evaluator.
It produces a versioned post-hoc audit over immutable traces.  Original
Answer F1 / JAUR remain preserved and are always emitted next to rescored values.

The hard rescoring path is deterministic and high precision:
- identifier formatting: dry_humor <-> "dry humor"
- natural temporal wording: Friday 21:30 <-> "Friday at 21:30"
- numeric surface forms: 2 <-> "two"
- structured values: every scalar leaf must be expressed; one leaf is not enough
- a small ontology-level alias map for status values such as not_authorized

BERTScore and lexical metrics are supplementary only; they never flip hard
correctness labels.
"""

import json, math, re, unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .answer_validation import (
    canonical_references, token_prf, rouge_l_prf, chrf, sentence_bleu4,
    _best_over_refs, add_bertscore,
)
from .evaluator import runtime_gold, answer_score
from .io import read_jsonl

CONDITIONS = ("native", "css", "qasc", "place")

# Only representation-level function words are removed.  Negation is NEVER
# discarded because it changes semantics.
_FUNCTION_WORDS = {"a", "an", "the", "at", "on"}
_NUM_WORDS = {
    "0":"zero","1":"one","2":"two","3":"three","4":"four","5":"five",
    "6":"six","7":"seven","8":"eight","9":"nine","10":"ten",
    "11":"eleven","12":"twelve","13":"thirteen","14":"fourteen",
    "15":"fifteen","16":"sixteen","17":"seventeen","18":"eighteen",
    "19":"nineteen","20":"twenty",
}

# Public ontology-level equivalences, not case-specific answer patches.
# Aliases are intentionally conservative and can only match the scalar value
# they represent.  They do not consult trajectory gold beyond that scalar.
_ENUM_ALIASES = {
    "not authorized": (
        "not authorized", "not allowed", "no authorization", "no approval",
        "do not proceed", "don't proceed", "dont proceed", "should not proceed",
        "shouldn't proceed", "shouldnt proceed", "cannot proceed", "can't proceed",
        "cant proceed", "hold off", "still waiting", "wait before",
    ),
    "confirmed": ("confirmed", "verified", "definitely confirmed"),
    "available": ("available",),
}

_CONTRACT_CUES = {
    "clarification_required": (
        "need clarification", "please clarify", "which one", "which item",
        "specify", "could you provide", "can you provide", "more details",
        "more context", "not enough context", "need more context",
        "what exactly", "are you referring",
    ),
    "sensitive_unavailable": (
        "insufficient memory", "do not have", "don't have", "dont have",
        "not stored", "no longer stored", "removed", "deleted", "forgot",
        "forgotten", "not retained", "not available", "not retrievable",
        "provide it again", "provide them again", "need the link",
        "do you have the link",
    ),
    "retention_exclusion": (
        "no longer retrievable", "should no longer be retrievable", "not retrievable",
        "do not have", "don't have", "dont have", "not stored", "no longer stored",
        "removed", "deleted", "forgot", "forgotten", "not retained", "dropped",
        "dropped it", "no longer have",
    ),
    "negative_inactive_or_unknown": (
        "insufficient memory", "not confirmed", "nothing is confirmed",
        "nothing confirmed", "still rumor", "still a rumor", "just rumor",
        "do not know", "don't know", "dont know", "not enough information",
        "not enough context", "cannot determine", "can't determine", "cant determine",
        "not current", "no longer current", "not applicable", "not valid",
        "should not use", "shouldn't use", "shouldnt use", "not usable",
        "need to verify", "best to verify", "need confirmation",
    ),
}


def _nfkc(s: Any) -> str:
    return unicodedata.normalize("NFKC", str(s if s is not None else ""))


def normalize_surface(s: Any) -> str:
    """Canonical human-readable surface used only for value equivalence."""
    x = _nfkc(s).casefold()
    # Identifier separators are representation artifacts in trusted semantics.
    x = x.replace("_", " ")
    # Hyphens between alphanumerics are treated as separators for phrases, but
    # original exact matching is checked separately for filename/code values.
    x = re.sub(r"(?<=[a-z0-9])-+(?=[a-z0-9])", " ", x)
    # Normalize common currency spelling without discarding the numeric amount.
    x = x.replace("nt$", " nt ").replace("ntd", " nt ")
    x = re.sub(r"[^\w:]+", " ", x, flags=re.UNICODE)
    toks = [t for t in x.split() if t not in _FUNCTION_WORDS]
    return " ".join(toks)


def _token_seq(s: Any) -> list[str]:
    return normalize_surface(s).split()


def _contains_tokens(text_tokens: list[str], value_tokens: list[str]) -> bool:
    if not value_tokens:
        return False
    n = len(value_tokens)
    return any(text_tokens[i:i+n] == value_tokens for i in range(max(0, len(text_tokens)-n+1)))


def _scalar_variants(v: Any) -> list[str]:
    raw = _nfkc(v).strip()
    norm = normalize_surface(raw)
    out = [raw.casefold(), norm]
    # If the value is a small integer, allow the ordinary word form too.
    if raw in _NUM_WORDS:
        out.append(_NUM_WORDS[raw])
    # Currency amounts may naturally be said without the currency prefix when
    # the question/context already fixes currency.
    m = re.fullmatch(r"(?:nt\$|ntd\s*)?\s*(\d+(?:\.\d+)?)", raw.casefold())
    if m:
        out.append(m.group(1))
        if m.group(1) in _NUM_WORDS:
            out.append(_NUM_WORDS[m.group(1)])
    # Enum aliases are keyed on canonical human-readable form.
    out.extend(_ENUM_ALIASES.get(norm, ()))
    # De-duplicate while preserving deterministic order.
    return list(dict.fromkeys(x for x in out if x))


def scalar_value_match(text: str, value: Any, *, semantic_aliases: bool = False) -> bool:
    """High-precision deterministic equivalence for one scalar leaf.

    ``semantic_aliases=False`` is the primary representation-only rescore.
    The broader ontology aliases are available only for a separately labeled
    semantic-sensitivity diagnostic.
    """
    raw_text = _nfkc(text).casefold()
    norm_tokens = _token_seq(text)
    raw_value = _nfkc(value).strip().casefold()
    if raw_value and raw_value in raw_text:
        return True
    variants = _scalar_variants(value)
    if not semantic_aliases:
        aliases = set(_ENUM_ALIASES.get(normalize_surface(value), ()))
        variants = [x for x in variants if x not in aliases or normalize_surface(x) == normalize_surface(value)]
    for variant in variants:
        vt = _token_seq(variant)
        if vt and _contains_tokens(norm_tokens, vt):
            return True
    return False


def _leaf_values(v: Any) -> list[Any]:
    if isinstance(v, dict):
        # Runtime artifact IDs are representation metadata; when text is present,
        # text is the semantic value users can verbalize.
        if v.get("text") is not None:
            return [v["text"]]
        out: list[Any] = []
        for _, x in sorted(v.items()):
            out.extend(_leaf_values(x))
        return out
    if isinstance(v, list):
        out: list[Any] = []
        for x in v:
            out.extend(_leaf_values(x))
        return out
    return [v]


def semantic_value_match(text: str, value: Any, *, semantic_aliases: bool = False) -> bool:
    """Match a value; structured objects require *all* scalar leaves."""
    leaves = [x for x in _leaf_values(value) if x is not None and str(x).strip()]
    return bool(leaves) and all(scalar_value_match(text, x, semantic_aliases=semantic_aliases) for x in leaves)


def forbidden_semantic_mentions(text: str, ineligible: list[dict], raw_query: str = "", *, semantic_aliases: bool = False) -> list[dict]:
    """Equivalent of evaluator forbidden detection using the same value matcher.

    Values already supplied by the current user query are not memory leakage unless
    the answer introduces a new entity association, mirroring the preregistered
    evaluator's exemption.
    """
    out = []
    q_norm = normalize_surface(raw_query)
    a_norm = normalize_surface(text)
    for f in ineligible:
        v = f.get("value")
        if not semantic_value_match(text, v, semantic_aliases=semantic_aliases):
            continue
        if semantic_value_match(raw_query, v, semantic_aliases=semantic_aliases):
            ent = normalize_surface(f.get("entity") or "")
            if ent and ent not in q_norm and ent in a_norm:
                out.append(f)
            continue
        out.append(f)
    return out


def _contract_match(response: str, contract: str) -> bool:
    low = normalize_surface(response)
    if "insufficient memory" in low:
        return True
    return any(normalize_surface(c) in low for c in _CONTRACT_CUES.get(contract, _CONTRACT_CUES["negative_inactive_or_unknown"]))


def equivalence_answer_score(response: str, gold: dict, raw_query: str = "", *, semantic_aliases: bool = False) -> tuple[float, dict]:
    """Rescore expected values with representation equivalence.

    For no-positive-fact contracts, the primary representation rescore deliberately
    preserves the preregistered contract scorer.  A broader semantic-contract
    diagnostic is available with ``semantic_aliases=True`` and is never used as the
    recommended hard correction.
    """
    expected = gold.get("expected_facts") or []
    ineligible = gold.get("ineligible_facts") or []
    forbidden = forbidden_semantic_mentions(response, ineligible, raw_query, semantic_aliases=semantic_aliases)
    if forbidden:
        return 0.0, {
            "expected_matches": [semantic_value_match(response, f.get("value"), semantic_aliases=semantic_aliases) for f in expected],
            "forbidden_match_count": len(forbidden),
            "contract_match": None,
        }
    if not expected:
        if not semantic_aliases:
            score = answer_score(response, expected, ineligible, gold.get("expect_abstain", False), query=gold.get("query_semantics"), raw_query=raw_query)
            return float(score), {"expected_matches": [], "forbidden_match_count": 0, "contract_match": bool(score >= .999999), "contract_policy": "preregistered_unchanged"}
        contract = gold.get("answer_contract") or "negative_inactive_or_unknown"
        ok = _contract_match(response, contract)
        return float(ok), {"expected_matches": [], "forbidden_match_count": 0, "contract_match": bool(ok), "contract_policy": "semantic_sensitivity"}
    matches = [semantic_value_match(response, f.get("value"), semantic_aliases=semantic_aliases) for f in expected]
    tp = sum(matches)
    fp = len(forbidden)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / len(expected)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return float(f1), {"expected_matches": matches, "forbidden_match_count": fp, "contract_match": None}


def _suite_ids(data_suite_root: Path) -> list[str]:
    mf = json.loads((data_suite_root / "suite_manifest.json").read_text(encoding="utf-8"))
    ids = mf.get("trajectory_ids") or mf.get("trajectories") or []
    if ids and isinstance(ids[0], dict):
        ids = [x.get("trajectory_id") or x.get("id") for x in ids]
    return [str(x) for x in ids]


def load_frozen_heldout_rows(campaign_root: str | Path, data_suite_root: str | Path) -> tuple[list[dict], dict]:
    root = Path(campaign_root)
    suite = Path(data_suite_root)
    tids = _suite_ids(suite)
    rows: list[dict] = []
    meta = {"campaign_root": str(root.resolve()), "data_suite_root": str(suite.resolve()), "trajectory_ids": tids}
    for model in ("7b", "3b"):
        for traj in tids:
            data_root = suite / traj
            _, gold_q = runtime_gold(data_root, {})
            for cond in CONDITIONS:
                base = root / model / "heldout" / traj / "primary" / cond
                trace_path = base / "trace.jsonl"
                eval_path = base / "evaluation.json"
                if not trace_path.exists() or not eval_path.exists():
                    raise FileNotFoundError(f"missing frozen held-out outputs: {trace_path} / {eval_path}")
                traces = {r["turn_id"]: r for r in read_jsonl(trace_path)}
                ev = json.loads(eval_path.read_text(encoding="utf-8"))
                eval_q = {q["turn_id"]: q for q in ev.get("queries", [])}
                for turn_id, gold in gold_q.items():
                    tr = traces.get(turn_id)
                    oq = eval_q.get(turn_id)
                    if tr is None or oq is None:
                        raise ValueError(f"frozen output incomplete: {model}/{traj}/{cond}/{turn_id}")
                    q = tr.get("query") or {}
                    resp = str(q.get("response") if q.get("response") is not None else tr.get("assistant_response") or "")
                    raw_query = str(tr.get("raw_utterance") or "")
                    eq_f1, detail = equivalence_answer_score(resp, gold, raw_query, semantic_aliases=False)
                    sem_f1, sem_detail = equivalence_answer_score(resp, gold, raw_query, semantic_aliases=True)
                    eq_gvr = float(bool(forbidden_semantic_mentions(resp, gold.get("ineligible_facts") or [], raw_query, semantic_aliases=False)))
                    sem_gvr = float(bool(forbidden_semantic_mentions(resp, gold.get("ineligible_facts") or [], raw_query, semantic_aliases=True)))
                    support_correct = bool(oq.get("support_correct"))
                    rescored_jaur = float(
                        float(oq.get("query_state_f1") or 0.0) >= .999999
                        and support_correct
                        and float(oq.get("needed_sra") or 0.0) >= .999999
                        and eq_gvr == 0.0
                        and eq_f1 >= .999999
                    )
                    semantic_diag_jaur = float(
                        float(oq.get("query_state_f1") or 0.0) >= .999999
                        and support_correct
                        and float(oq.get("needed_sra") or 0.0) >= .999999
                        and sem_gvr == 0.0
                        and sem_f1 >= .999999
                    )
                    refs = canonical_references(gold)
                    tpr = _best_over_refs(resp, refs, token_prf)
                    rpr = _best_over_refs(resp, refs, rouge_l_prf)
                    rows.append({
                        "model": model, "trajectory_id": traj, "condition": cond,
                        "turn_id": turn_id, "conversation_id": tr.get("conversation_id"),
                        "family": oq.get("family") or gold.get("family"),
                        "raw_query": raw_query, "response": resp,
                        "answer_contract": gold.get("answer_contract"),
                        "expected_values": [f.get("value") for f in gold.get("expected_facts") or []],
                        "canonical_references": refs,
                        "original_answer_f1": float(oq.get("answer_f1") or 0.0),
                        "representation_equivalence_answer_f1": eq_f1,
                        "semantic_sensitivity_answer_f1": sem_f1,
                        "answer_f1_delta": eq_f1 - float(oq.get("answer_f1") or 0.0),
                        "original_gvr": float(oq.get("gvr") or 0.0),
                        "representation_equivalence_gvr": eq_gvr,
                        "semantic_sensitivity_gvr": sem_gvr,
                        "original_jaur": float(oq.get("jaur") or 0.0),
                        "representation_rescored_jaur": rescored_jaur,
                        "semantic_sensitivity_jaur": semantic_diag_jaur,
                        "jaur_delta": rescored_jaur - float(oq.get("jaur") or 0.0),
                        "query_state_f1": float(oq.get("query_state_f1") or 0.0),
                        "needed_sra": float(oq.get("needed_sra") or 0.0),
                        "support_correct": support_correct,
                        "representation_match_detail": detail,
                        "semantic_sensitivity_match_detail": sem_detail,
                        "token_f1": tpr[2], "rouge_l_f1": rpr[2],
                        "chrf": _best_over_refs(resp, refs, chrf),
                        "bleu4": _best_over_refs(resp, refs, sentence_bleu4),
                        "bertscore_precision": None, "bertscore_recall": None, "bertscore_f1": None,
                    })
    return rows, meta


def _mean(rows: list[dict], key: str) -> float | None:
    vals = [float(r[key]) for r in rows if r.get(key) is not None]
    return float(np.mean(vals)) if vals else None


def summarize(rows: list[dict]) -> dict:
    metrics = ("original_answer_f1", "representation_equivalence_answer_f1", "semantic_sensitivity_answer_f1", "original_jaur", "representation_rescored_jaur", "semantic_sensitivity_jaur", "bertscore_f1", "token_f1", "rouge_l_f1")
    out: dict[str, Any] = {"models": {}}
    for model in ("7b", "3b"):
        mr = [r for r in rows if r["model"] == model]
        m = {"macro": {}, "trajectories": {}, "families": {}}
        for cond in CONDITIONS:
            xs = [r for r in mr if r["condition"] == cond]
            m["macro"][cond] = {k: _mean(xs, k) for k in metrics}
            m["macro"][cond]["n"] = len(xs)
        for traj in sorted({r["trajectory_id"] for r in mr}):
            m["trajectories"][traj] = {}
            for cond in CONDITIONS:
                xs = [r for r in mr if r["trajectory_id"] == traj and r["condition"] == cond]
                m["trajectories"][traj][cond] = {k: _mean(xs, k) for k in metrics}
                m["trajectories"][traj][cond]["n"] = len(xs)
        for fam in sorted({r["family"] for r in mr}):
            m["families"][fam] = {}
            for cond in CONDITIONS:
                xs = [r for r in mr if r["family"] == fam and r["condition"] == cond]
                m["families"][fam][cond] = {k: _mean(xs, k) for k in metrics}
                m["families"][fam][cond]["n"] = len(xs)
        changed = [r for r in mr if abs(r["answer_f1_delta"]) > 1e-12 or abs(r["jaur_delta"]) > 1e-12]
        m["changed_query_rows"] = len(changed)
        m["changed_query_rows_positive_answer_delta"] = sum(r["answer_f1_delta"] > 1e-12 for r in changed)
        m["changed_query_rows_negative_answer_delta"] = sum(r["answer_f1_delta"] < -1e-12 for r in changed)
        out["models"][model] = m
    return out


def paired_bootstrap_place(rows: list[dict], metric: str, reps: int = 4000, seed: int = 42) -> dict:
    """Paired trajectory-block bootstrap for PLACE-Native on frozen held-outs."""
    rng = np.random.default_rng(seed)
    by = {(r["model"], r["trajectory_id"], r["condition"], r["turn_id"]): r for r in rows}
    result = {}
    for model in ("7b", "3b"):
        trajs = sorted({r["trajectory_id"] for r in rows if r["model"] == model})
        diffs = []
        by_traj = {}
        for traj in trajs:
            tids = sorted({r["turn_id"] for r in rows if r["model"] == model and r["trajectory_id"] == traj})
            ds = []
            for tid in tids:
                a = by.get((model, traj, "place", tid)); b = by.get((model, traj, "native", tid))
                if a is not None and b is not None:
                    ds.append(float(a[metric]) - float(b[metric]))
            by_traj[traj] = ds; diffs.extend(ds)
        point = float(np.mean(diffs)) if diffs else None
        boots = []
        if trajs:
            for _ in range(reps):
                chosen = rng.choice(trajs, size=len(trajs), replace=True)
                z = [d for t in chosen for d in by_traj[t]]
                if z: boots.append(float(np.mean(z)))
        result[model] = {
            "contrast": "PLACE - Native", "metric": metric, "estimate": point,
            "query_n": len(diffs), "trajectory_cluster_n": len(trajs),
            "interval95": [float(np.percentile(boots,2.5)), float(np.percentile(boots,97.5))] if boots else [None,None],
            "interval_type": "trajectory_block_bootstrap_sensitivity",
        }
    return result
