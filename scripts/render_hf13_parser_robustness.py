#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path

def fmt(x):return 'N/A' if x is None else f'{float(x):.4f}'
def table(means):
    out=['| Condition | State F1 | Authorized support F1 | Answer F1 | JAUR | Exposure | GVR |','|---|---:|---:|---:|---:|---:|---:|']
    for c in ('TT','PT','TP','PP'):
        m=means.get(c) or {};out.append(f"| {c} | {fmt(m.get('state_f1'))} | {fmt(m.get('authorized_support_f1'))} | {fmt(m.get('answer_f1'))} | {fmt(m.get('jaur'))} | {fmt(m.get('exposure'))} | {fmt(m.get('gvr'))} |")
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('result');ap.add_argument('--output',required=True);a=ap.parse_args();z=json.load(open(a.result));ps=z['parser_summary'];tt=z.get('tt_correct_subset') or {};fb=z.get('fallback_summary') or {};fm=z.get('family_metrics') or {};ic=z.get('parser_input_contract') or {}
    out=['# HF13 Parser Robustness Diagnostic','',f"Role: **{z.get('role')}**. Primary-table eligible: **{z.get('primary_table_eligible')}**.",'',
         '## Parser input and correctness','',
         f"Parser input: target raw utterance + up to **{ic.get('context_turn_limit','?')}** prior raw turns from the same conversation + fixed public schema ontology. Persistent memory/trusted annotations/evaluator gold are not parser-visible.",'',
         '| Parser metric | Value |','|---|---:|',
         f"| Event schema-valid rate | {fmt(ps.get('event_schema_valid_rate'))} |",f"| Query schema-valid rate | {fmt(ps.get('query_schema_valid_rate'))} |",
         f"| Event runtime-valid rate | {fmt(ps.get('event_runtime_valid_rate'))} |",f"| Query runtime-valid rate | {fmt(ps.get('query_runtime_valid_rate'))} |",
         f"| Event field micro-F1 | {fmt(ps.get('event_field_micro_f1'))} |",f"| Query full-field micro-F1 | {fmt(ps.get('query_field_micro_f1'))} |",f"| Query-intent micro-F1 | {fmt(ps.get('query_intent_micro_f1'))} |",f"| Query-target micro-F1 | {fmt(ps.get('query_target_micro_f1'))} |",f"| Query-scope micro-F1 | {fmt(ps.get('query_scope_micro_f1'))} |",
         f"| Event exact-match rate | {fmt(ps.get('event_exact_match_rate'))} |",f"| Query exact-match rate | {fmt(ps.get('query_exact_match_rate'))} |",
         f"| Transition agreement | {fmt(ps.get('transition_agreement_rate'))} |",f"| Event fallback rate | {fmt(fb.get('event_fallback_rate'))} |",f"| Query fallback rate | {fmt(fb.get('query_fallback_rate'))} |",'',
         '## Family-stratified parser diagnostics','',
         '| Family | n | Event F1 | Query intent F1 | Query full F1 | Query target F1 | Query scope F1 | Transition agree. | Event runtime valid | Query runtime valid |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for fam,m in fm.items():out.append(f"| {fam} | {m.get('n',0)} | {fmt(m.get('event_field_f1_mean'))} | {fmt(m.get('query_intent_f1_mean'))} | {fmt(m.get('query_field_f1_mean'))} | {fmt(m.get('query_target_f1_mean'))} | {fmt(m.get('query_scope_f1_mean'))} | {fmt(m.get('transition_agreement_rate'))} | {fmt(m.get('event_runtime_valid_rate'))} | {fmt(m.get('query_runtime_valid_rate'))} |")
    out += ['', '## All 24 prespecified cases','',*table(z.get('condition_means') or {}),'',
            '## TT-correct propagation subset','',tt.get('definition',''),f"Subset size: **{tt.get('case_count',0)} / {z.get('case_count')}**",'',*table(tt.get('condition_means') or {}),'',
            'The TT-correct subset is secondary/conditional: it isolates parser-induced degradation among cases where the trusted-semantics control already achieved joint success. The all-case table remains the prespecified diagnostic population.','',
            '## Interpretation guardrail','',
            'PT/TP/PP downstream scores must be interpreted jointly with parser semantic F1 and fallback rates. Query-intent F1 measures parser-observable request semantics; full query/target/scope F1 additionally reflects referent resolution and is therefore diagnostic rather than a smoke-gate requirement. A fail-closed fallback can improve abstention on some queries without constituting successful semantic parsing; fallback-driven gains are therefore not parser-quality evidence.','']
    Path(a.output).write_text('\n'.join(out),encoding='utf-8')
if __name__=='__main__':main()
