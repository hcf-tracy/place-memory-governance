from __future__ import annotations
from pathlib import Path
from .io import read_json,read_jsonl
from .schema import validate_trusted_event,validate_trusted_query,validate_raw_row
class ConditionView:
    def __init__(self,root,condition):
        self.root=Path(root);self.condition=condition
        if condition not in {'native','css','qasc','place'}:raise ValueError(condition)
        self.turns=read_jsonl(self.root/'trajectory_raw.jsonl');[validate_raw_row(x) for x in self.turns];self.boundaries=read_json(self.root/'conversation_boundaries.json');self._events={};self._queries={}
        if condition in {'css','place'}:
            all_events=read_jsonl(self.root/'trusted_events.jsonl');[validate_trusted_event(x) for x in all_events];self._events={x['turn_id']:x for x in all_events}
        # QASC/PLACE receive trusted QUERY semantics only at read time. QASC has no
        # runtime path to trusted write events.
        if condition in {'qasc','place'}:
            xs=read_jsonl(self.root/'trusted_queries.jsonl');[validate_trusted_query(x) for x in xs];self._queries={x['turn_id']:x for x in xs}
    def event_for(self,tid):return self._events.get(tid)
    def query_for(self,tid):return self._queries.get(tid)
    @property
    def hidden(self):raise AttributeError('hidden evaluator data is not available to runtime')
class HarnessSchedule:
    def __init__(self,root):
        r=Path(root);self.roles={x['turn_id']:x['turn_role'] for x in read_jsonl(r/'turn_roles.jsonl')};bs=read_jsonl(r/'runtime_bindings.jsonl');self.bindings={x['target_turn_id']:x for x in bs};self.bindings_by_id={x['binding_id']:x for x in bs};self.queries={x['turn_id']:x for x in read_jsonl(r/'trusted_queries.jsonl')}
    def is_query_checkpoint(self,tid):return 'query_checkpoint' in self.roles.get(tid,[])
class EvaluatorData:
    def __init__(self,root):
        r=Path(root);h=r/'hidden';self.gold_state=read_jsonl(h/'gold_state.jsonl');self.gold_query={x['turn_id']:x for x in read_jsonl(h/'gold_query.jsonl')};self.causal_pairs=read_jsonl(h/'causal_pairs.jsonl');self.utility_cases=read_jsonl(r/'utility_cases.jsonl');self.utility_gold={x['case_id']:x for x in read_jsonl(h/'utility_gold.jsonl')};self.fault_cases=read_jsonl(r/'fault_cases.jsonl');self.fault_gold={x['case_id']:x for x in read_jsonl(h/'fault_gold.jsonl')};self.parser_manifest=read_jsonl(r/'parser_diagnostic/manifest.jsonl');self.parser_gold={x['turn_id']:x for x in read_jsonl(h/'parser_gold.jsonl')};self.manifest=read_json(h/'benchmark_manifest.json')
