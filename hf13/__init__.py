"""HF13 longitudinal PLACE benchmark.

Primary contract:
- Native: raw sentence only -> native Mem0 write/retrieval.
- CSS: raw content + trusted user-level event; CSS derives internal transitions.
- QASC: Native physical memory + raw query + trusted query semantics.
- PLACE: CSS state formation + QASC use-time authorization.
Hidden gold is evaluator-only. Parser variants are diagnostic-only.
"""
__version__ = "13.0.0"
CONDITIONS = ("native", "css", "qasc", "place")
