from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def stable_hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def normalize_text(value: Any) -> str:
    text = str(value or "").lower()
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def flatten_store(raw: Any) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, dict):
        for key in ("results", "memories", "data"):
            if isinstance(raw.get(key), list):
                return [x if isinstance(x, dict) else {"memory": str(x)} for x in raw[key]]
        if "memory" in raw or "text" in raw:
            return [raw]
        return [{"memory": json.dumps(raw, ensure_ascii=False, default=str)}]
    if isinstance(raw, list):
        return [x if isinstance(x, dict) else {"memory": str(x)} for x in raw]
    return [{"memory": str(raw)}]


def record_text(record: dict[str, Any]) -> str:
    for key in ("memory", "text", "content", "value"):
        if key in record and record[key] is not None:
            return str(record[key])
    return json.dumps(record, ensure_ascii=False, default=str)


def record_id(record: dict[str, Any]) -> str | None:
    for key in ("id", "memory_id", "uuid"):
        if record.get(key):
            return str(record[key])
    return None


# Turn-local evaluation directives must never leak through state inheritance.
# Inheritance carries persistent state, not the response contract of a previous turn.
TURN_LOCAL_GOLD_FIELDS = {
    "should_abstain",
    "answer_contract",
    "answer_contract_revision",
    "answer_forbidden_privacy_terms",
    "authorized_sensitive_use",
}


def resolve_checkpoint(checkpoints: dict[str, Any], step: int) -> dict[str, Any]:
    """Resolve checkpoint inheritance into one complete gold contract."""
    seen: set[int] = set()

    def _resolve(s: int) -> dict[str, Any]:
        if s in seen:
            raise ValueError(f"Cyclic checkpoint inheritance at step {s}")
        seen.add(s)
        node = copy.deepcopy(checkpoints.get(str(s), checkpoints.get(s, {})))
        parent = node.pop("inherit", None)
        if parent is None:
            return node
        base = _resolve(int(parent))
        # Persistent state may inherit; response/abstention directives are local
        # to the current turn unless the child explicitly re-declares them.
        for local_key in TURN_LOCAL_GOLD_FIELDS:
            if local_key not in node:
                base.pop(local_key, None)
        # Apply explicit removals before merging inherited state deltas.
        remove_ids=set(str(x) for x in node.pop("remove_required_state_object_ids", []) or [])
        if remove_ids:
            base["required_state"]=[x for x in base.get("required_state",[]) if str(x.get("object_id") or x.get("key")) not in remove_ids]
        remove_forbidden=set(str(x) for x in node.pop("remove_forbidden_state_object_ids", []) or [])
        if remove_forbidden:
            base["forbidden_state"]=[x for x in base.get("forbidden_state",[]) if str(x.get("object_id") or x.get("key")) not in remove_forbidden]
        # State lists are merged by stable object identity; other lists are complete child contracts.
        for key, value in node.items():
            if key in {"required_state", "forbidden_state"}:
                merged: dict[str, dict[str, Any]] = {
                    str(x.get('object_id') or x.get('key')): x for x in base.get(key, []) if isinstance(x, dict)
                }
                for item in value or []:
                    if isinstance(item, dict):
                        merged[str(item.get('object_id') or item.get('key'))] = item
                base[key] = list(merged.values())
            elif isinstance(value, list):
                # Child list fields are complete checkpoint contracts, not deltas.
                base[key] = value
            elif isinstance(value, dict) and isinstance(base.get(key), dict):
                base[key].update(value)
            else:
                base[key] = value
        return base

    return _resolve(step)


def canonical_atom_id(atom: dict[str, Any]) -> str:
    if atom.get("object_id"):
        return str(atom["object_id"])
    scope = atom.get("scope") if isinstance(atom.get("scope"), dict) else {}
    identity = {
        "key": atom.get("key"),
        "task": scope.get("task"),
        "trip": scope.get("trip", scope.get("trip_id")),
        "person": scope.get("person", scope.get("person_id")),
        "branch": scope.get("branch", scope.get("branch_id")),
        "subject": atom.get("subject"),
    }
    return "atom:" + stable_hash(identity)[:24]


@dataclass
class CanonicalState:
    atoms: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_snapshot(cls, atoms: Iterable[dict[str, Any]]) -> "CanonicalState":
        return cls({canonical_atom_id(x): copy.deepcopy(x) for x in atoms})

    def apply_gold(self, checkpoint: dict[str, Any]) -> None:
        # Short-suite checkpoints marked full_state_after_inheritance are
        # authoritative complete snapshots after resolve_checkpoint().  Merging
        # them incrementally can retain stale/purged objects that are absent from
        # the gold snapshot and corrupt reference/oracle conformance.
        if str(checkpoint.get("state_contract") or "") == "full_state_after_inheritance":
            self.atoms = {canonical_atom_id(atom): copy.deepcopy(atom) for atom in (checkpoint.get("required_state", []) or [])}
            return
        for atom in checkpoint.get("required_state", []) or []:
            self.atoms[canonical_atom_id(atom)] = copy.deepcopy(atom)
        for forbidden in checkpoint.get("forbidden_state", []) or []:
            remove = []
            for identity, current in self.atoms.items():
                if forbidden.get("object_id") and str(current.get("object_id")) != str(forbidden.get("object_id")):
                    continue
                if forbidden.get("key") and str(current.get("key")) != str(forbidden.get("key")):
                    continue
                matched = True
                for key, value in forbidden.items():
                    if key in {"object_id", "key", "sensitivity", "purge_reason"}:
                        continue
                    if key == "scope" and isinstance(value, dict):
                        current_scope = current.get("scope") if isinstance(current.get("scope"), dict) else {}
                        if any(current_scope.get(k) != v for k, v in value.items() if v is not None):
                            matched = False; break
                    elif value is not None and current.get(key) != value:
                        matched = False; break
                if matched: remove.append(identity)
            for identity in remove:self.atoms.pop(identity, None)

    def snapshot(self) -> list[dict[str, Any]]:
        return [copy.deepcopy(self.atoms[k]) for k in sorted(self.atoms)]

    def task_eligible(self) -> list[dict[str, Any]]:
        out = []
        for atom in self.snapshot():
            lifecycle = atom.get("lifecycle", "active")
            activation = atom.get("activation", "task_eligible")
            if lifecycle in {"active", "candidate"} and activation not in {
                "suppressed", "historical_only", "forbidden", "not_yet_selected"
            }:
                out.append(atom)
        return out

    def historical(self) -> list[dict[str, Any]]:
        return [
            a for a in self.snapshot()
            if a.get("activation") == "historical_only"
            or a.get("lifecycle") in {"archived", "superseded", "expired", "completed", "cancelled"}
        ]


def atom_to_text(atom: dict[str, Any]) -> str:
    # Managed records must be physically distinct even when they share a
    # semantic key (for example current + historical superseded versions).
    # object_id/version are storage identity, not benchmark gold.
    fields = [
        f"object_id={atom.get('object_id')}",
        f"version={atom.get('version')}",
        f"key={atom.get('key')}",
        f"value={json.dumps(atom.get('value'), ensure_ascii=False)}",
    ]
    for key in ("lifecycle", "activation", "epistemic", "source", "valid_until", "unit", "sensitivity"):
        if key in atom:
            fields.append(f"{key}={atom[key]}")
    return "[LGCM_ATOM] " + "; ".join(fields)
