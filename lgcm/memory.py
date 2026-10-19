from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import gc
import io
import sys
from contextlib import redirect_stdout, redirect_stderr
from typing import Any

from .core import atom_to_text, flatten_store, record_id, record_text


class Mem0Backend:
    def __init__(self, config: dict[str, Any]):
        # Benchmark runs must not create Mem0's global telemetry Qdrant store
        # (~/.mem0/migrations_qdrant). In Mem0 1.0.x every Memory instance can
        # otherwise open that same embedded-Qdrant directory, which is exclusive
        # and causes AlreadyLocked when source-ablation creates a second client.
        # Telemetry is orthogonal to memory semantics, so disabling it improves
        # reproducibility without changing the evaluated algorithm.
        os.environ["MEM0_TELEMETRY"] = "False"
        try:
            from mem0 import Memory  # type: ignore
            # Defensive patch for processes where mem0 was imported earlier.
            try:
                import mem0.memory.telemetry as _mem0_telemetry  # type: ignore
                import mem0.memory.main as _mem0_main  # type: ignore
                _mem0_telemetry.MEM0_TELEMETRY = False
                _mem0_main.MEM0_TELEMETRY = False
            except Exception:
                pass
        except ImportError as exc:
            raise RuntimeError("Install mem0ai: pip install mem0ai") from exc

        self.memory = Memory.from_config(config)

        # Native-prompt experiments must never silently fall back to a researcher-defined
        # prompt. Failing closed preserves experimental validity.
        try:
            from mem0.configs import prompts as mem0_prompts  # type: ignore
            native_prompt = str(mem0_prompts.MEMORY_ANSWER_PROMPT)
        except Exception as exc:
            raise RuntimeError(
                "Could not load mem0.configs.prompts.MEMORY_ANSWER_PROMPT. "
                "Native-prompt experiments cannot continue because a fallback prompt "
                "would invalidate the comparison."
            ) from exc

        if not native_prompt.strip():
            raise RuntimeError("Mem0 MEMORY_ANSWER_PROMPT is empty.")

        self.native_answer_prompt = native_prompt
        self.native_prompt_sha256 = hashlib.sha256(
            native_prompt.encode("utf-8")
        ).hexdigest()
        try:
            self.mem0_version = importlib.metadata.version("mem0ai")
        except importlib.metadata.PackageNotFoundError:
            try:
                self.mem0_version = importlib.metadata.version("mem0")
            except importlib.metadata.PackageNotFoundError:
                self.mem0_version = "unknown"
        self.last_inferred_diagnostics: dict[str, Any] = {"stdout": "", "stderr": "", "error_lines": [], "exception": None}
        self.prompt_provenance = {
            "prompt_source": "mem0.configs.prompts.MEMORY_ANSWER_PROMPT",
            "prompt_sha256": self.native_prompt_sha256,
            "prompt_text": self.native_answer_prompt,
            "mem0_version": self.mem0_version,
            "fallback_used": False,
        }

    def add_inferred(
        self,
        messages: list[dict[str, str]],
        user_id: str | None = None,
        *,
        agent_id: str | None = None,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        """Run Mem0 **native** inference without any custom prompt override.

        V7.12.5 uses this method for the online primary and closed-loop experiments.
        ``prompt`` is deliberately never supplied to ``Memory.add``.  Therefore Mem0
        1.0.11 selects its own USER/AGENT fact-extraction prompt and its native
        DEFAULT_UPDATE_MEMORY_PROMPT.  The optional identifiers/metadata are routing
        and provenance only; they contain no benchmark gold labels.

        Provider-internal diagnostics are preserved because Mem0 can log an action
        processing error while returning normally.
        """
        if not any((user_id, agent_id, run_id)):
            raise ValueError("add_inferred requires user_id, agent_id, or run_id")
        out_buf, err_buf = io.StringIO(), io.StringIO()
        result = None
        exc_info = None
        try:
            with redirect_stdout(out_buf), redirect_stderr(err_buf):
                result = self.memory.add(
                    messages,
                    user_id=user_id,
                    agent_id=agent_id,
                    run_id=run_id,
                    metadata=dict(metadata or {}),
                    infer=True,
                )
        except BaseException as exc:
            exc_info = {"type": type(exc).__name__, "message": str(exc)}
            raise
        finally:
            stdout, stderr = out_buf.getvalue(), err_buf.getvalue()
            # Preserve the provider's normal user-visible behavior.
            if stdout:
                sys.stdout.write(stdout); sys.stdout.flush()
            if stderr:
                sys.stderr.write(stderr); sys.stderr.flush()
            lines = []
            for line in (stdout + "\n" + stderr).splitlines():
                low = line.lower()
                if any(token in low for token in ("error processing memory action", "error in new memory actions response", "empty response from llm", "invalid json response", "traceback", "exception", " error:", "failed to update", "failed to delete", "could not parse", "jsondecodeerror")):
                    lines.append(line)
            invalid_json = [x for x in lines if "invalid json" in x.casefold() or "jsondecode" in x.casefold()]
            action_errors = [x for x in lines if any(t in x.casefold() for t in ("error processing memory action","failed to update","failed to delete"))]
            self.last_inferred_diagnostics = {"stdout":stdout,"stderr":stderr,"error_lines":lines,"provider_internal_error_count":len(lines),"invalid_json_count":len(invalid_json),"action_error_count":len(action_errors),"exception":exc_info}
        return result

    def add_managed_atom(self, atom: dict[str, Any], user_id: str, trajectory_id: str) -> Any:
        # Store a canonical envelope in the physical record text in addition to
        # metadata. Some vector stores/framework versions normalize nested
        # metadata on round-trip; the immutable text envelope provides a second,
        # backend-observable lossless projection source for sigma_B.
        text = atom_to_text(atom)
        canonical_obj = atom.get("lgcm_object") if isinstance(atom.get("lgcm_object"), dict) else None
        canonical_json = json.dumps(canonical_obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str) if canonical_obj else None
        if canonical_json:
            text += "\n[LGCM_OBJECT] " + canonical_json
        metadata = {
            "lgcm_managed": True,
            "lgcm_key": atom.get("key"),
            "object_id": atom.get("object_id"),
            "trajectory_id": trajectory_id,
            "lifecycle": atom.get("lifecycle"),
            "activation": atom.get("activation"),
            # Required for the backend semantic projection sigma_B. The
            # backend remains a storage substrate; this metadata is not gold.
            "lgcm_object": canonical_obj,
            "lgcm_object_sha256": hashlib.sha256(canonical_json.encode("utf-8")).hexdigest() if canonical_json else None,
            "lgcm_version": atom.get("version"),
        }
        attempts = [
            lambda: self.memory.add(text, user_id=user_id, metadata=metadata, infer=False),
            lambda: self.memory.add(
                [{"role": "user", "content": text}],
                user_id=user_id,
                metadata=metadata,
                infer=False,
            ),
        ]
        last: Exception | None = None
        for fn in attempts:
            try:
                return fn()
            except Exception as exc:
                last = exc
        raise RuntimeError(f"Could not add managed atom to Mem0 with infer=False: {last}")

    def search_scope(
        self,
        query: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        kwargs = {"query": query, "limit": limit}
        if user_id is not None: kwargs["user_id"] = user_id
        if agent_id is not None: kwargs["agent_id"] = agent_id
        if run_id is not None: kwargs["run_id"] = run_id
        raw = self.memory.search(**kwargs)
        return flatten_store(raw)

    def get_all_scope(
        self,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        kwargs = {"limit": int(limit)}
        if user_id is not None: kwargs["user_id"] = user_id
        if agent_id is not None: kwargs["agent_id"] = agent_id
        if run_id is not None: kwargs["run_id"] = run_id
        raw = flatten_store(self.memory.get_all(**kwargs))
        if len(raw) >= int(limit):
            raise RuntimeError(
                f"Mem0 get_all_scope reached safety cap {limit}; refusing to use a potentially truncated snapshot"
            )
        return raw

    def search(self, query: str, user_id: str, limit: int = 10) -> list[dict[str, Any]]:
        try:
            raw = self.memory.search(query=query, user_id=user_id, limit=limit)
        except TypeError:
            raw = self.memory.search(query, user_id=user_id, limit=limit)
        return flatten_store(raw)

    def get_all(self, user_id: str) -> list[dict[str, Any]]:
        return flatten_store(self.memory.get_all(user_id=user_id))

    def delete(self, memory_id: str) -> Any:
        return self.memory.delete(memory_id)

    def reconcile_managed(
        self,
        atoms: list[dict[str, Any]],
        user_id: str,
        trajectory_id: str,
        *,
        hard_cleanup: bool = False,
    ) -> dict[str, Any]:
        """Replace the LGCM-managed physical snapshot for one trajectory.

        ``hard_cleanup`` is the bounded final repair path used after an exact
        post-write projection mismatch.  It deliberately does *not* weaken
        semantic verification.  Instead it performs repeated physical cleanup
        before rewriting the authoritative canonical snapshot with ``infer=False``.
        """
        deleted: list[str] = []
        errors: list[str] = []
        expected_ids = {str(a.get("object_id")) for a in atoms if a.get("object_id")}

        def cleanup_pass(pass_index: int) -> int:
            removed = 0
            try:
                existing = self.get_all(user_id)
            except Exception as exc:
                errors.append(f"cleanup_pass_{pass_index}_get_all: {exc}")
                return 0
            for record in existing:
                metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
                text = record_text(record)
                managed = bool(metadata.get("lgcm_managed")) or text.startswith("[LGCM_ATOM]")
                record_oid = metadata.get("object_id")
                same_trajectory = (
                    metadata.get("trajectory_id") == trajectory_id
                    or (not metadata.get("trajectory_id") and managed)
                )
                # During the final hard cleanup, also remove a managed record with
                # an expected object id even if an older backend write lost the
                # trajectory metadata.  User isolation still comes from get_all(user_id).
                expected_identity = bool(record_oid and str(record_oid) in expected_ids)
                rid = record_id(record)
                if managed and (same_trajectory or (hard_cleanup and expected_identity)) and rid:
                    try:
                        self.delete(rid)
                        deleted.append(str(rid))
                        removed += 1
                    except Exception as exc:
                        errors.append(f"cleanup_pass_{pass_index}_delete {rid}: {exc}")
            return removed

        cleanup_pass(1)
        # Qdrant/Mem0 deletes can be observed through a subsequent read only after
        # the client has completed the mutation.  A bounded second sweep catches
        # stale duplicate managed records without creating an unbounded repair loop.
        if hard_cleanup:
            cleanup_pass(2)

        added = 0
        for atom in atoms:
            try:
                self.add_managed_atom(atom, user_id, trajectory_id)
                added += 1
            except Exception as exc:
                errors.append(f"add {atom.get('key')}: {exc}")
        return {
            "deleted": deleted,
            "added": added,
            "errors": errors,
            "hard_cleanup": bool(hard_cleanup),
            "expected_object_ids": sorted(expected_ids),
        }


    def close(self) -> None:
        """Release all backend resources that may own local-store locks.

        Mem0 1.0.x's ``Memory.close()`` closes SQLite but does not close its
        vector-store clients, and its optional telemetry store is a second Qdrant
        client. We therefore close both wrappers and underlying clients explicitly.
        """
        seen: set[int] = set()
        roots = [
            getattr(self.memory, "vector_store", None),
            getattr(self.memory, "_telemetry_vector_store", None),
            getattr(self.memory, "_entity_store", None),
            getattr(self.memory, "graph", None),
            getattr(self.memory, "db", None),
            self.memory,
        ]
        candidates = []
        for root in roots:
            if root is None:
                continue
            candidates.append(root)
            for attr in ("client", "_client", "driver", "_driver"):
                child = getattr(root, attr, None)
                if child is not None:
                    candidates.append(child)
        # Close clients before wrappers where possible.
        for obj in reversed(candidates):
            if obj is None or id(obj) in seen:
                continue
            seen.add(id(obj))
            close = getattr(obj, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
        # Break references so CPython does not keep file-lock owners alive through
        # cycles until a later GC pass.
        for attr in ("_telemetry_vector_store", "_entity_store"):
            if hasattr(self.memory, attr):
                try:
                    setattr(self.memory, attr, None)
                except Exception:
                    pass
        gc.collect()

    def __enter__(self) -> "Mem0Backend":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @staticmethod
    def texts(records: list[dict[str, Any]]) -> list[str]:
        return [record_text(r) for r in records]
