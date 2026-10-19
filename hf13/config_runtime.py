from __future__ import annotations
import copy, os
from pathlib import Path
from typing import Any
import yaml


def _normalize_base_url(value: str | None) -> str | None:
    if not value:
        return None
    v=str(value).strip().rstrip('/')
    if not v:
        return None
    if not (v.startswith('http://') or v.startswith('https://')):
        v='http://'+v
    return v


def apply_runtime_overrides(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of *cfg* aligned to the active HF13 Ollama runtime.

    HF11's ``server/start_ollama_gpu.sh`` can auto-select a free TCP port and
    persists it in ``server_runtime/ollama_runtime.env``.  When that environment
    is loaded, ``OLLAMA_BASE_URL`` is authoritative for the answer model and for
    both Mem0 Ollama clients (LLM + embedder).  This keeps the legacy GPU launcher
    compatible with HF13 even when the selected port is not 11434.
    """
    out=copy.deepcopy(cfg or {})
    env_base=_normalize_base_url(os.environ.get('OLLAMA_BASE_URL') or os.environ.get('OLLAMA_HOST'))
    model=out.setdefault('model',{})
    base=env_base or _normalize_base_url(model.get('base_url')) or 'http://127.0.0.1:11434'
    model['base_url']=base
    mem=out.setdefault('memory',{}).setdefault('mem0_config',{})
    llm=mem.setdefault('llm',{}).setdefault('config',{})
    emb=mem.setdefault('embedder',{}).setdefault('config',{})
    # Mem0 1.0.11 OllamaConfig/BaseEmbedderConfig both accept ollama_base_url.
    llm['ollama_base_url']=base
    emb['ollama_base_url']=base
    return out


def load_config(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return apply_runtime_overrides({})
    raw=yaml.safe_load(Path(path).read_text()) or {}
    return apply_runtime_overrides(raw)
