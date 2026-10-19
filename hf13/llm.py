from __future__ import annotations

from typing import Any
import time

import requests

LAST_LLM_DIAGNOSTICS = {'status': 'clean', 'attempts': 0, 'errors': []}


def get_last_diagnostics():
    return dict(LAST_LLM_DIAGNOSTICS)


def ollama_chat(
    base_url,
    model,
    messages,
    *,
    temperature=0,
    max_tokens=384,
    timeout=180,
    max_retries=2,
    format_json=False,
    format_schema: dict[str, Any] | None = None,
):
    """Call Ollama chat with optional JSON-schema constrained structured output.

    ``format_schema`` takes precedence over legacy ``format_json``.  Ollama's
    local chat API accepts a JSON Schema object in ``format``; this is preferred
    for parser diagnostics because it constrains *shape* without exposing any
    evaluator gold semantics.
    """
    global LAST_LLM_DIAGNOSTICS
    errors = []
    for attempt in range(1, int(max_retries) + 2):
        try:
            payload = {
                'model': model,
                'messages': messages,
                'stream': False,
                'options': {'temperature': temperature, 'num_predict': max_tokens},
            }
            if format_schema is not None:
                payload['format'] = format_schema
            elif format_json:
                payload['format'] = 'json'
            r = requests.post(base_url.rstrip('/') + '/api/chat', json=payload, timeout=timeout)
            r.raise_for_status()
            obj = r.json()
            text = str((obj.get('message') or {}).get('content') or '').strip()
            if not text:
                raise RuntimeError('empty Ollama chat response')
            LAST_LLM_DIAGNOSTICS = {
                'status': 'recovered' if errors else 'clean',
                'attempts': attempt,
                'errors': errors,
                'structured_output_schema': format_schema is not None,
            }
            return text
        except Exception as e:
            errors.append(f'{type(e).__name__}: {e}')
            if attempt > int(max_retries):
                LAST_LLM_DIAGNOSTICS = {
                    'status': 'failed',
                    'attempts': attempt,
                    'errors': errors,
                    'structured_output_schema': format_schema is not None,
                }
                raise
            time.sleep(min(2.0, .35 * (2 ** (attempt - 1))))
    raise RuntimeError('unreachable')


def _cfg(cfg):
    m = cfg.get('model') or {}
    return (
        m.get('base_url', 'http://127.0.0.1:11434'),
        m.get('model_id', 'qwen2.5:3b'),
        float(m.get('temperature', 0)),
        int(m.get('max_new_tokens_answer', 384)),
        int(m.get('timeout_seconds', 180)),
    )


def _ctx(context, limit=12):
    return [
        {'role': x.get('role'), 'content': str(x.get('content') or '')}
        for x in (context or [])[-limit:]
        if x.get('role') in {'user', 'assistant'}
    ]


def answer_with_support(query, support, cfg, *, context=None):
    base, model, temp, mx, to = _cfg(cfg)
    sys = (
        'Answer the latest user query using only visible context and MEMORY SUPPORT. '
        'Do not invent persistent-memory facts. If support is insufficient, say '
        'INSUFFICIENT_MEMORY.\n\nMEMORY SUPPORT:\n'
        + ('\n'.join('- ' + x for x in support) if support else '(none)')
    )
    return ollama_chat(
        base,
        model,
        [{'role': 'system', 'content': sys}, *_ctx(context), {'role': 'user', 'content': query}],
        temperature=temp,
        max_tokens=mx,
        timeout=to,
    )


def answer_natural_turn(utterance, cfg, *, context=None, memory_support=None, min_numbered_items=0):
    base, model, temp, mx, to = _cfg(cfg)
    sys = (
        'Respond naturally to the latest user message. Persistent-memory support may be '
        'used when relevant; do not claim persistent facts absent from visible context '
        'or MEMORY SUPPORT.'
    )
    if min_numbered_items:
        sys += f' Provide at least {int(min_numbered_items)} distinct numbered item lines; do not predict which item may be referenced later.'
    sys += '\n\nMEMORY SUPPORT:\n' + ('\n'.join('- ' + x for x in (memory_support or [])) if memory_support else '(none)')
    return ollama_chat(
        base,
        model,
        [{'role': 'system', 'content': sys}, *_ctx(context), {'role': 'user', 'content': utterance}],
        temperature=temp,
        max_tokens=mx,
        timeout=to,
    )
