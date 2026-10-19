from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Iterable

def read_jsonl(path: str|Path) -> list[dict[str,Any]]:
    p=Path(path)
    out=[]
    if not p.exists(): return out
    for i,line in enumerate(p.read_text(encoding='utf-8').splitlines(),1):
        if line.strip():
            try: out.append(json.loads(line))
            except Exception as e: raise ValueError(f"{p}:{i}: invalid JSON: {e}") from e
    return out

def write_jsonl(path: str|Path, rows: Iterable[dict[str,Any]]) -> None:
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in rows),encoding='utf-8')

def read_json(path: str|Path) -> Any:
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write_json(path: str|Path, obj: Any) -> None:
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,ensure_ascii=False,sort_keys=True,indent=2)+'\n',encoding='utf-8')


def append_jsonl(path: str|Path, row: dict[str,Any]) -> None:
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a',encoding='utf-8') as f:
        f.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n')
        f.flush()

def write_text_atomic(path: str|Path, text: str) -> None:
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(text,encoding='utf-8')
    tmp.replace(p)
