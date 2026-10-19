#!/usr/bin/env python3
from __future__ import annotations
import argparse,importlib.metadata,importlib.util,json,platform,sys
from pathlib import Path
EXPECTED='1.0.11'
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output');a=ap.parse_args()
 try:mv=importlib.metadata.version('mem0ai')
 except importlib.metadata.PackageNotFoundError:mv=None
 mods={m:bool(importlib.util.find_spec(m)) for m in ('mem0','yaml','requests','pytest','matplotlib','networkx','ollama')};py=sys.version_info
 rep={'schema_version':'asf-v7.12.4-mem-gov-env-v1','passed':bool(py.major==3 and py.minor==11 and mv==EXPECTED and all(mods.values())),'python':platform.python_version(),'python_executable':sys.executable,'mem0ai':mv,'expected_mem0ai':EXPECTED,'imports':mods,'note':'Python 3.11 + plain Mem0 1.0.11. The active package intentionally omits mem0ai[graph], Apache AGE, Kuzu and Neo4j; Letta has a separate Node runtime.'}
 t=json.dumps(rep,indent=2);print(t)
 if a.output:Path(a.output).write_text(t+'\n')
 raise SystemExit(0 if rep['passed'] else 2)
if __name__=='__main__':main()
