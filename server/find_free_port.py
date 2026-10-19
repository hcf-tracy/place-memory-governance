#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, socket, sys

def free(host: str, port: int) -> bool:
    s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    try:
        s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,0)
        s.bind((host,port))
        return True
    except OSError:
        return False
    finally:
        s.close()

def main():
    ap=argparse.ArgumentParser(description='Return the first free TCP port in an inclusive range.')
    ap.add_argument('--host',default='127.0.0.1'); ap.add_argument('--start',type=int,default=11434); ap.add_argument('--end',type=int,default=11999)
    ap.add_argument('--json',action='store_true'); a=ap.parse_args()
    if not (1 <= a.start <= a.end <= 65535): raise SystemExit('invalid port range')
    for p in range(a.start,a.end+1):
        if free(a.host,p):
            if a.json: print(json.dumps({'host':a.host,'port':p,'base_url':f'http://{a.host}:{p}'}))
            else: print(p)
            return 0
    print(f'No free TCP port in {a.host}:{a.start}-{a.end}',file=sys.stderr); return 2
if __name__=='__main__': raise SystemExit(main())
