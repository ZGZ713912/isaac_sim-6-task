#!/usr/bin/env python3
"""Fetch completed review checkpoints and evaluate retention on the local GPU.

This independent, read-only monitor never sends training controls. Losing the
monitor, SSH or TensorBoard has no effect on the learner's lifecycle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def discover(host,session):
    code = """import hashlib,json,re
from pathlib import Path
root=Path(SESSION).resolve()
rows=[]
for p in root.glob('attempts/*/train/checkpoints/update_*_review.pt'):
    m=re.fullmatch(r'update_(\d+)_review.pt',p.name)
    update=int(m[1]) if m else 0
    if update and update%2000==0 and p.with_suffix('.json').exists():
        rows.append(dict(update=update,path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
print(json.dumps(sorted(rows,key=lambda r:r['update'])))
""".replace("SESSION",repr(session))
    result = subprocess.run(["ssh",host,"python3 -c "+shlex.quote(code)],text=True,capture_output=True,check=True)
    return json.loads(result.stdout)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host",default="kaiser")
    p.add_argument("--session",required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--python",default=sys.executable)
    p.add_argument("--poll-seconds",type=float,default=45.)
    p.add_argument("--once",action="store_true")
    a = p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    cache = a.output/"checkpoints";cache.mkdir(exist_ok=True)
    while True:
        try:
            rows = discover(a.host,a.session)
            for row in rows:
                out = a.output/f"review_{row['update']:08d}"
                if (out/"summary.json").exists(): continue
                checkpoint = cache/f"update_{row['update']:08d}_review.pt"
                subprocess.run(["scp",a.host+":"+row["path"],str(checkpoint)],check=True)
                if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != row["sha256"]:
                    raise ValueError("Fetched review checkpoint SHA-256 changed")
                cmd = [a.python,"-u",str(ROOT/"scripts/evaluate_wheelbipe_scenarios.py"),"--suite","retention",
                    "--checkpoint",str(checkpoint),"--output",str(out),"--training-profile","foundation-jump-v9",
                    "--course-update",str(row["update"]),"--num-envs","16","--episodes","16",
                    "--seeds","173","293","--retention-csv",str(a.output/"retention.csv")]
                with (a.output/f"review_{row['update']:08d}.log").open("w") as log:
                    subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
                print(json.dumps(dict(event="retention_complete",update=row["update"],output=str(out))),flush=True)
                warnings = json.loads((out/"warnings.json").read_text())
                if warnings: print("RETENTION WARNING "+json.dumps(warnings),flush=True)
        except (subprocess.CalledProcessError,OSError,ValueError) as error:
            print(json.dumps(dict(event="monitor_error",error=repr(error))),flush=True)
        if a.once: break
        time.sleep(max(5.,a.poll_seconds))


if __name__ == "__main__":
    main()
