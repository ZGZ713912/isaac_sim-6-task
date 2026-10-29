"""Serial zero-PPO local comparison; does not modify either source tree."""
import json
from pathlib import Path
import subprocess

ROOT = Path('/home/noir/.codex/worktrees/prone-motion-long/isaac_wheeled_rl_train')
OUT = Path('/home/noir/Documents/workspace/isaac_wheeled_rl_train/reports/local_speedup_20261029')
PYTHON = '/home/noir/miniconda3/envs/isaacsim6/bin/python'
cases = [
    ('final_graph_nominal002', ROOT, 'cuda_graph', 'nominal', '0', '0'),
    ('final_eager_requested001', OUT/'baseline_source_0c197fdd', 'eager', 'requested', '1', '9000'),
    ('final_graph_requested001', ROOT, 'cuda_graph', 'requested', '1', '9000'),
]
results = []
for name, source, backend, profile, strength, updates in cases:
    command = [PYTHON, str(ROOT/'scripts/benchmark_prone_motion.py'), '--root', str(source),
               '--output', str(OUT/name), '--backend', backend,
               '--profile', profile, '--strength', strength, '--actor-updates', updates,
               '--trace-steps', '256', '--warmup', '16', '--steps', '192', '--profile-steps', '0']
    print('START',name,flush=True)
    with (OUT/(name+'.log')).open('w') as stream:
        run = subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
    results.append(dict(name=name,command=command,exit_code=run.returncode))
    (OUT/'final_matrix.json').write_text(json.dumps(results,indent=2)+'\n')
    if run.returncode:
        raise SystemExit(run.returncode)
    result = json.loads((OUT/name/'result.json').read_text())
    if result['status'] != 'completed' or result['source_mismatches_at_exit']:
        raise RuntimeError('Invalid comparison: '+name)
    print('COMPLETE',name,result['step_mean_s'],flush=True)
