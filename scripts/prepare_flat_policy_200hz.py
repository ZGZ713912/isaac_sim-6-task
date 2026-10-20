#!/usr/bin/env python3
"""Freeze a frequency-only continuation receipt; do not start PPO."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path,
        default=ROOT/'runs/flat_base_20261019/pilot029_static_grip1338/model_001338.pt')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import torch
    from wheeled_tasks.chassis.flat_base_continuation import digest, encode, read_plan
    from wheeled_tasks.chassis.flat_policy_rate import make_policy_rate_plan, validate_policy_rate_transition
    parent = args.parent.resolve()
    state = torch.load(parent,map_location='cpu',weights_only=True)
    info = state['infos']
    old = read_plan(ROOT,info['plan_file'])
    if digest(ROOT/info['plan_file']) != info['plan_sha256']:
        raise ValueError('Parent checkpoint plan changed')
    out = args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    new = make_policy_rate_plan(ROOT,old,info['plan_file'])
    plan_path = out/'plan.json'
    plan_path.write_bytes(encode(new))
    clock = validate_policy_rate_transition(ROOT,old,new,info['control_math_sha256'])
    preflight = out/'preflight'
    subprocess.run([sys.executable,str(ROOT/'scripts/train_flat_base.py'),'--plan',str(plan_path),
        '--output',str(preflight),'--updates','1','--preflight-only'],cwd=ROOT,check=True)
    report = json.loads((preflight/'progress.json').read_text())
    revision = dict(schema='flat-training-plan-revision-v1',parent_checkpoint_sha256=digest(parent),
        old_plan_file=info['plan_file'],old_plan_sha256=info['plan_sha256'],new_plan_sha256=digest(plan_path),
        old_source_sha256=info['source_sha256'],new_source_sha256=report['source_sha256'],
        changed_plan_fields=sorted(k for k in old.keys()|new.keys() if old.get(k)!=new.get(k)))
    (out/'revision.json').write_bytes(encode(revision))
    (out/'preparation.json').write_bytes(encode(dict(status='static_prepared_no_training',
        parent_checkpoint=str(parent),parent_checkpoint_sha256=digest(parent),
        successful_updates_total=info['successful_updates_total'],actor_updates_total=info['actor_updates_total'],
        new_optimizer_updates=0,policy_clock=clock,plan_sha256=digest(plan_path),
        static_preflight=report['static_preflight'],source_sha256=report['source_sha256'],
        native_restore_and_capacity_verified=False,behavior_qualified=False)))
    (out/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(status='static_prepared_no_training',policy_hz=200,physics_hz=1000,
        num_steps_per_env=new['num_steps_per_env'],new_optimizer_updates=0)))


if __name__ == '__main__':
    main()
