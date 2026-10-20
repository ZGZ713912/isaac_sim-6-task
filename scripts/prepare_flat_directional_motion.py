#!/usr/bin/env python3
"""Prepare an eight-update 200 Hz directional-motion pilot without starting PPO."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import torch
    from wheeled_tasks.chassis.flat_base_continuation import digest, encode, read_plan
    from wheeled_tasks.chassis.directional_motion_plan import make_directional_plan
    parent = ROOT / 'reports/flat_base_speed_20261020/policy200_capacity002/initial_state.pt'
    if digest(parent) != 'e07adbb5a3c2c086d9cbcb38e97b3be5f09ddd4114e749d14db391d62c3c51d1':
        raise ValueError('Verified zero-update 200 Hz parent changed')
    state = torch.load(parent, map_location='cpu', weights_only=True)
    info = state['infos']
    base = read_plan(ROOT, info['plan_file'])
    if digest(ROOT / info['plan_file']) != info['plan_sha256'] or info['critic_warmup_remaining'] != 0:
        raise ValueError('Unexpected parent plan or pending warmup')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    plan = make_directional_plan(ROOT, base, str(parent.relative_to(ROOT)), digest(parent),
                                 info['plan_file'], info['plan_sha256'])
    plan_path = out / 'plan.json'
    plan_path.write_bytes(encode(plan))
    if read_plan(ROOT, plan_path) != plan:
        raise ValueError('Directional plan did not validate')
    subprocess.run([sys.executable, str(ROOT / 'scripts/train_flat_base.py'), '--plan', str(plan_path),
        '--output', str(out / 'preflight'), '--updates', '8', '--preflight-only'], cwd=ROOT, check=True)
    static = json.loads((out / 'preflight/progress.json').read_text())
    revision = dict(schema='flat-training-plan-revision-v1', parent_checkpoint_sha256=digest(parent),
        old_plan_file=info['plan_file'], old_plan_sha256=info['plan_sha256'], new_plan_sha256=digest(plan_path),
        old_source_sha256=info['source_sha256'], new_source_sha256=static['source_sha256'],
        changed_plan_fields=sorted(k for k in base.keys() | plan.keys() if base.get(k) != plan.get(k)))
    (out / 'revision.json').write_bytes(encode(revision))
    preparation = dict(status='prepared_no_optimizer_updates', qualified=False, parent_checkpoint=str(parent),
        parent_checkpoint_sha256=digest(parent), new_optimizer_updates=0,
        causal_hypotheses=[
            'Previous first-actor KL 67.84: bound frozen features and project Adam mean changes in rollout std units',
            'Shared correction can couple directions: separate forward/reverse/positive-yaw/negative-yaw banks',
            'Command-zero braking must remain trainable from encoder rolling and IMU yaw',
            'Low-friction cold launch: reachable straight reference retains original six-action balance response'],
        new_actor_parameters=1656, old_actor_std_critic_Adam_RNG_counters='retained; only new bank Adam slots empty',
        reference_initialization='measured reverse prior plus forward launch reference; no PPO credit',
        pilot=dict(num_envs=4096, rollout_steps=384, policy_hz=200, physics_hz=1000,
            new_critic_only_updates=2, new_actor_updates=6, new_optimizer_updates=8),
        local_request_budget=dict(previous_optimizer_updates=18, additional_optimizer_ceiling=8,
            request_optimizer_ceiling=26, previous_other_motion_branch_optimizer_updates=10, no_refunds=True),
        source_sha256=static['source_sha256'], plan_sha256=digest(plan_path),
        evaluation='independent native parent/initial/trained matched forward reverse yaw and complete stops; no automatic big run')
    (out / 'preparation.json').write_bytes(encode(preparation))
    (out / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(status=preparation['status'], new_actor_parameters=1656, pilot_updates=8)))


if __name__ == '__main__':
    main()
