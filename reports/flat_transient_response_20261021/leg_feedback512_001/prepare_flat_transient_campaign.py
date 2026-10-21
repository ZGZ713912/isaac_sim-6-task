#!/usr/bin/env python3
"""Freeze a full-state local response stage and its exact migration receipt."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--num-envs', type=int, choices=(128, 512, 1024), default=512)
    parser.add_argument('--stage', choices=('pilot', 'long5000'), default='pilot')
    parser.add_argument('--admission', type=Path, nargs=2)
    parser.add_argument('--extended-dwell', action='store_true',
        help='Expose full-speed braking with the diagnosed eight-second training phases')
    parser.add_argument('--parent-leg-feedback', action='store_true',
        help='Relax only the frozen parent leg recurrence; learned leg intent and wheels remain direct')
    args = parser.parse_args()
    import torch
    from wheeled_tasks.chassis.flat_base_continuation import digest, encode, read_plan
    from wheeled_tasks.chassis.transient_campaign_plan import make_campaign_plan
    from train_flat_base import restored_training_frames
    parent = args.parent.resolve()
    saved = torch.load(parent, map_location='cpu', weights_only=True)
    info = saved['infos']
    old = read_plan(ROOT, info['plan_file'])
    if not old.get('transient_response') or info['critic_warmup_remaining']:
        raise ValueError('Continue the response actor only after its original critic warmup')
    if digest(ROOT / info['plan_file']) != info['plan_sha256']:
        raise ValueError('Parent plan changed')
    base_file = 'reports/flat_transient_response_20261021/candidate003/plan.json'
    base = read_plan(ROOT, base_file)
    admissions = [dict(file=str(p.resolve().relative_to(ROOT)), sha256=digest(p))
                  for p in args.admission or []]
    spec = dict(base_plan_file=base_file, base_plan_sha256=digest(ROOT/base_file),
        checkpoint=str(parent.relative_to(ROOT)), checkpoint_sha256=digest(parent),
        parent_plan_file=info['plan_file'], parent_plan_sha256=info['plan_sha256'],
        num_envs=args.num_envs, stage=args.stage,
        lineage_start_successful_updates=1346, short_update_ceiling=512,
        admission=admissions)
    plan = make_campaign_plan(base, spec)
    if args.extended_dwell:
        from wheeled_tasks.chassis.transient_dwell_plan import make_dwell_plan, EVIDENCE
        plan = make_dwell_plan(base, spec, digest(ROOT/EVIDENCE))
    if args.parent_leg_feedback:
        if not args.extended_dwell:
            raise ValueError('Parent-leg feedback retains the diagnosed full-speed dwell')
        from wheeled_tasks.chassis.transient_leg_feedback_plan import make_leg_feedback_plan, EVIDENCE, DECISION
        from wheeled_tasks.chassis.transient_dwell_plan import EVIDENCE as DWELL_EVIDENCE
        plan = make_leg_feedback_plan(base, spec, digest(ROOT/DWELL_EVIDENCE),
                                      digest(ROOT/EVIDENCE), digest(ROOT/DECISION))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    plan_file = output/'plan.json'
    plan_file.write_bytes(encode(plan))
    read_plan(ROOT, plan_file)
    updates = 64 if args.stage == 'pilot' else 5000
    subprocess.run([sys.executable, str(ROOT/'scripts/train_flat_base.py'),
        '--plan', str(plan_file), '--output', str(output/'preflight'),
        '--updates', str(updates), '--preflight-only'], cwd=ROOT, check=True)
    static = json.loads((output/'preflight/progress.json').read_text())
    revision = dict(schema='flat-training-plan-revision-v1',
        parent_checkpoint_sha256=digest(parent), old_plan_file=info['plan_file'],
        old_plan_sha256=info['plan_sha256'], new_plan_sha256=digest(plan_file),
        old_source_sha256=info['source_sha256'], new_source_sha256=static['source_sha256'],
        changed_plan_fields=sorted(k for k in old.keys() | plan.keys() if old.get(k) != plan.get(k)))
    if info['num_envs'] != args.num_envs:
        revision['population_transition'] = dict(schema='flat-training-population-transition-v1',
            old_num_envs=info['num_envs'], new_num_envs=args.num_envs,
            num_steps_per_env=old['num_steps_per_env'],
            inherited_successful_updates=info['successful_updates_total'],
            inherited_successful_training_frames=restored_training_frames(info, old),
            additional_critic_warmup_updates=0)
    (output/'revision.json').write_bytes(encode(revision))
    (output/'preparation.json').write_bytes(encode(dict(status='prepared_no_optimizer_updates',
        parent_checkpoint_sha256=digest(parent), plan_sha256=digest(plan_file),
        stage=args.stage, num_envs=args.num_envs, new_optimizer_updates=0,
        requested_updates=updates, frames_per_update=args.num_envs*plan['num_steps_per_env'],
        inherited_training_transitions=restored_training_frames(info, old),
        population_comparison='updates and rollout transitions are both retained; populations are not exposure-equivalent',
        source_sha256=static['source_sha256'], hardware_motion_authorized=False)))
    (output/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(status='prepared', stage=args.stage, num_envs=args.num_envs,
                          updates=updates, plan=str(plan_file))))


if __name__ == '__main__':
    main()
