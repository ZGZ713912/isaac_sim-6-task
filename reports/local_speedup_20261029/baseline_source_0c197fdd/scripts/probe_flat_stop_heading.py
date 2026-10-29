#!/usr/bin/env python3
"""Paired zero-PPO heading-memory diagnosis using the frozen native evaluator.

Arm 2 uses heading memory, arm 3 uses the same checkpoint with the original
command reference, and arm 0 is the old parent. No checkpoint is written.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def seal_before_close(env, seal):
    """Seal before Kit fast shutdown can terminate the Python process."""
    original_close = env.close

    def close(*args, **kwargs):
        try:
            seal()
        finally:
            original_close(*args, **kwargs)

    env.close = close


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--mode')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--candidate-variants', type=Path, nargs='+', required=True)
    parser.add_argument('--comparison-indices', type=int, nargs='+', required=True)
    parser.add_argument('--policy-plan', type=Path, required=True)
    args, _ = parser.parse_known_args()
    if args.mode != 'checkpoint_compare':
        raise ValueError('Heading diagnosis requires the checkpoint comparison protocol')
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    if (len(args.candidate_variants) != 1 or set(args.comparison_indices) != {0, 2, 3}
            or sha(args.candidate) != sha(args.candidate_variants[0])):
        raise ValueError('Require parent0, intervention2 and identical-checkpoint control3')
    plan = json.loads(args.policy_plan.read_text())
    if plan['policy_dt'] != .005 or plan['physics_dt'] != .001:
        raise ValueError('Heading diagnosis requires the frozen 200/1000Hz plan')
    output = args.output.resolve()
    from flat_stop_heading_probe import StopHeadingMemory
    import flat_rmcs_command_reference as command_module
    import probe_flat_speed_prior as evaluator
    original_install = command_module.install_reference
    audit = dict(schema='flat-stop-heading-diagnostic-v1', qualification_eligible=False,
                 qualified=False, new_optimizer_updates=0,
                 scope='Arm2 holds the last straight heading through a zero planar request; arm3 has the same weights and unchanged reference. Explicit turns invalidate the heading. This is not the current RMCS command contract.',
                 source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in (Path(__file__), ROOT/'scripts/flat_stop_heading_probe.py')})

    def seal():
        if not output.is_dir():
            return
        snapshot = {k: v for k, v in audit.items() if k != '_probe'}
        probe = audit.get('_probe')
        if probe is not None:
            snapshot['held_lane_ticks'] = probe.held_lane_ticks
        report = output/'report.json'
        if report.is_file():
            snapshot['base_report_sha256'] = sha(report)
            arms = json.loads(report.read_text()).get('checkpoint_comparison', {})
            if arms:
                snapshot['same_candidate_weights'] = arms['2']['sha256'] == arms['3']['sha256']
                if not snapshot['same_candidate_weights']:
                    raise ValueError('Heading intervention and control weights differ')
        (output/'heading_memory_override.json').write_text(json.dumps(snapshot, indent=2)+'\n')

    def install(env, contract, **kwargs):
        reference = original_install(env, contract, **kwargs)
        from torch import tensor
        import torch
        selected = tensor([g.startswith('g2_') for g in env.scene_groups], device=env.device, dtype=torch.bool)
        baseline = tensor([g.startswith('g3_') for g in env.scene_groups], device=env.device, dtype=torch.bool)
        if not bool(selected.any()) or int(selected.sum()) != int(baseline.sum()):
            raise ValueError('Require equally populated intervention2 and same-weight control3')
        probe = StopHeadingMemory(reference, selected)
        reference.update = probe.update
        audit.update(intervention_lanes=int(selected.sum()), control_lanes=int(baseline.sum()),
                     held_lane_ticks=0)
        audit['_probe'] = probe
        seal()
        seal_before_close(env, seal)
        return reference

    command_module.install_reference = install
    try:
        status = evaluator.main()
    finally:
        command_module.install_reference = original_install
        seal()
    return status


if __name__ == '__main__':
    raise SystemExit(main())
