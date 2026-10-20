#!/usr/bin/env python3
"""Prepare one bounded, backward-weighted speed-reference pilot; no simulation."""
from copy import deepcopy
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import torch
    from wheeled_algo.command_residual import mean_from_checkpoint
    from wheeled_algo.speed_reference import SpeedReferenceActor, KIND
    from wheeled_tasks.chassis.flat_base_continuation import digest, encode, read_plan, write_bundle
    from wheeled_tasks.chassis.control_contract import load_chassis_control
    from wheeled_tasks.chassis.preflight import design_preflight
    torch.set_num_threads(1)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    probe_path = ROOT/'reports/flat_base_speed_20261020/lean_probe002/report.json'
    probe = json.loads(probe_path.read_text())
    if probe['status'] != 'evaluated' or probe['new_optimizer_updates'] != 0:
        raise ValueError('Require the completed, separate zero-PPO initialization probe')
    parent_path = ROOT/'runs/flat_base_20261019/pilot029_static_grip1338/model_001338.pt'
    assert digest(parent_path) == '39f7f2d364d82905741478fd254c7fdde3d9c101a613cfba1ba79d14a483fd73'
    state = torch.load(parent_path, map_location='cpu', weights_only=True)
    info = state['infos']
    old = read_plan(ROOT, info['plan_file'])
    assert digest(ROOT/info['plan_file']) == info['plan_sha256']
    gains, selection = [], {}
    for direction in ('forward', 'backward'):
        candidates = []
        for gain in probe['gains']:
            rows = [v for n,v in probe['result']['cases'].items()
                    if n.startswith(f'g{gain:g}_') and (n.endswith(direction+'_2') or n.endswith(direction+'_3'))]
            if len(rows) != 8:
                raise ValueError('Incomplete initialization cohort')
            complete = sum(v['full_horizon_episodes'] for v in rows)
            supported = all(v['frames'] > 0 for v in rows)
            score = sum(v['vx_mae_m_s'] for v in rows)/len(rows) if supported else float('inf')
            candidates.append((16-complete, score, gain))
        best = min(candidates)
        # A controller that falls does not earn initialization credit from low
        # MAE among the remaining frames. Fall-free baseline is always present.
        if best[0] != 0:
            raise ValueError('No fully surviving initialization for '+direction)
        gains.append(best[2])
        selection[direction] = dict(selected_gain=best[2], objective='survival first, then body-vx MAE',
            candidates=[dict(failed_episodes=a, body_vx_mae_m_s=b if b != float('inf') else None, gain=c) for a,b,c in candidates])
    plan = deepcopy(old)
    plan.update(actor_adaptation=KIND, additional_critic_warmup_updates=2,
        speed_reference=dict(checkpoint=str(parent_path.relative_to(ROOT)), sha256=digest(parent_path),
            margin_m_s=[.15,.3,.05,.05], lean_gain_rad_per_m_s=gains),
        training_scope='six_parameter_reachable_speed_and_lean_reference_frozen_parent_backward_emphasis')
    catalogue = {g['name']:g for g in old['train_groups']}
    fractions = {'stand_mid':.04, 'stand_high':.02, 'forward_low':.02, 'backward_low':.02,
        'forward_high':.15, 'forward_edge3':.05, 'backward_high':.20, 'backward_edge3':.10,
        'turn_left':.10, 'turn_left_edge4':.05, 'turn_right':.10, 'turn_right_edge4':.05,
        'start_stop_3_0':.05}
    groups = []
    for name,fraction in fractions.items():
        group = deepcopy(catalogue[name]); group['fraction']=fraction
        if name in ('turn_left','turn_right'):
            group['skill']['amplitude_range']=[.5,1.]
        groups.append(group)
    backward_start = deepcopy(catalogue['start_stop_3_0'])
    backward_start.update(name='start_stop_backward_3_0', fraction=.05)
    backward_start['skill']['command'][0] = -3.
    groups.append(backward_start)
    plan['train_groups'] = groups
    plan['group_episode_seconds'] = {}
    plan_path=out/'plan.json'; plan_path.write_bytes(encode(plan))
    assert read_plan(ROOT,plan_path) == plan
    config, seed = write_bundle(ROOT,out/'config4096',4096,plan_file=str(plan_path.relative_to(ROOT)))
    static = design_preflight(config,load_chassis_control(ROOT/config['control_math_source']),ROOT,
        stage='speed',role='train',seed=seed,num_envs=4096)
    parent_mean = mean_from_checkpoint(state,ROOT/plan['actor_bundle']/'source.onnx')
    child = SpeedReferenceActor(parent_mean,digest(parent_path),plan['speed_reference']['margin_m_s'],gains)
    obs = torch.randn(4096,35); obs[:,0].clamp_(-.75,.75); obs[:,2].clamp_(-1.5,1.5)
    with torch.no_grad():
        assert torch.equal(parent_mean(obs),child(obs))
    proof = child.verify_original_from_source_full(ROOT/plan['actor_bundle']/'source.onnx')
    sources = set(info['source_sha256']) | {'src/wheeled_algo/speed_reference.py'}
    hashes = {name:digest(ROOT/name) for name in sorted(sources)}
    changed = sorted(k for k in old.keys()|plan.keys() if old.get(k)!=plan.get(k))
    revision=dict(schema='flat-training-plan-revision-v1',parent_checkpoint_sha256=digest(parent_path),
        old_plan_file=info['plan_file'],old_plan_sha256=info['plan_sha256'],new_plan_sha256=digest(plan_path),
        old_source_sha256=info['source_sha256'],new_source_sha256=hashes,changed_plan_fields=changed)
    (out/'revision.json').write_bytes(encode(revision))
    receipt=dict(status='prepared',recorded_at=datetime.now(timezone.utc).isoformat(),
        qualified=False,trainable_parameters=6,new_optimizer_updates=0,
        probe_sha256=digest(probe_path),selection=selection,source_sha256=hashes,
        plan_sha256=digest(plan_path),static_preflight=static,policy_declaration=proof,
        static_and_low_action_equality_frames=len(obs),
        pilot=dict(critic_updates=2,actor_updates=2,total_optimizer_updates=4,num_envs=4096,
            expected_successful_updates=1342,expected_actor_updates=1158,
            max_total_new_actor_updates_without_a_new_decision=8),
        prior_excluded_motion_pilot=dict(path='runs/flat_base_20261019/pilot030_motion_grip1348/model_001348.pt',
            optimizer_updates=10,retained_without_refund=True,
            accounting='other branch consumption; add to branch counters for campaign expenditure'))
    (out/'preparation.json').write_bytes(encode(receipt))
    for name in hashes:
        dst=out/'source'/name;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes((ROOT/name).read_bytes())
    (out/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(status='prepared',selection=selection,changed=changed,trainable_parameters=6)))


if __name__=='__main__':
    main()
