#!/usr/bin/env python3
"""Prepare a fixed 2-critic/4-actor pilot for moving-stop observability."""
from copy import deepcopy
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reproduce-frozen-50hz',action='store_true',
        help='Explicitly reproduce the historical 50 Hz experiment')
    args=parser.parse_args()
    if not args.reproduce_frozen_50hz:
        parser.error('This is a frozen 50 Hz reproduction. New versions use prepare_flat_policy_200hz.py.')
    import torch
    from wheeled_algo.command_residual import mean_from_checkpoint
    from wheeled_algo.motion_state_residual import MotionStateResidualActor,KIND
    from wheeled_tasks.chassis.flat_base_continuation import read_plan,digest,encode,write_bundle
    from wheeled_tasks.chassis.control_contract import load_chassis_control
    from wheeled_tasks.chassis.preflight import design_preflight
    torch.set_num_threads(1)
    parent=ROOT/'runs/flat_base_20261020/pilot032_intent1346/initial_state.pt'
    assert digest(parent)=='71b55c321d8dbccd1c676456870985833ff25f64bb339725d700340118bb4b17'
    state=torch.load(parent,map_location='cpu',weights_only=True);info=state['infos']
    old=read_plan(ROOT,info['plan_file']);assert digest(ROOT/info['plan_file'])==info['plan_sha256']
    negative=ROOT/'reports/flat_base_speed_20261020/comparison_envelope001/confirmation.json'
    assert json.loads(negative.read_text())['status']=='rejected_as_complete_speed_solution'
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    plan=deepcopy(old)
    plan.update(actor_adaptation=KIND,motion_state_parent=dict(checkpoint=str(parent.relative_to(ROOT)),sha256=digest(parent)),
        learning_rate=.001,additional_critic_warmup_updates=0,checkpoint_interval=1,
        training_scope='frozen_v10_with_414_parameter_motion_state_residual; true-goal high-speed and braking')
    catalogue={g['name']:g for g in old['train_groups']}
    fractions=dict(stand_mid=.04,stand_high=.02,forward_low=.02,backward_low=.02,
        forward_high=.10,forward_edge3=.10,backward_high=.10,backward_edge3=.20,
        turn_left_edge4=.05,turn_right_edge4=.05,start_stop_3_0=.10,start_stop_backward_3_0=.10)
    groups=[];limits={}
    for name,fraction in fractions.items():
        group=deepcopy(catalogue[name]);group['fraction']=fraction
        if group['skill']['kind']=='start_stop':
            group['skill']['segment_seconds']=3.;limits[name]=12.
        groups.append(group)
    for name,sign in (('yaw_start_stop_positive',1.),('yaw_start_stop_negative',-1.)):
        group=deepcopy(catalogue['turn_left_edge4'if sign>0 else'turn_right_edge4'])
        group.update(name=name,fraction=.05)
        group['skill'].update(command=[0.,sign*4.,.305],sample_amplitude=False,yaw_start_stop_segment_seconds=3.)
        group['skill'].pop('amplitude_range',None)
        groups.append(group);limits[name]=12.
    plan['train_groups']=groups;plan['group_episode_seconds']=limits
    plan_path=out/'plan.json';plan_path.write_bytes(encode(plan));assert read_plan(ROOT,plan_path)==plan
    config,seed=write_bundle(ROOT,out/'config4096',4096,plan_file=str(plan_path.relative_to(ROOT)))
    static=design_preflight(config,load_chassis_control(ROOT/config['control_math_source']),ROOT,
        stage='speed',role='train',seed=seed,num_envs=4096)
    sources=set(info['source_sha256'])|{'src/wheeled_algo/motion_state_residual.py'}
    hashes={name:digest(ROOT/name)for name in sorted(sources)}
    changed=sorted(k for k in old.keys()|plan.keys()if old.get(k)!=plan.get(k))
    revision=dict(schema='flat-training-plan-revision-v1',parent_checkpoint_sha256=digest(parent),
        old_plan_file=info['plan_file'],old_plan_sha256=info['plan_sha256'],new_plan_sha256=digest(plan_path),
        old_source_sha256=info['source_sha256'],new_source_sha256=hashes,changed_plan_fields=changed)
    (out/'revision.json').write_bytes(encode(revision))
    source=ROOT/plan['actor_bundle']/'source.onnx';preceding=mean_from_checkpoint(state,source)
    spec=plan['speed_reference'];v7=torch.load(ROOT/spec['checkpoint'],map_location='cpu',weights_only=True)
    child=MotionStateResidualActor(mean_from_checkpoint(v7,source),spec['sha256'],spec['margin_m_s'],spec['lean_gain_rad_per_m_s'],spec['encoder_gain'])
    values=child.state_dict();values.update(preceding.state_dict());child.load_state_dict(values,strict=True)
    with torch.no_grad():
        generator=torch.Generator().manual_seed(191269)
        obs=torch.randn(8192,35,generator=generator)
        assert torch.equal(child(obs),preceding(obs))
        stop=torch.zeros(2,35);stop[:,9]=-1.;stop[0,20]=5.;stop[0,21]=-5.;stop[1,6]=2.
        assert child.motion_gate(stop).min()==1.
    preparation=dict(status='prepared',qualified=False,new_optimizer_updates=0,source_sha256=hashes,
        plan_sha256=digest(plan_path),parent_checkpoint_sha256=digest(parent),negative_confirmation_sha256=digest(negative),
        causal_hypothesis='Command-only gate has zero gradient at a moving stop; expose encoder/IMU motion to the residual gate',
        trainable_parameters=414,parent_state='all v10 weights/std/reference and Adam/RNG/counters retained; reuse two pending critic updates without adding more',
        initial_identity_frames=8192,static_preflight=static,
        pilot=dict(new_critic_only_updates=2,new_actor_updates=4,new_optimizer_updates=6,num_envs=4096,
            expected_successful_updates=1344,expected_actor_updates=1160),
        revised_request_budget=dict(already_consumed_optimizer_updates=12,already_consumed_actor_updates=8,
            new_optimizer_ceiling=6,total_request_optimizer_ceiling=18,total_request_actor_ceiling=12,
            reason='new architecture and causal zero-gradient hypothesis after negative long confirmation',no_refunds=True))
    (out/'preparation.json').write_bytes(encode(preparation))
    for name in hashes:
        target=out/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/name).read_bytes())
    (out/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(status='prepared',changed=changed,trainable_parameters=414,new_updates=6)))


if __name__=='__main__':main()
