"""Authenticate the narrowly scoped50Hz-policy200-to1000Hz physics change."""
import hashlib
import json
from pathlib import Path


def validate_flat_clock(root, plan):
    """Authenticate legacy clocks and the explicit 200 Hz policy version."""
    root = Path(root)
    timing = json.loads((root / plan['control_file']).read_text())['timing']
    if plan['schema'] in ('flat-base-finetune-plan-v18', 'flat-base-finetune-plan-v19'):
        from .flat_policy_rate import CONTROL
        if plan['control_file'] != CONTROL:
            raise ValueError('200 Hz policy requires its explicit control source')
        expected = dict(physics_dt=.001, policy_dt=.005, pc_control_dt=.001, decimation=5)
        if (plan['physics_dt'], plan['policy_dt']) != (.001, .005):
            raise ValueError('v18 requires a 200 Hz policy and 1000 Hz physics')
    elif plan['policy_dt'] != .02:
        raise ValueError('Legacy flat plans must retain their saved 50 Hz policy')
    elif plan['physics_dt'] == .005:
        expected = dict(physics_dt=.005, policy_dt=.02, pc_control_dt=.005, decimation=4)
    elif plan['physics_dt'] == .001 and plan['schema'] in ('flat-base-finetune-plan-v15', 'flat-base-finetune-plan-v16', 'flat-base-finetune-plan-v17'):
        if (plan['control_file'] != 'contracts/v6_wheel_response_control_v1.json' or
                plan['control_sha256'] != 'cc30b2b0572999224c8feb79ffa7afb3df1b8a28800bca4a153663b9ea8d4bf8'):
            raise ValueError('1000Hz flat training requires its exact authenticated control source')
        expected = dict(physics_dt=.001, policy_dt=.02, pc_control_dt=.001, decimation=20)
    else:
        raise ValueError('Undeclared flat feedback/physics clock')
    if any(timing.get(k) != v for k,v in expected.items()):
        raise ValueError('Physics, policy, feedback-PD and decimation disagree')
    return expected


def validate_flat_clock_transition(root, old, new, parent_control_sha256):
    """No model, gains, action timing, discount, population or rewards migrate."""
    root = Path(root)
    before, after = validate_flat_clock(root, old), validate_flat_clock(root, new)
    if (before['physics_dt'], after['physics_dt']) != (.005,.001):
        raise ValueError('Only the declared200-to1000Hz transition is supported')
    if parent_control_sha256 != old['control_sha256']:
        raise ValueError('Parent control hash is inconsistent with its saved plan')
    allowed = {'schema','physics_dt','control_file','control_sha256','training_scope','additional_critic_warmup_updates'}
    changed = {k for k in old.keys() | new.keys() if old.get(k) != new.get(k)}
    if not {'schema','physics_dt','control_file','control_sha256'} <= changed or changed - allowed:
        raise ValueError('The clock transition also changed undeclared training semantics')
    controls = []
    for plan in (old,new):
        path = root / plan['control_file']
        if hashlib.sha256(path.read_bytes()).hexdigest() != plan['control_sha256']:
            raise ValueError('Frozen control source changed')
        controls.append(json.loads(path.read_text()))
    ignored = {'control_id','simulation_timing_approximation','timing'}
    if {k:v for k,v in controls[0].items() if k not in ignored} != {
            k:v for k,v in controls[1].items() if k not in ignored}:
        raise ValueError('The control transition changed gains, geometry or actuator semantics')
    if any(old[k] != new[k] for k in ('gamma','lam','num_steps_per_env','policy_dt')):
        raise ValueError('Policy rollout and discount timing changed')
    return dict(schema='flat-feedback-clock-transition-v1', old=before, new=after,
        parent_control_sha256=parent_control_sha256, new_control_sha256=new['control_sha256'],
        changed_plan_fields=sorted(changed), policy_clock_unchanged=True,
        gamma_gae_and_rollout_unchanged=True, gains_geometry_actuators_unchanged=True,
        saved_models_optimizer_and_RNG='exact before fresh physical reset',
        hardware_equivalence_verified=False)
