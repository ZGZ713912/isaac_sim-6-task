"""Physical pairing for a named, explicitly excluded deterministic mean diagnostic."""
import json
from pathlib import Path
import numpy as np
from compare_gap_robustness import read
from train_robust_getup import digest


def pair(before, after, *, override_key, schema):
    before, after = Path(before), Path(after)
    b, rb, sb, tb = read(before)
    a, ra, sa, ta = read(after)
    assert b.get('evaluation_overrides', {}) == {}
    override = a.get('evaluation_overrides', {})
    assert set(override) == {override_key} and ra['evaluation_overrides'] == override
    meta = override[override_key]
    assert meta['schema'] == schema and meta['eligible_for_checkpoint_mean_qualification'] is False
    assert meta['new_PPO'] == 0
    fields = ('seed', 'seconds', 'severity', 'actuator_model', 'actuator_timing_profile',
              'nominal_dynamics', 'ground_friction_range')
    settings_exact = all(b.get(k) == a.get(k) for k in fields)
    populations_exact = [r['name'] for r in sb['rows']] == [r['name'] for r in sa['rows']]
    domains_exact = all(x['first_reset_domain'] == y['first_reset_domain']
        and x['material_coefficients'] == y['material_coefficients'] for x, y in zip(rb['lanes'], ra['lanes'], strict=True))
    case_fields = ('name', 'command', 'recovery', 'sequence', 'push', 'push_sign', 'pose_index', 'world_origin_m')
    cases_exact = all(all(x.get(k) == y.get(k) for k in case_fields) for x, y in zip(rb['lanes'], ra['lanes'], strict=True))
    passive = tb['passive'][0] & ta['passive'][0]
    initial = dict(actor_inputs_exact=bool(np.array_equal(tb['observations'][0], ta['observations'][0])),
        passive_mask_exact=bool(np.array_equal(tb['passive'][0], ta['passive'][0])))
    initial.update({'passive_poststep_' + k + '_exact': bool(np.array_equal(tb[k][0, passive], ta[k][0, passive]))
        for k in ('motor_position', 'motor_velocity')})
    x, y = [np.load(p / 'evaluation_precontrol_state.npz') for p in (before, after)]
    assert x.files == y.files
    initial.update({'precontrol_' + k + '_exact': bool(np.array_equal(x[k], y[k])) for k in x.files})
    startups = [json.loads((p / 'startup.json').read_text()) for p in (before, after)]
    rigid = startups[0]['dynamics_randomization']['sampled_properties_sha256'] == startups[1]['dynamics_randomization']['sampled_properties_sha256']
    checks = dict(settings=settings_exact, populations=populations_exact, cases=cases_exact,
        sensor_actuator_material=domains_exact, rigid=rigid, initial=all(initial.values()))
    out = dict(schema='declared-mean-probe-physical-pairing-v1', checks=checks, initial_exact=initial,
        sensor_actuator_material_exact=domains_exact, rigid_parameters_exact=rigid,
        precontrol_physical_state_captured=True, intervention=meta,
        before_trajectory_sha256=digest(before / 'evaluation/trajectory.npz'),
        after_trajectory_sha256=digest(after / 'evaluation/trajectory.npz'),
        standard_policy_only_comparison_accepted=False, eligible_for_checkpoint_mean_qualification=False,
        qualification=None, hardware_ready=False)
    assert all(checks.values()), checks
    return out
