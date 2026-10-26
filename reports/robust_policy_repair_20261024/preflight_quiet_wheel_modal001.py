"""Recorded-action scope and actual wrapper import before native admission."""
from copy import deepcopy
import ast
import json
from pathlib import Path
import sys
import time

sys.path[:0] = ['src', 'scripts', str(Path(__file__).resolve().parent)]
import numpy as np
import torch
from export_robust_getup import load_mean
from train_robust_getup import digest, write_json
from quiet_wheel_modal_probe001 import apply_modal

torch.set_num_threads(1)
R = Path('reports/robust_policy_repair_20261024')
B = Path('runs/robust_policy_repair_20261024')
cp = B / 'quiet_stance050003/model_probe.pt'
old, info = load_mean(cp)
neutral, probe = deepcopy(old), deepcopy(old)
apply_modal(neutral, neutral=True)
weights = apply_modal(probe)
allowed = {'quiet_velocity.' + x + '.weight' for x in ('low', 'mid', 'high')}
checks = dict(neutral_state_exact=all(torch.equal(v, neutral.state_dict()[k]) for k, v in old.state_dict().items()),
    only_three_existing_feedback_tensors_changed=all(torch.equal(v, probe.state_dict()[k]) for k, v in old.state_dict().items() if k not in allowed),
    all_existing_leg_feedback_rows_exact=all(torch.equal(v.weight[:2], probe.quiet_velocity[k].weight[:2]) for k, v in old.quiet_velocity.items()))
rows = []
fixed = Path(json.loads((R / 'diagnosis_plan036.json').read_text())['jobs'][0]['output'])
for directory in (fixed, B / 'diagnosis038_quiet_stance050_development'):
    path = directory / 'evaluation/trajectory.npz'
    report = json.loads((path.parent / 'report.json').read_text())
    assert report['checkpoint_sha256'] == digest(cp)
    with np.load(path) as archive:
        obs = archive['observations'].reshape(-1, 35)
    legs = outside = neutral_exact = True
    changed = 0
    peak = 0.
    with torch.inference_mode():
        for values in np.array_split(obs, (len(obs) + 511) // 512):
            x = torch.from_numpy(values.copy())
            a, b, zero = old(x), probe(x), neutral(x)
            eligible = old.quiet_stance_gate(x).ravel() > 0.
            legs &= torch.equal(a[:, :4], b[:, :4])
            outside &= torch.equal(a[~eligible], b[~eligible])
            neutral_exact &= torch.equal(a, zero)
            changed += int((a != b).any(-1).sum())
            peak = max(peak, float((a[:, 4:] - b[:, 4:]).abs().max()))
            assert torch.isfinite(b).all() and (b.abs() <= old.action_bound + 1e-6).all()
    rows.append(dict(trace=str(path), trace_sha256=digest(path), frames=len(obs),
        legs_exact=legs, outside_quiet_exact=outside, neutral_exact=neutral_exact,
        changed_frames=changed, wheel_action_delta_peak=peak))
checks.update(common_row_sums_exact=all(torch.equal(v.weight[2:].sum(-1),probe.quiet_velocity[k].weight[2:].sum(-1)) for k,v in old.quiet_velocity.items()),
    independent_wheel_coordinates=all(v.weight[2,1]==0. and v.weight[3,0]==0. for v in probe.quiet_velocity.values()),
    recorded_legs_exact=all(x['legs_exact'] for x in rows),
    outside_quiet_exact=all(x['outside_quiet_exact'] for x in rows),
    neutral_function_exact=all(x['neutral_exact'] for x in rows),
    effect_present=all(x['changed_frames'] > 0 for x in rows),
    unchanged_checkpoint=digest(cp) == 'c025624014b5ff47d0967f47291d7c9aa16e2d4f89d4ba1de30d9a1c90deb5e9')
wrapper = R / 'evaluate_quiet_wheel_modal001.py'
tree = ast.parse(wrapper.read_text())
# Import definitions and hook setup, excluding only the final native main try.
tree.body = tree.body[:-1]
fixture = R / 'quiet_wheel_modal_wrapper_CPU001'
fixture.mkdir(exist_ok=False)
sys.argv = [str(wrapper), '--checkpoint', str(cp), '--output', str(fixture)]
namespace = {'__file__': str(wrapper.resolve()), '__name__': 'quiet_gain_wrapper_fixture'}
exec(compile(tree, str(wrapper), 'exec'), namespace)
try:
    actual, _ = namespace['load'](cp)
    checks['actual_wrapper_mean_matches_probe'] = all(torch.equal(v, actual.state_dict()[k]) for k, v in probe.state_dict().items())
    record = json.loads((fixture / 'runtime_function_readback.json').read_text())
    checks['explicit_unqualified_override'] = record['eligible_for_checkpoint_mean_qualification'] is False
finally:
    namespace['export'].load_mean = namespace['native_load']
    namespace['evaluate'].write_json = namespace['native_write']
result = dict(schema='quiet-wheel-common-preserved-diagonal-preflight-v1', checks={k: bool(v) for k, v in checks.items()},
    weights=weights, rows=rows, checkpoint=str(cp), checkpoint_sha256=digest(cp),
    wrapper_sha256=digest(wrapper), probe_sha256=digest(R / 'quiet_wheel_modal_probe001.py'),
    proposed_development_gate='All21 stand episodes survive; at least19/21 pose/position passes, no previous c025 development pass lost; same physical safeguards. Diagnostic only, no PPO/adoption/export.',
    new_PPO=0, training_baseline_consumed=2646,
    accounting_scope='CPU probe adds0PPO;2646 is the prior completed baseline, excluding concurrent native016 work. See round016/progress.json for its live charged attempts.',
    qualification=None, created_unix=time.time())
write_json(R / 'quiet_wheel_modal_preflight001.json', result)
print(json.dumps(dict(checks=checks, rows=rows), indent=2))
assert all(checks.values()), checks
