"""One predeclared opposite-direction entry NN edit; zero optimizer updates."""
from pathlib import Path
from copy import deepcopy
import hashlib
import json
import subprocess
import sys
import time

sys.path[:0] = ['src', 'scripts']
import numpy as np
import torch
from export_robust_getup import load_mean
from migrate_gap_entry_preservation import migrate
from wheeled_algo.gap_entry_handoff import handoff_entry_gate

torch.set_num_threads(1)
B = Path('runs/robust_policy_repair_20261024')
R = Path('reports/robust_policy_repair_20261024')
source = B/'final_pair_circular_corrected002/model_migrated.pt'
control = B/'periodic_entry_encoder001/model_probe.pt'
teacher = B/'entry_positive030_teacher001/model_probe.pt'
split = B/'entry_positive030_split001/model_migrated.pt'
candidate = B/'entry_positive030_periodic001/model_probe.pt'
assert json.loads((R/'diagnosis_queue021.json').read_text())['status'] == 'completed'
prior, _ = load_mean(source)
state = torch.load(source, map_location='cpu', weights_only=False)
changed = deepcopy(state)
key = 'mlp.mean.recovery.4.bias'
delta = torch.tensor([.3, .3, -.3, -.3, 0., 0.])
changed['actor_state_dict'][key] += delta
changed['infos']['recovery_common_bias_probe'] = dict(schema='opposite-common-bias-teacher-v1',
    source_checkpoint=str(source.resolve()), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    tensor=key, bias_delta=delta.tolist(), new_PPO=0,
    scope='Offline reference head for explicit entry-only cloning; this shared-head teacher is never run or adopted')
teacher.parent.mkdir(exist_ok=False, parents=True)
torch.save(changed, teacher)
load_mean(teacher)
migrate(source, teacher, split, 2603)
subprocess.run([sys.executable, 'scripts/migrate_periodic_entry_encoder.py', '--source', str(split),
                '--output', str(candidate)], check=True)
old, oi = load_mean(control)
new, ni = load_mean(candidate)
old_state, new_state = old.state_dict(), new.state_dict()
exact_other = all(torch.equal(v, new_state[k]) for k,v in old_state.items() if k != 'entry_recovery.4.bias')
assert exact_other
torch.testing.assert_close(new_state['entry_recovery.4.bias']-old_state['entry_recovery.4.bias'],
                           delta, atol=3e-8, rtol=0.)
rows = []
for name in ('diagnosis012_final_arithmetic_fixed/evaluation',
             'diagnosis012_final_arithmetic_nominal/robust',
             'diagnosis012_final_arithmetic_rmcs/evaluation'):
    path = B/name/'trajectory.npz'
    z = np.load(path)
    frames = z['observations'].reshape(-1, 35)
    unchanged = count = 0
    finite = wheels_exact = True
    max_delta = 0.
    for start in range(0, len(frames), 2048):
        obs = torch.tensor(frames[start:start+2048])
        outside = handoff_entry_gate(obs)[:, 0] == 0.
        with torch.inference_mode():
            x, y = old(obs), new(obs)
        unchanged += int((x[outside] == y[outside]).all(-1).sum())
        count += int(outside.sum())
        wheels_exact &= torch.equal(x[:, 4:], y[:, 4:])
        finite &= bool(torch.isfinite(y).all())
        d = (x[:, :4]-y[:, :4])*.25
        max_delta = max(max_delta, float(torch.atan2(d.sin(), d.cos()).abs().max()))
    rows.append(dict(trace=str(path), frames=len(frames), entry0_frames=count,
        entry0_frames_bit_exact=unchanged, wheel_outputs_bit_exact=wheels_exact, finite=finite,
        maximum_physical_target_delta_rad=max_delta))
checks = dict(exact_other_weights=exact_other, only_four_entry_bias_values_changed=True,
    clocks_equal=oi['completed_updates']==ni['completed_updates']==1501,
    entry0_all_bit_exact=all(x['entry0_frames']==x['entry0_frames_bit_exact'] for x in rows),
    all_wheel_outputs_exact=all(x['wheel_outputs_bit_exact'] for x in rows),
    all_finite=all(x['finite'] for x in rows))
out = dict(schema='entry-opposite-bias-periodic-preflight-v1', checkpoint=str(candidate),
    sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(), reference=str(control),
    reference_sha256=hashlib.sha256(control.read_bytes()).hexdigest(), checks=checks, replays=rows,
    intervention='Only the four existing entry NN common-output biases: +.30 left / -.30 right, opposite to earlier -.30 / +.30. Both use periodic-chart-v1.',
    hypothesis='The seam-smoothed NN settles with wheels above the body. Test the untested opposite common-drive sign, inferred from measured common angles and the table support configuration.',
    new_PPO=0, actual_experiment_consumed=2603, qualification=None, hardware_started=False, created_unix=time.time())
(R/'entry_opposite_bias_preflight001.json').write_text(json.dumps(out, indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='replays'}, indent=2))
assert all(checks.values())
