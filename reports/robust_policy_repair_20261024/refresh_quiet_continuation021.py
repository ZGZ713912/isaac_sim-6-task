"""Account actual native attempts/updates; this does not control training."""
import json
from pathlib import Path
import time

R = Path(__file__).resolve().parent
ROOT = R.parents[1]
run = ROOT / 'runs/robust_policy_repair_20261024/pilot021_quiet_continuation'
native = json.loads((run / 'progress.json').read_text())
p = json.loads((R / 'progress.json').read_text())
n, attempts = native['new_optimizer_updates'], native['new_optimizer_attempts']
row = dict(name='pilot021_quiet_continuation', updates=n, attempted=attempts, planned_updates=32,
    source_clock=1517, clock=1517 + n, status=native['status'], output=str(run.relative_to(ROOT)))
rows = p['experiments']
found = next((i for i, old in enumerate(rows) if old['name'] == row['name']), None)
if found is None:
    rows.append(row)
else:
    rows[found] = row
p.update(new_PPO_completed=330 + n, new_PPO_attempted=330 + attempts,
    actual_experiment_consumed=2710 + attempts, repair_optimizer_updates=330 + n,
    repair_optimizer_attempts=330 + attempts, active_learning_clock=1517 + n,
    updated_unix=time.time(), latest_quiet_continuation_training=dict(output=str(run.relative_to(ROOT)),
        new_completed=n, new_attempted=attempts, native_status=native['status']))
if native['status'] == 'training':
    supervisor = json.loads((R / 'training021_status.json').read_text())
    p.update(status='quiet_continuation021_training', active_training='pilot021_quiet_continuation',
        active_supervisor=supervisor['pid'], active_diagnosis=None, new_PPO_admitted=32)
if native['status'] in ('completed', 'stopped', 'failed'):
    p.update(active_training=None, active_supervisor=None, active_learning_clock=None,
             new_PPO_admitted=0, status='quiet_continuation_trial_needs_physical_evaluation')
tmp = R / 'progress.tmp'
tmp.write_text(json.dumps(p, indent=2) + '\n')
tmp.replace(R / 'progress.json')
print(json.dumps(dict(native=native['status'], completed=n, attempted=attempts,
    repair_total=330 + n, experiment_consumed=2710 + attempts)))
