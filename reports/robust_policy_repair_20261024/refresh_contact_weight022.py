"""Account actual native attempts/updates; this does not control training."""
import json
from pathlib import Path
import time

R = Path(__file__).resolve().parent
ROOT = R.parents[1]
run = ROOT / 'runs/robust_policy_repair_20261024/pilot022_contact_weight16'
native = json.loads((run / 'progress.json').read_text())
p = json.loads((R / 'progress.json').read_text())
n, attempts = native['new_optimizer_updates'], native['new_optimizer_attempts']
row = dict(name='pilot022_contact_weight16', updates=n, attempted=attempts, planned_updates=16,
    source_clock=1501, clock=1501 + n, status=native['status'], output=str(run.relative_to(ROOT)))
rows = p['experiments']
found = next((i for i, old in enumerate(rows) if old['name'] == row['name']), None)
if found is None:
    rows.append(row)
else:
    rows[found] = row
p.update(new_PPO_completed=362 + n, new_PPO_attempted=362 + attempts,
    actual_experiment_consumed=2742 + attempts, repair_optimizer_updates=362 + n,
    repair_optimizer_attempts=362 + attempts, active_learning_clock=1501 + n,
    updated_unix=time.time(), latest_contact_weight_training=dict(output=str(run.relative_to(ROOT)),
        new_completed=n, new_attempted=attempts, native_status=native['status']))
if native['status'] == 'training':
    supervisor = json.loads((R / 'training022_status.json').read_text())
    p.update(status='contact_weight022_training', active_training='pilot022_contact_weight16',
        active_supervisor=supervisor['pid'], active_diagnosis=None, new_PPO_admitted=16)
if native['status'] in ('completed', 'stopped', 'failed'):
    p.update(active_training=None, active_supervisor=None, active_learning_clock=None,
             new_PPO_admitted=0, status='contact_weight_trial_needs_physical_evaluation')
tmp = R / 'progress.tmp'
tmp.write_text(json.dumps(p, indent=2) + '\n')
tmp.replace(R / 'progress.json')
print(json.dumps(dict(native=native['status'], completed=n, attempted=attempts,
    repair_total=362 + n, experiment_consumed=2742 + attempts)))
