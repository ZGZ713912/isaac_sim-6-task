"""Mirror real bounded-round accounting without adopting a training endpoint."""
import json
import sys
import time
from pathlib import Path

sys.path[:0] = ['scripts', 'src']
from train_robust_getup import write_json

R = Path('reports/robust_policy_repair_20261024')
plan = json.loads((R / 'plan014_fallen_output001.json').read_text())
s = json.loads((Path(plan['supervisor_output']) / 'progress.json').read_text())
t = json.loads((Path(plan['training_output']) / 'progress.json').read_text())
n, attempts = t['new_optimizer_updates'], t['new_optimizer_attempts']
p = json.loads((R / 'progress.json').read_text())
p.update(status='fallen_output_round014_' + s['status'],
    new_PPO_completed=250+n, new_PPO_attempted=250+attempts,
    repair_optimizer_updates=250+n, actual_experiment_consumed=2630+attempts,
    active_learning_clock=t['completed_updates'],
    active_training=plan['training_output'] if t['status'] in ('starting', 'training') else None,
    active_evaluation=plan['evaluation_output'] if s['status'] == 'fixed_evaluation' else None,
    active_job=s.get('active_job'),
    lineage=f'Best9a clock1501 unchanged; independent014 source1501 completed{n} attempts{attempts}, branch clock{t["completed_updates"]}. Repair{250+n}, experiment{2630+attempts}; no refunds.',
    updated_unix=time.time())
if s['status'] in ('failed', 'stopped', 'review_ready', 'phase_guard_stopped'):
    p['active_supervisor'] = None
    p['active_job'] = None
    p['new_PPO_admitted'] = 0
write_json(R / 'progress.json', p)
print(json.dumps(dict(status=s['status'], real_completed=n, real_attempted=attempts,
    branch_clock=t['completed_updates'], repair_consumed=250+n, experiment_consumed=2630+attempts)))
