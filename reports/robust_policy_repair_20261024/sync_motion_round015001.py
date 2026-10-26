"""Read native counters without changing a running learner or its frozen files."""
import json
from pathlib import Path
import sys
import time

sys.path[:0] = ['src', 'scripts']
from train_robust_getup import write_json

R = Path('reports/robust_policy_repair_20261024')
plan = json.loads((R/'plan015_motion_output001.json').read_text())
supervisor = json.loads((Path(plan['supervisor_output'])/'progress.json').read_text())
path = Path(plan['training_output'])/'progress.json'
native = json.loads(path.read_text()) if path.exists() else {}
complete, attempted = [native.get(k, 0) for k in ('new_optimizer_updates', 'new_optimizer_attempts')]
assert 0 <= complete <= attempted <= 8
progress = json.loads((R/'progress.json').read_text())
active = supervisor.get('active_job')
progress.update(status='motion_output_round015_'+supervisor['status'],
    new_PPO_completed=258+complete, new_PPO_attempted=258+attempted,
    repair_optimizer_updates=258+complete, repair_optimizer_attempts=258+attempted,
    actual_experiment_consumed=2638+attempted,
    active_job=active, active_supervisor=supervisor['pid'] if active else None,
    active_training=str(plan['training_output']) if active and active['name']=='training' else None,
    active_evaluation=str(plan['evaluation_output']) if active and active['name']=='fixed_evaluation' else None,
    active_learning_clock=1501+complete if active else None,
    active_pilot=dict(name='pilot015_motion_output', source_clock=1501,
        completed=complete, attempted=attempted, planned=8, clock=1501+complete,
        optimizer_update_in_flight=native.get('optimizer_update_in_flight', False),
        supervisor_status=supervisor['status']),
    new_PPO_admitted=8-attempted if active and active['name']=='training' else 0,
    qualified=False, selected_deployment_model=None, updated_unix=time.time())
write_json(R/'progress.json', progress)
print(json.dumps(dict(status=supervisor['status'], completed=complete, attempted=attempted,
    clock=1501+complete, repair_completed=258+complete, experiment_consumed=2638+attempted,
    optimizer_update_in_flight=native.get('optimizer_update_in_flight', False))))
