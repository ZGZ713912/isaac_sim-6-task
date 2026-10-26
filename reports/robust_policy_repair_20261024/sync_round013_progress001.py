"""Refresh active accounting without changing best-candidate lineage."""
import json,time
from pathlib import Path
import sys
sys.path[:0]=['src','scripts']
from train_robust_getup import write_json
R=Path('reports/robust_policy_repair_20261024');B=Path('runs/robust_policy_repair_20261024')
q=json.loads((B/'round013/progress.json').read_text())
p=json.loads((R/'progress.json').read_text())
attempted=q['new_optimizer_attempts'];done=q['new_optimizer_updates']
assert 0<=done<=attempted<=8
p.update(status='projected_common_entry_'+q['status'],new_PPO_attempted=242+attempted,
    new_PPO_completed=242+done,repair_optimizer_updates=242+done,
    actual_experiment_consumed=2622+attempted,active_learning_clock=q.get('current_learning_clock'),
    current_candidate_clock=1501,current_learning_clock=1501,
    lineage=f'Best candidate clock1501 remains unchanged. Independent013 has {done} completed and {attempted} attempted updates from1501; repair {242+done} completed; experiment {2622+attempted} consumed. No refunds.',
    updated_unix=time.time())
write_json(R/'progress.json',p)
print(json.dumps(dict(status=q['status'],attempts=attempted,completed=done,active_clock=q.get('current_learning_clock'),best_clock=1501,experiment_consumed=p['actual_experiment_consumed'])))
