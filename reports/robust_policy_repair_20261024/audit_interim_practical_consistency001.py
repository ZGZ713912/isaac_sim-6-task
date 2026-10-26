"""Keep the original12s endpoint metric distinct from the8s strict pilot gate."""
import json
from pathlib import Path
import sys
import time

sys.path[:0] = ['src', 'scripts']
from torch.utils.tensorboard import SummaryWriter
from analyze_robust_getup import analyze
from train_robust_getup import digest, write_json

R = Path('reports/robust_policy_repair_20261024')
B = Path('runs/robust_policy_repair_20261024')
protocol = Path('reports/unified_robust_20261021/acceptance_protocol004.json')
assert digest(protocol) == '05e6469f6da54b726fc2bfb5e53ff7f90f996bfbcceaf942e453f6437ee39ab3'
spec = json.loads(protocol.read_text())
models = []
for label, directory, clock in [('best9a', 'diagnosis025_motion_damping080_fixed', 1501),
                               ('fallen014', 'pilot014_fallen_fixed', 1509),
                               ('motion015', 'pilot015_motion_fixed', 1509)]:
    path = B / directory / 'evaluation'
    report = json.loads((path / 'report.json').read_text())
    result = analyze(path)
    assert report['seconds'] == 8.
    groups = {}
    for name, prefix in [('entry', 'real_start'), ('fallen', ('left_symmetric', 'right_symmetric', 'inverted_symmetric'))]:
        lanes = [row for row in report['lanes'] if row['name'].startswith(prefix)]
        groups[name] = dict(episodes=len(lanes), survived=sum(x['survived'] for x in lanes),
            recovered=sum(x['recovered'] for x in lanes),
            uninterrupted_two_second_success=sum(x['followup_success'] and x['survived'] for x in lanes),
            diagnostic_8s_tail95_success=sum(result['practical_followup_by_lane'][x['lane']] for x in lanes))
    writer = SummaryWriter(str(path / 'metrics'), filename_suffix='.interim-practical-consistency')
    for group, values in groups.items():
        for name, value in values.items():
            writer.add_scalar('InterimEndpointDiagnostic/' + group + '/' + name, value, clock)
    writer.close()
    models.append(dict(label=label, evaluation=str(path), report_sha256=digest(path / 'report.json'),
        trajectory_sha256=report['trajectory_sha256'], duration_seconds=8., groups=groups,
        original_12s_protocol_qualification=False))
out = dict(schema='interim-versus-original-endpoint-metric-audit-v1',
    original_protocol=str(protocol), original_protocol_sha256=digest(protocol),
    original_complete_definition=spec['complete_definition'], original_preserved_metrics=spec['preserved_metrics'],
    models=models,
    interpretation='The completed8s diagnostic tail is not the required12s complete metric. Original strict failure decisions remain unchanged; this audit grants no adoption, expanded-test admission or PPO extension.',
    new_PPO=0, actual_experiment_consumed=2646, repair_optimizer_updates=266,
    qualification=None, hardware_started=False, created_unix=time.time())
write_json(R / 'interim_practical_consistency001.json', out)
print(json.dumps(dict(models=models), indent=2))
