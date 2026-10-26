"""Read paired output-partition counterfactuals; never adopts a model or admits PPO."""
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path[:0] = ['src', 'scripts', 'reports/robust_policy_repair_20261024']
import torch
from export_robust_getup import load_mean
from train_robust_getup import digest, write_json
from run_gap_long_training import behavior_metrics
from fallen_output_partition_probe001 import partition
from pair_fallen_partition_diagnostic001 import pair

torch.set_num_threads(1)
R = Path('reports/robust_policy_repair_20261024')
B = Path('runs/robust_policy_repair_20261024')
plan = json.loads((R / 'diagnosis_plan040.json').read_text())
queue = json.loads((R / 'diagnosis_queue040.json').read_text())
assert queue['status'] == 'completed' and all(j['status'] == 'completed' for j in queue['jobs'])
assert all(digest(Path(n)) == h for n, h in plan['source_sha256'].items())
protocol = json.loads((R / 'fallen_partition_protocol001.json').read_text())
gate = protocol['gate']
source_mean, _ = load_mean(protocol['source'])
results = {}
for job in queue['jobs']:
    mode = job['command'][job['command'].index('--retained-fallen-output-rows') + 1]
    run = Path(job['output'])
    standing_path = R / ('fallen014_' + mode + '_standing001.json')
    comparison_path = R / ('fallen014_' + mode + '_comparison001.json')
    if not standing_path.exists():
        subprocess.run([sys.executable, 'scripts/analyze_gap_standing.py', '--run', str(run), '--output', str(standing_path)], check=True, stdout=subprocess.DEVNULL)
    write_json(comparison_path, pair(B / 'diagnosis025_motion_damping080_fixed', run))
    standing = json.loads(standing_path.read_text())
    assert standing['trajectory_sha256'] == digest(run / 'evaluation/trajectory.npz')
    comparison = json.loads(comparison_path.read_text())
    report = json.loads((run / 'evaluation/report.json').read_text())
    runtime = json.loads((run / 'runtime_function_readback.json').read_text())
    progress = json.loads((run / 'progress.json').read_text())
    mean, _ = load_mean(protocol['checkpoint'])
    expected = partition(mean, source_mean, mode)
    metrics = behavior_metrics(run, standing_path)
    impacts = sum(any('impact' in x for x in row['termination_causes']) for row in report['lanes'])
    checks = dict(paired=all(comparison['initial_exact'].values())
            and comparison['sensor_actuator_material_exact'] and comparison['rigid_parameters_exact'],
        exact_runtime_layer=runtime['final_layer_runtime_sha256'] == expected['final_layer_runtime_sha256'],
        diagnostic_exclusion_explicit=report['evaluation_overrides']['fallen_output_partition_probe']['eligible_for_checkpoint_mean_qualification'] is False,
        source_archived=all(digest(run / 'source' / n) == h for n, h in progress['source_sha256'].items()),
        standing=standing['pose_and_position_pass'] >= gate['standing_at_least'],
        motion=metrics['motion']['vx_mae_m_s'] <= gate['motion_MAE_at_most'],
        entry_recovery=metrics['entry']['recovered'] >= gate['entry_recovery_at_least'],
        entry_followup=metrics['entry']['followup_success'] >= gate['entry_followup_at_least'],
        fallen_recovery=metrics['fallen']['recovered'] >= gate['fallen_recovery_at_least'],
        fallen_followup=metrics['fallen']['followup_success'] >= gate['fallen_followup_at_least'],
        entry_speed=metrics['entry']['max_leg_speed_rad_s'] <= gate['entry_peak_at_most_rad_s'],
        fallen_speed=metrics['fallen']['max_leg_speed_rad_s'] <= gate['fallen_peak_at_most_rad_s'],
        impacts=impacts <= gate['impacts_at_most'], closure=report['max_gap_m'] <= .003,
        passive=report['passive_motor_peak_nm'] == 0.)
    results[mode] = dict(checks={k: bool(v) for k, v in checks.items()}, metrics=metrics,
        standing_pass=standing['pose_and_position_pass'], impacts=impacts, runtime=runtime)
out = dict(schema='fallen-output-partition-native-review-v1', results=results,
    frozen_inputs=len(plan['source_sha256']), frozen_sources_exact=True,
    decision='Diagnostic completed; evaluate individual nonregression and target behavior before any next proposal. No saved-mean qualification or PPO admission.',
    native_run_clock=1509, best_clock=1501, new_PPO=0, actual_experiment_consumed=2638,
    repair_optimizer_updates=258, qualification=None, hardware_started=False, created_unix=time.time())
write_json(R / 'fallen_partition_observed_decision001.json', out)
write_json(R / 'diagnosis040_closed_source_seal001.json', dict(schema='closed-native-queue-source-seal-v1',
    queue='040', all_frozen_sources_verified=True, source_files=len(plan['source_sha256']),
    decision_sha256=digest(R / 'fallen_partition_observed_decision001.json'), new_PPO=0,
    qualification=None, created_unix=time.time()))
p = json.loads((R / 'progress.json').read_text())
p.update(status='fallen_output_partition_diagnosed', active_job=None, active_supervisor=None,
    active_evaluation=None, active_native_queue=None, active_diagnosis_queue=None,
    latest_research_decision=out['decision'], new_PPO_admitted=0, updated_unix=time.time())
write_json(R / 'progress.json', p)
print(json.dumps({mode: dict(standing=d['standing_pass'], entry=d['metrics']['entry'], fallen=d['metrics']['fallen'],
    failed=[k for k, v in d['checks'].items() if not v]) for mode, d in results.items()}, indent=2))
