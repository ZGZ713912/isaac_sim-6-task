"""Read ended native standing experiments without constructing a model."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


runs = [ROOT / 'runs/scratch_standing_20261026' / name for name in (
    'pilot005_static_load', 'pilot006_contact_events', 'pilot007_contact_cost_zero')]
reports = {}
for run in runs:
    progress = read(run / 'progress.json')
    result = read(run / 'evaluation/result.json')
    assert progress['status'] == 'completed_unqualified' and progress['checkpoint_reads'] == []
    assert result['learned_policy'] and not result['fixed_target_probe']
    assert sha(run / 'evaluation/native_trace.npz') == result['trace_sha256']
    groups = []
    for height in (.23, .305, .42):
        rows = [row for row in result['cases'] if row['height_m'] == height]
        assert len(rows) == 7
        groups.append(dict(height_m=height, cases=7,
            standing_pass=sum(row['standing_pass'] for row in rows),
            full_horizon=sum(row['full_horizon'] for row in rows),
            missing_physical_ms=sum(row['missing_physical_ms'] for row in rows),
            minimum_support_fraction=min(row['dual_support_fraction'] for row in rows),
            mean_vibration_mrad=sum(row['leg_vibration_3_95hz_rms_mrad'] for row in rows)/7,
            maximum_leg_speed_rad_s=max(row['leg_speed_peak_rad_s'] for row in rows),
            mean_height_mae_m=sum(row['height_mae_m'] for row in rows)/7,
            mean_forward_speed_rms_m_s=sum(row['forward_speed_rms_m_s'] for row in rows)/7,
            maximum_tilt_deg=max(row['tilt_peak_deg'] for row in rows)))
    reports[run.name] = dict(completed_updates=progress['completed_updates'], groups=groups,
        result=str(run / 'evaluation/result.json'), result_sha256=sha(run / 'evaluation/result.json'),
        final_checkpoint_sha256=sha(run / 'model_final.pt'),
        full_evaluation_standing_pass=result['standing_pass'],
        trained_heights_m=[.305], hardware_ready=False)

a, b = runs[1:]
args_a, args_b = [dict(read(run / 'progress.json')['args']) for run in (a, b)]
args_a.pop('output'); args_b.pop('output')
argument_diff = {key: [args_a[key], args_b[key]] for key in args_a if args_a[key] != args_b[key]}
initial_a, initial_b = [read(run / 'initialization.json') for run in (a, b)]
result_a, result_b = [read(run / 'evaluation/result.json') for run in (a, b)]
checks = dict(
    paired_source_manifests_identical=read(a / 'source_hashes.json') == read(b / 'source_hashes.json'),
    paired_agent_configs_identical=read(a / 'agent_config.json') == read(b / 'agent_config.json'),
    paired_initial_actor_tensor_hash_identical=initial_a['actor_tensor_sha256'] == initial_b['actor_tensor_sha256'],
    paired_initial_critic_tensor_hash_identical=initial_a['critic_tensor_sha256'] == initial_b['critic_tensor_sha256'],
    only_training_argument_difference_is_event_cost=argument_diff == {'contact_break_cost': [.1, 0.]},
    paired_fixed_initial_root_positions_identical=result_a['initial_root_positions'] == result_b['initial_root_positions'],
    paired_case_labels_identical=[(r['height_m'], r['pitch_deg']) for r in result_a['cases']] ==
                                [(r['height_m'], r['pitch_deg']) for r in result_b['cases']],
    paired_original_protocol_identical=result_a['protocol'] == result_b['protocol'],
    all_three_strict_fixed_evaluations_unqualified=all(r['full_evaluation_standing_pass'] == 0 for r in reports.values()))
assert all(checks.values()), checks
old, cost, new = [reports[run.name]['groups'][1] for run in runs]
report = dict(schema='scratch-fixed-evaluation-comparison-v1', reader_sha256=sha(__file__),
    runs=reports, paired006_007_checks=checks, all_checks_pass=True,
    new007_to005_missing_ms_ratio=new['missing_physical_ms']/old['missing_physical_ms'],
    new007_to006_missing_ms_ratio=new['missing_physical_ms']/cost['missing_physical_ms'],
    new007_to005_vibration_ratio=new['mean_vibration_mrad']/old['mean_vibration_mrad'],
    new007_to006_vibration_ratio=new['mean_vibration_mrad']/cost['mean_vibration_mrad'],
    selected_for_hardware=False, new_PPO_by_reader=0,
    decision='Reject007 as a stable-standing solution; retain all original gates. '
             '007 improves over006 but is worse than005 in total support loss. '
             'Next isolate subsequent leg-target changes on frozen007, after verifying its read-only baseline.',
    scope='006/007 are one controlled same-seed fresh-training pair, not multi-seed proof. '
          '005 has an earlier reset cache and a256-update budget; it is an imperfect reference. '
          'All runs trained only the middle height. Low/high failures remain explicitly unqualified.')
output = BASE / 'scratch_fixed005_006_007_comparison001.json'
assert not output.exists()
output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
print(json.dumps(dict(checks=checks, missing_ms=[old['missing_physical_ms'],cost['missing_physical_ms'],new['missing_physical_ms']],
    new_to005_missing_ratio=report['new007_to005_missing_ms_ratio'],
    new_to006_missing_ratio=report['new007_to006_missing_ms_ratio'],
    new_to005_vibration_ratio=report['new007_to005_vibration_ratio'])))
