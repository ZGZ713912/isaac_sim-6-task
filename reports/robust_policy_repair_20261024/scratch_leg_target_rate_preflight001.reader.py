from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
a = BASE / 'scratch_leg_target_rate_zero_preflight001'
b = BASE / 'scratch_leg_target_rate_cost020_preflight001'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value, output):
    value = json.loads(json.dumps(value).replace(str(output), '<OUTPUT>'))
    if 'design_preflight' in value:
        value['design_preflight']['sha256'] = '<CHECKED_DESIGN_SHA>'
    return value


ta, tb = [read(folder / 'task_contract.json') for folder in (a, b)]
sa, sb = [read(folder / 'source_hashes.json') for folder in (a, b)]
configs = [read(folder / 'config/contract.json') for folder in (a, b)]
for folder, config in zip((a, b), configs):
    assert sha(folder / 'config/design.json') == config['design_preflight']['sha256']
ca, cb = [dict(task['standing_leg_target_rate_contract']) for task in (ta, tb)]
weights = [ca.pop('weight_density_per_second'), cb.pop('weight_density_per_second')]
old = read(ROOT / 'runs/scratch_standing_20261026/pilot007_contact_cost_zero/source_hashes.json')
checks = dict(source_snapshots_identical=sa == sb,
    all263_actual_sources_match=len(sa) == 263 and all(sha(ROOT / n) == h and sha(a/'source_snapshot'/n) == h
        and sha(b/'source_snapshot'/n) == h for n, h in sa.items()),
    bank_identical=ta['bank_sha256'] == tb['bank_sha256'], settings_identical=ta['settings'] == tb['settings'],
    mechanism_identical=canonical(configs[0], a) == canonical(configs[1], b),
    design_identical=canonical(read(a/'config/design.json'), a) == canonical(read(b/'config/design.json'), b),
    objective_only_weight_differs=ca == cb and weights == [0., .2],
    original_contact_cost_zero=ta['standing_contact_continuity_contract'] == tb['standing_contact_continuity_contract']
        and ta['standing_contact_continuity_contract']['reward_cost_per_break'] == 0.,
    fixed_evaluation_and_mechanical_sources_preserved=all(sa[n] == h for n, h in old.items()
        if n != 'scripts/train_scratch_standing.py'),
    only_one_new_training_module=set(sa) - set(old) == {'src/wheeled_tasks/chassis/standing_leg_target_rate.py'},
    no_model_or_optimizer_created=not any(folder.glob('model*.pt') for folder in ())
        and all(not list(folder.glob('model*.pt')) and not (folder/'progress.json').exists() for folder in (a,b)))
rejections = []
with tempfile.TemporaryDirectory() as temporary:
    output = Path(temporary) / 'rejected'
    for flags in (['--leg-target-rate-cost', '.2'], ['--precise-objective', '--leg-target-rate-cost', '-.1'],
                  ['--precise-objective', '--leg-target-rate-cost', 'nan'],
                  ['--precise-objective', '--leg-target-rate-cost', '17']):
        process = subprocess.run([sys.executable, str(ROOT/'scripts/train_scratch_standing.py'),
            '--output', str(output), *flags], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        rejections.append(dict(flags=flags, returncode=process.returncode,
            output_not_created=not output.exists(), stderr_tail=process.stderr[-400:]))
checks['all4_invalid_CLI_combinations_rejected'] = all(row['returncode'] == 2 and row['output_not_created'] for row in rejections)
assert all(checks.values()), checks
report = dict(schema='standing-leg-target-rate-preflight-v1', observed_utc=datetime.now(timezone.utc).isoformat(),
    checks=checks, all_checks_pass=True, zero_cost_preflight=str(a), candidate_preflight=str(b),
    new_training_weight_density_per_second=.2, target_delta_scale_rad=.001,
    unchanged_train_seed=261001, requested_fresh_updates=512,
    reader_sha256=sha(__file__), source_hashes=sa, CLI_rejections=rejections,
    new_PPO=0, checkpoint_reads=0, native_App_started=False, actor_constructed=False,
    behavior_qualification=False, hardware_ready=False,
    scope='Only an optional training cost is connected; no smoothing/filter/clamp is applied to outputs. '
          'The new weight is a pilot hypothesis and does not establish improved behavior.')
output = BASE/'scratch_leg_target_rate_preflight001.json'
assert not output.exists()
output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
print(json.dumps(dict(checks=checks, all_checks_pass=True)))
