"""Check the actual integrated preflight without constructing a native learner."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import torch

ROOT = Path(__file__).resolve().parents[2]
BASE = Path('/home/noir/Documents/workspace/isaac_wheeled_rl_train')
REPORT = BASE / 'reports/robust_policy_repair_20261024'
PREFLIGHT = BASE / 'runs/scratch_standing_20261026/admission013_preflight001'
OLD = BASE / 'runs/scratch_standing_20261026/pilot012_isolated_sensor_requested001'
OUT = REPORT / 'domain_admission_integrated_preflight001.json'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


assert not OUT.exists()
manifest = read(PREFLIGHT / 'source_hashes.json')
old_manifest = read(OLD / 'source_hashes.json')
old_task = read(OLD / 'task_contract.json')['settings']
task = read(PREFLIGHT / 'task_contract.json')['settings']
cfg = read(PREFLIGHT / 'config/contract.json')
old_cfg = read(OLD / 'config/contract.json')
suite = ET.parse(REPORT / 'domain_admission_integrated_CPU001.xml').getroot().find('testsuite')
queue = read(REPORT / 'requested_v2_evaluation_queue004/receipt.json')
sys.path.insert(0, str(ROOT / 'scripts'))
from domain_learning_admission_candidate import Admission

sample = read(OLD / 'randomization_study.initial_samples.json')
enabled = torch.tensor(sample['samples']['enabled'], dtype=torch.bool)
rng_before = torch.get_rng_state().clone()
schedule = Admission(enabled, warmup_updates=192, ramp_updates=128, seed=261501)
schedule_rows = [dict(completed_actor_clock=clock, nominal_lanes=int((schedule.mask(clock) & ~enabled).sum()),
    randomized_lanes=int((schedule.mask(clock) & enabled).sum())) for clock in (0, 192, 193, 256, 319, 320, 511)]
checks = dict(
    source_manifest_count=len(manifest) == 266,
    only_training_entry_changed=sorted(name for name in old_manifest if old_manifest[name] != manifest[name]) == ['scripts/train_scratch_standing.py'],
    only_admission_helper_added=set(manifest) - set(old_manifest) == {'scripts/domain_learning_admission_candidate.py'},
    actual_snapshot_matches_checkout=all(sha(ROOT / name) == value and sha(PREFLIGHT / 'source_snapshot' / name) == value for name, value in manifest.items()),
    actual_recipe_is_only_admission=sorted(key for key in set(task) | set(old_task) if task.get(key) != old_task.get(key)) == ['domain_learning_admission', 'version'],
    full_requested_domains_unchanged=task['randomization_study'] == old_task['randomization_study'],
    full_native_physics_unchanged=all(cfg[key] == old_cfg[key] for key in cfg if key != 'design_preflight'),
    P_D_spring_fixed=task['leg_kp_scale_range'] == task['leg_kd_scale_range'] == task['spring_strength_range'] == [1., 1.],
    independent_inertia_six_percent=cfg['dynamics_randomization']['inertia_scale'] == [.94, 1.06],
    actual_plan=task['domain_learning_admission']['warmup_updates'] == 192 and task['domain_learning_admission']['ramp_updates'] == 128 and task['domain_learning_admission']['total_updates'] == 512,
    complete_population_in_final192_updates=bool(schedule.mask(320).all()) and not bool(schedule.mask(319).all()),
    priority_draw_preserves_main_rng=torch.equal(rng_before, torch.get_rng_state()),
    CPU_tests=int(suite.attrib['tests']) == 62 and int(suite.attrib['skipped']) == 6 and int(suite.attrib['failures']) == int(suite.attrib['errors']) == 0,
    all12_frozen_queue_sources_unchanged=all(sha(ROOT / name) == value for name, value in queue['source_sha256'].items()),
    no_native_launch_from_preflight=not (PREFLIGHT / 'progress.json').exists() and not (PREFLIGHT / 'model_initial.pt').exists(),
)
assert all(checks.values()), checks
result = dict(schema='domain-admission-integrated-preflight-v1', observed_utc=datetime.now(timezone.utc).isoformat(),
    preflight=str(PREFLIGHT), checks=checks, all_checks_pass=True, source_count=266,
    source_sha256=manifest, plan=task['domain_learning_admission'],
    schedule_for_same_startup_enabled_sample=schedule_rows,
    existing_ground_friction_preserved_for_all_lanes=True,
    CPU_tests_passed=56, CPU_tests_explicitly_skipped=6,
    CPU_receipt_sha256=sha(REPORT / 'domain_admission_integrated_CPU001.xml'),
    native_PPO_by_this_reader=0, native_learner_constructed=False,
    native_behavior_verified=False, qualified=False, hardware_ready=False,
    producer_sha256=sha(__file__))
OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
(REPORT / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
print(json.dumps(dict(checks=len(checks), all_pass=True, source_count=266, schedule=schedule_rows)))
