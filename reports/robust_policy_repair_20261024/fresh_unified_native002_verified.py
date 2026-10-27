"""Verify native runtime receipt and freeze the next bounded fresh plan."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
BASE = Path('/home/noir/Documents/workspace/isaac_wheeled_rl_train')
REL = Path('reports/robust_policy_repair_20261024')
sys.path.insert(0, str(ROOT/'scripts'))
import train_fresh_unified as entry
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    path = BASE/REL/'fresh_unified_native002/receipt.json'
    native = json.loads(path.read_text())
    bank = BASE/REL/'pose_bank_height_matched073.json'
    guard = entry.require_native_runtime_receipt(path, entry.sources(), bank)
    accumulator = EventAccumulator(str(path.parent), size_guidance={'scalars':0}).Reload()
    for name in entry.NATIVE_REQUIRED_CHECKS:
        values = accumulator.Scalars('NativeRuntimeCheck/'+name)
        if len(values) != 1 or values[0].value != 1. or values[0].step != 0:
            raise ValueError('Native actual-check RSL readback differs: '+name)
    output = ROOT/REL/'fresh_unified_native002_verified.json'
    if output.exists():
        raise ValueError('Preserve the actual native verification')
    result = dict(schema='fresh-unified-native-wiring-verified-v1',
        observed_utc=datetime.now(timezone.utc).isoformat(), native_receipt=guard,
        checks=native['checks'], all_checks_pass=True, actual_RSL_checks_read_back=len(entry.NATIVE_REQUIRED_CHECKS),
        source_commit=native['source_commit'], source_count=len(native['source_sha256']),
        native_policy_steps=144, physical_steps=720, new_PPO=0, checkpoint_reads=[],
        original_process_exited=True, real_lane01_autoresets=native['lane01_autoreset_counts'],
        logical_fixture_actual_completion_seconds=native['actual_completion_time_s'],
        fixture_limit='Physically upright positive control of actual native stable-hold reward wiring; no fallen self-righting skill is claimed',
        observed_physical_extrema=native['observed_physical_extrema'],
        next_fresh_plan=dict(entry='scripts/train_fresh_unified.py',
            output=str(BASE/'runs/fresh_unified_20261027/pilot015_requested_mixed1536'),
            seed=271001, training_case_seed=271701, domain_admission_seed=271501,
            num_envs=256, rollout_steps=128, updates=1536, foundation_updates=192, full_tasks_at=768,
            domain_warmup_updates=192, domain_ramp_updates=128, requested_strength=1.,
            actor_critic_Adam='Fresh actor/critic and empty Adam; every old checkpoint read forbidden',
            source_count=272, behavior_evaluation='Separate cold same-PT full13 suite and holdout domains after exact learner exit',
            actual_training_updates_previously_consumed=4226, prior_failed014_reused=False, formal_long_training_admitted=False, qualified=False, hardware_ready=False),
        qualified=False, hardware_ready=False)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    shutil.copyfile(output, BASE/REL/output.name)
    shutil.copyfile(Path(__file__), BASE/REL/Path(__file__).name)
    print(json.dumps(dict(actual_RSL_checks=len(entry.NATIVE_REQUIRED_CHECKS), native_policy_steps=144, physical_steps=720, new_PPO=0,
        next_bounded_fresh_updates=1536, qualified=False)))


if __name__ == '__main__':
    main()
