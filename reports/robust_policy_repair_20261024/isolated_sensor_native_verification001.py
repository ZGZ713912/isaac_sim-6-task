"""Retain the native negative control, repaired clock proof and closed old queue."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

WORK=Path(__file__).resolve().parents[2]
BASE=Path('/home/noir/Documents/workspace/isaac_wheeled_rl_train')
REPORT=BASE/'reports/robust_policy_repair_20261024'
OUT=REPORT/'isolated_sensor_native_verification001.json'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


assert not OUT.exists()
paths=[REPORT/name/'receipt.json' for name in ['sensor_reset_native_legacy002','sensor_reset_native_isolated001']]
legacy,fixed=map(read,paths)
assert all(row['status']=='completed_unqualified' and row['all_checks_pass'] for row in [legacy,fixed])
assert all(not Path(f"/proc/{row['pid']}").exists() for row in [legacy,fixed])
assert legacy['seed']==fixed['seed']==271504
assert legacy['actual_domains']==fixed['actual_domains']
assert legacy['untouched_before']==fixed['untouched_before']
assert legacy['physics_runtime']==fixed['physics_runtime']
assert len(legacy['actual_domains']['samples'])==17
assert fixed['maximum_untouched_sensor_delta']==0.
failed=read(REPORT/'sensor_reset_native_legacy001/receipt.json')
assert failed['status']=='failed' and failed['physics_steps']==0 and 'NameError' in failed['error']
queue_path=REPORT/'requested_v2_evaluation_queue003'
queue=read(queue_path/'receipt.json')
assert queue['status']=='completed_unqualified' and queue['active_child'] is None
assert not Path(f"/proc/{queue['pid']}").exists()
assert queue['checkpoint_sha256']=='f3de95f8f1df97ea90a64fdfe1522b210b8bfdcf5438d95acb18df15c774bfa0'
assert sha(queue['checkpoint'])==queue['checkpoint_sha256']
for profile in queue['profiles']:
    assert sha(Path(profile['run'])/'receipt.json')==profile['receipt_sha256']
    assert sha(profile['review'])==profile['review_sha256']
    checks=profile['paired_nonintervened_draw_checks']
    assert checks is None or all(checks.values())
new=read(REPORT/'isolated_sensor_first_verification001.json')
assert new['all_checks_pass'] and new['actual_initial_domain_arrays_compared']==19
result=dict(schema='native-sensor-reset-isolation-controlled-proof-v1',observed_utc=datetime.now(timezone.utc).isoformat(),
    native_runs=[dict(path=str(path.parent),receipt_sha256=sha(path),pid=row['pid'],status=row['status'],checks=row['checks'])
        for path,row in zip(paths,[legacy,fixed],strict=True)],
    same_preintervention_observations=True,same_native_physics=True,
    same17_actual_sensor_actuator_rigid_arrays=True,same_actual_ground_material_samples=True,
    native_physics_steps_per_run=40,
    legacy=dict(changed_untouched_lanes=sum(legacy['untouched_changed']),untouched_lanes=len(legacy['untouched_changed']),
        sensor_cursor=[legacy['sensor_cursor_before'],legacy['sensor_cursor_after']],maximum_observation_delta=legacy['maximum_untouched_sensor_delta']),
    repaired=dict(changed_untouched_lanes=sum(fixed['untouched_changed']),sensor_cursor=[fixed['sensor_cursor_before'],fixed['sensor_cursor_after']],
        maximum_observation_delta=fixed['maximum_untouched_sensor_delta']),
    sensor_delta_units='Raw18D observation coordinates have mixed documented scales; do not label this joint motion or real robot vibration.',
    first_native_probe_error=dict(receipt=str(REPORT/'sensor_reset_native_legacy001/receipt.json'),
        receipt_sha256=sha(REPORT/'sensor_reset_native_legacy001/receipt.json'),status='failed',new_PPO=0,
        reason='Standalone caller had not assigned the training helper module torch binding; fixed before legacy002. All negative evidence retained.'),
    production_CPU_checks=dict(tests=136,passed=136,receipt=str(REPORT/'selective_sensor_production_CPU002.xml'),
        receipt_sha256=sha(REPORT/'selective_sensor_production_CPU002.xml'),
        source_path='Explicit PYTHONPATH to this worktree src; default editable installation points to the original checkout.',
        earlier_wrong_import_receipt=str(REPORT/'selective_sensor_production_CPU001.xml')),
    reward_only_011_closed_queue=dict(path=str(queue_path),receipt_sha256=sha(queue_path/'receipt.json'),
        profiles=[dict(profile=row['profile'],middle_height=row['middle_height'],paired_checks=row['paired_nonintervened_draw_checks']) for row in queue['profiles']],
        conclusion='All5 profiles remain0/7 at trained height. The long-tail reward-only experiment did not restore balance.'),
    actual_new_training=dict(run=new['run'],pid=new['pid'],start_ticks=new['start_ticks'],
        verification=str(REPORT/'isolated_sensor_first_verification001.json'),checks=len(new['checks']),
        actual_initial_domain_arrays_compared=19,RSL_config_scalars_verified=55),
    source_bug_fixed=True,fresh_learning_adopted=True,training_behavior_improvement_proven=False,
    new_PPO_by_this_reader=0,long_training_started=False,ONNX_exported=False,hardware_ready=False,
    producer_sha256=sha(__file__))
write(OUT,result);(REPORT/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
ledger_path=REPORT/'scratch_campaign001.json';raw=ledger_path.read_bytes();ledger=json.loads(raw)
assert ledger['active_pid']==new['pid'] and ledger['prior_closed_fresh_updates']==3585
ledger.update(latest_sensor_reset_native_proof=str(OUT),latest_controlled_comparison=str(queue_path/'receipt.json'),
    latest_observation='Actual native neighbor-reset clock leak repaired and parameter paired; fresh128-update pilot012 is learning full requested widths from zero, behavior unqualified.')
assert ledger_path.read_bytes()==raw
temporary=ledger_path.with_suffix('.isolatednative001.tmp');write(temporary,ledger);temporary.replace(ledger_path)
print(json.dumps(dict(native_checks=fixed['checks'],legacy_changed_lanes=sum(legacy['untouched_changed']),
    repaired_changed_lanes=sum(fixed['untouched_changed']),actual_initial_draws_paired=True)))
