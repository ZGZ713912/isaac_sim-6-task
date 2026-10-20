"""Conditional traction sanity check; never changes a native acceptance result."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from wheeled_tasks.chassis.flat_base_continuation import digest, encode, read_plan

target = HERE / 'static_grip_candidate002/stopping_feasibility001.json'
if target.exists():
    raise SystemExit('Preserve the existing conditional stopping audit')
plan_path = ROOT / 'contracts/flat_base_finetune_ground_0p2_0p5_v1.json'
plan = read_plan(ROOT, str(plan_path.relative_to(ROOT)))
material = plan['contact_material']
assert material['combine_mode'] == 'average'
assert material['ground_static_friction_range'] == material['ground_dynamic_friction_range'] == [.2, .5]
wheel = material['wheel_dynamic_friction']
ground = material['ground_dynamic_friction_range']
effective = [(value + wheel) / 2 for value in ground]
worker_path = ROOT / plan['source_worker_file']
worker = json.loads(worker_path.read_text())
gravity = 9.81
rows = {}
for name in ('start_stop_1', 'start_stop_2', 'start_stop_3'):
    case = next(row for row in worker['evaluation']['cases'] if row['name'] == name)
    speed = abs(case['command'][0])
    skill = case['skill']
    seconds = max(skill.get('stop_settle_seconds', 1.), skill.get('velocity_transition_seconds', 0.))
    threshold = case['settled_stop_speed_max']
    rows[name] = dict(request_speed_m_s=speed, settle_seconds=seconds,
        settled_speed_threshold_m_s=threshold,
        ideal_acceleration_limit_m_s2=[value * gravity for value in effective],
        ideal_time_to_threshold_s=[max(0., speed - threshold) / (value * gravity) for value in effective],
        ideal_residual_speed_after_settle_m_s=[max(0., speed - value * gravity * seconds) for value in effective],
        ideal_required_effective_mu=max(0., speed - threshold) / (gravity * seconds))
target.write_bytes(encode(dict(schema='flat-ground025-conditional-stopping-feasibility-v1',
    recorded_at=datetime.now(timezone.utc).isoformat(), plan_sha256=digest(plan_path),
    source_worker_sha256=digest(worker_path), wheel_dynamic_mu=wheel,
    ground_dynamic_mu_range=ground, effective_dynamic_mu_range=effective,
    assumed_gravity_m_s2=gravity,
    assumptions=['ideal Coulomb longitudinal traction', 'total wheel normal force equals mass*g',
        'no useful additional braking force', 'initial actual speed equals requested cruise speed',
        'no vertical transient or body contact'],
    uncertainty='PhysX solver, vertical load transients, rolling resistance and actual approach speed are not measured by this calculation.',
    cases=rows, native_trajectory_verified=False, acceptance_changed=False,
    new_optimizer_updates=0, qualified=False,
    scope='A conditional plausibility check to interpret completed native stop traces; not proof of impossibility or an excuse for a failed test.',
    reader_sha256=digest(Path(__file__)))))
print(json.dumps(rows))
