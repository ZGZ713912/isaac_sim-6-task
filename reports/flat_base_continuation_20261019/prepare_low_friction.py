"""Prepare separate material plans without starting a learner or altering old evidence."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from wheeled_tasks.chassis.flat_base_continuation import digest, encode, read_plan, write_bundle
from wheeled_tasks.chassis.control_contract import load_chassis_control
from wheeled_tasks.chassis.preflight import design_preflight

HERE = Path(__file__).resolve().parent / "low_friction_candidate001"
if HERE.exists():
    raise SystemExit("Preserve existing low-friction candidate files")
HERE.mkdir()
old_file = "reports/flat_base_continuation_20261019/wheel_rate_candidate001/plan.json"
old = read_plan(ROOT, old_file)
low = deepcopy(old)
low["schema"] = "flat-base-finetune-plan-v17"
low["training_scope"] = "separate_wheel_0p1_ground_0p2_to_0p3_low_grip_candidate"
low["contact_material"] = {
    "enabled": True, "combine_mode": "average",
    "ground_static_friction_range": [.2, .3], "ground_dynamic_friction_range": [.2, .3],
    "wheel_static_friction": .1, "wheel_dynamic_friction": .1,
    "body_static_friction": .25, "body_dynamic_friction": .25}
randomized = deepcopy(low)
randomized["training_scope"] = "after_low_grip_validation_ground_0p1_0p2_to_0p3_0p3_to_0p4_randomization"
randomized["contact_material"].update(
    ground_static_friction_range=[.1, .4], ground_dynamic_friction_range=[.1, .4],
    ground_sampling_bands=[{"range": [.1, .1], "fraction": 1 / 3},
        {"range": [.2, .3], "fraction": 1 / 3}, {"range": [.3, .4], "fraction": 1 / 3}])
receipts = {}
for name, plan in (("low", low), ("randomized", randomized)):
    path = HERE / (name + "_plan.json")
    path.write_bytes(encode(plan))
    relative = str(path.relative_to(ROOT))
    assert read_plan(ROOT, relative) == plan
    config, seed = write_bundle(ROOT, HERE / (name + "_train_config"), 4096, plan_file=relative)
    control = load_chassis_control(ROOT / config["control_math_source"])
    receipt = design_preflight(config, control, ROOT, stage="speed", role="train", num_envs=4096, seed=seed)
    eval_config, eval_seed = write_bundle(ROOT, HERE / (name + "_eval_config"), 32,
        evaluation=True, plan_file=relative)
    eval_receipt = design_preflight(eval_config, control, ROOT, stage="speed", role="quick", num_envs=32, seed=eval_seed)
    changed = sorted(k for k in set(old) | set(plan) if old.get(k) != plan.get(k))
    assert changed == ["contact_material", "schema", "training_scope"]
    assert plan["policy_dt"] == .02 and plan["physics_dt"] == .001
    receipts[name] = dict(plan_file=relative, plan_sha256=digest(path),
        changed_plan_fields=changed, train_preflight=receipt, evaluation_preflight=eval_receipt)
parent = ROOT / "runs/flat_base_20261019/pilot028_wheel_rate1328/model_001328.pt"
result = dict(schema="flat-low-friction-material-preparation-v1", passed=True,
    recorded_at=datetime.now(timezone.utc).isoformat(), plans=receipts,
    source_plan_file=old_file, source_plan_sha256=digest(ROOT / old_file),
    parent_checkpoint=str(parent.relative_to(ROOT)), parent_checkpoint_sha256=digest(parent),
    sampling="startup-per-environment; randomized stage is a balanced three-band mixture",
    second_stage_requires="first-stage low-friction fixed behavior evaluation; no automatic long training",
    new_optimizer_updates=0, new_native_steps=0, hardware_motion_started=False,
    behavior_qualified=False)
(HERE / "preparation001.json").write_bytes(encode(result))
print(json.dumps(dict(passed=True, plans=list(receipts), output=str(HERE.relative_to(ROOT)))))
