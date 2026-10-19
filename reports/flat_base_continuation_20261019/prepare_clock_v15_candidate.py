"""Prepare isolated clock-alignment code while the v14 native queue stays frozen."""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DEST = HERE / 'clock_v15_source_candidate'
if DEST.exists():
    raise SystemExit('Preserve the existing isolated clock candidate')
DEST.mkdir()
parent = json.loads((ROOT / 'runs/flat_base_20261019/pilot019/progress.json').read_text())
for n in parent['source_sha256']:
    p = DEST / n
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes((ROOT / n).read_bytes())
for n in ('src/wheeled_tasks/__init__.py','src/wheeled_algo/__init__.py',
          'src/wheeled_tasks/v40/__init__.py','src/wheeled_tasks/v40/core.py','src/wheeled_tasks/v40/contract.py'):
    p = ROOT / n
    if p.exists():
        target = DEST / n
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(p.read_bytes())
helper = DEST / 'src/wheeled_tasks/chassis/flat_clock_transition.py'
helper.write_bytes((HERE / 'flat_clock_transition_candidate.py').read_bytes())


def edit(name, old, new):
    p = DEST / name
    text = p.read_text()
    if text.count(old) != 1:
        raise ValueError((name, old, text.count(old)))
    p.write_text(text.replace(old,new))


path = 'src/wheeled_tasks/chassis/flat_base_continuation.py'
edit(path, '10, 11, 12, 13, 14)):', '10, 11, 12, 13, 14, 15)):')
edit(path, '''    if (plan["policy_dt"], plan["physics_dt"], plan["leg_target_winding"], plan["flat_context"]) != (
            .02, .005, "paired_branch107_v1", [1, 0, 0, 0, 0, 0, 0]):
        raise ValueError("Flat transfer changed its declared 50/200 Hz paired-107 interface")''',
'''    if (plan["policy_dt"], plan["leg_target_winding"], plan["flat_context"]) != (
            .02, "paired_branch107_v1", [1, 0, 0, 0, 0, 0, 0]):
        raise ValueError("Flat transfer changed its declared50Hz paired-107 interface")
    from .flat_clock_transition import validate_flat_clock
    validate_flat_clock(root, plan)''')
edit(path, '''    row["runtime"] = dict(physics_dt=.005, policy_dt=.02, pc_control_dt=.005, decimation=4,''',
'''    row["runtime"] = dict(physics_dt=plan["physics_dt"], policy_dt=plan["policy_dt"],
        pc_control_dt=control["timing"]["pc_control_dt"], decimation=control["timing"]["decimation"],''')
path = 'src/wheeled_tasks/chassis/flat_command_reference.py'
edit(path, 'or env.policy_dt != .02 or env.cfg["physics_dt"] != .005',
    'or env.policy_dt != .02 or env.cfg["physics_dt"] not in (.005,.001)')
edit(path, 'Training RMCS reference requires the declared mixed 50/200 Hz flat pilot',
    'Training RMCS reference requires the declared mixed50Hz flat pilot at200/1000Hz physics')
edit(path, 'policy_hz=50, physics_hz=200,',
    'policy_hz=50, physics_hz=round(1 / env.cfg["physics_dt"]),')
path = 'scripts/train_flat_base.py'
edit(path, '''for key in ("flat_base_source_onnx_sha256", "asset_manifest_sha256", "control_math_sha256", "num_envs"):''',
'''for key in ("flat_base_source_onnx_sha256", "asset_manifest_sha256", "num_envs"):''')
edit(path, '''"learning_rate", "training_command_reference", "feedback_training_scope", "command_branch_restore"}''',
'''"learning_rate", "training_command_reference", "feedback_training_scope", "command_branch_restore",
                           "physics_dt", "control_file", "control_sha256"}''')
edit(path, '''            if migration and plan.get("command_branch_restore") and plan["command_branch_restore"] != old_plan.get("command_branch_restore"):''',
'''            if info["control_math_sha256"] != report["control_math_sha256"]:
                if not migration:
                    raise ValueError("A changed control hash requires the exact clock plan migration")
                from wheeled_tasks.chassis.flat_clock_transition import validate_flat_clock_transition
                report["feedback_clock_transition"] = validate_flat_clock_transition(
                    ROOT, old_plan, plan, info["control_math_sha256"])
            elif migration and old_plan["physics_dt"] != plan["physics_dt"]:
                raise ValueError("A physics-clock change must migrate the authenticated control source")
            if migration and plan.get("command_branch_restore") and plan["command_branch_restore"] != old_plan.get("command_branch_restore"):''')
for p in DEST.rglob('*.py'):
    ast.parse(p.read_text(), filename=str(p))
plan = deepcopy(json.loads((ROOT / 'contracts/flat_base_finetune_v14.json').read_text()))
plan.update(schema='flat-base-finetune-plan-v15',physics_dt=.001,
    control_file='contracts/v6_wheel_response_control_v1.json',
    control_sha256='cc30b2b0572999224c8feb79ffa7afb3df1b8a28800bca4a153663b9ea8d4bf8',
    training_scope='candidate_parent1024_exact_fullstate50Hz_policy1000Hz_physics_and_feedback_PD_same_rewards_sampling_gamma_GAE_and_rates')
candidate = HERE / 'clock_candidate_v15.json'
candidate.write_text(json.dumps(plan,indent=2) + '\n')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
changed = sorted(str(p.relative_to(DEST)) for p in DEST.rglob('*.py')
    if not (ROOT / p.relative_to(DEST)).exists() or p.read_bytes() != (ROOT / p.relative_to(DEST)).read_bytes())
receipt = dict(schema='flat-clock-v15-isolated-candidate-v1',changed_files=changed,
    candidate_source_sha256={n:sha(DEST / n) for n in changed},candidate_plan_sha256=sha(candidate),
    source_adopted=False,native_environment_steps=0,optimizer_updates_performed=0,behavior_qualified=False)
(HERE / 'clock_candidate_v15_source001.json').write_text(json.dumps(receipt,indent=2) + '\n')
print(json.dumps(receipt,indent=2))
