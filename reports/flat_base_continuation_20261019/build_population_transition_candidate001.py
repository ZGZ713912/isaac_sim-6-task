"""Prepare a separately reviewable trainer candidate; do not change production."""
from pathlib import Path
import ast
import hashlib
import json

HERE = Path(__file__).resolve().parent
out = HERE / 'population_transition_candidate001'
out.mkdir(exist_ok=False)
base = HERE / 'height_metrics_candidate_v16/train_flat_base.py'
text = base.read_text()
helper = (HERE / 'population_transition_functions001.py').read_text().split('\n', 1)[1]
text = text.replace('\ndef validate_large_block_evidence(', helper + '\n\ndef validate_large_block_evidence(', 1)
def replace(old, new):
    global text
    if text.count(old) != 1:
        raise ValueError('Candidate edit is not unique: ' + old)
    text = text.replace(old, new, 1)
replace('type=int, choices=plan["allowed_num_envs"], default=plan["default_num_envs"]',
        'type=int, default=None')
replace('    args = parser.parse_args()\n    plan = read_plan(ROOT, args.plan)',
        '    parser.add_argument("--capacity-probe", action="store_true", help="Native rollout/backward capacity only, with Adam writes disabled")\n'
        '    args = parser.parse_args()\n    plan = read_plan(ROOT, args.plan)\n'
        '    args.num_envs = plan["default_num_envs"] if args.num_envs is None else args.num_envs\n'
        '    if args.num_envs not in plan["allowed_num_envs"]:\n'
        '        parser.error("Population is absent from the selected plan")\n'
        '    if args.capacity_probe and (args.updates != 1 or not args.resume or args.device != "cuda:0"):\n'
        '        parser.error("Capacity mode requires one isolated CUDA probe with an exact parent")')
replace('report["prior_discontinued_optimizer_updates"] = plan["prior_discontinued_optimizer_updates"]',
        'report["training_transitions_total"] = 0\n'
        '    report["training_transition_count_scope"] = "successful inherited rollout frames; discarded optimizer consumption remains separately charged"\n'
        '    report["prior_discontinued_optimizer_updates"] = plan["prior_discontinued_optimizer_updates"]')
replace('("flat_base_source_onnx_sha256", "asset_manifest_sha256", "num_envs")',
        '("flat_base_source_onnx_sha256", "asset_manifest_sha256")')
replace('            migration = info["plan_sha256"] != report["plan_sha256"]',
        '            migration = info["plan_sha256"] != report["plan_sha256"]\n'
        '            population_changed = info["num_envs"] != args.num_envs\n'
        '            if population_changed and not migration:\n'
        '                raise ValueError("A different population requires an authenticated population-only plan migration")')
replace('"physics_dt", "control_file", "control_sha256", "moving_velocity_cost_scale"}',
        '"physics_dt", "control_file", "control_sha256", "moving_velocity_cost_scale",\n'
        '                           "allowed_num_envs", "default_num_envs"}')
replace('                report["changed_plan_fields"] = changed',
        '                report["changed_plan_fields"] = changed\n'
        '                if population_changed:\n'
        '                    report["population_transition"] = validate_population_transition(\n'
        '                        info, old_plan, plan, args.num_envs, revision, report["source_sha256"])\n'
        '                elif set(changed) & {"allowed_num_envs", "default_num_envs"}:\n'
        '                    raise ValueError("Population plan fields require an actual population transition")')
replace('            if info["control_math_sha256"] != report["control_math_sha256"]:',
        '            report["training_transitions_total"] = restored_training_frames(\n'
        '                info, old_plan if migration else plan)\n'
        '            if info["control_math_sha256"] != report["control_math_sha256"]:')
replace('            if migration:\n                report["critic_warmup_remaining"] += plan.get("additional_critic_warmup_updates", 0)',
        '            if migration and not population_changed:\n                report["critic_warmup_remaining"] += plan.get("additional_critic_warmup_updates", 0)')
old='env.training_transitions = report["successful_updates_total"] * args.num_envs * plan["num_steps_per_env"]'
if text.count(old) != 2:
    raise ValueError('Both frame clock assignments must be reviewed')
text=text.replace(old, 'env.training_transitions = report["training_transitions_total"]')
replace('            report["successful_updates_total"] += 1',
        '            report["successful_updates_total"] += 1\n'
        '            report["training_transitions_total"] += args.num_envs * plan["num_steps_per_env"]')
replace('writer.add_scalar("FlatTransfer/actor_updates_total", report["actor_updates_total"], report["successful_updates_total"])',
        'writer.add_scalar("FlatTransfer/training_transitions_total", report["training_transitions_total"], report["successful_updates_total"])\n'
        '                writer.add_scalar("FlatTransfer/actor_updates_total", report["actor_updates_total"], report["successful_updates_total"])')
old=text[text.index('        runner.alg.update = update\n'):text.index('    except Exception:\n')]
new='''        if args.capacity_probe:
            # Call the original learner, with no reporting/update wrapper installed.
            save_checkpoint(args.output / "initial_state.pt")
            probe_population_capacity(runner, env, plan, report, torch)
            report.update(status="capacity_complete", elapsed_s=time.monotonic() - start,
                requested_additional_updates=0, new_optimizer_updates=0)
            print("FLAT_CAPACITY_COMPLETE", json.dumps(report["capacity_probe"]), flush=True)
        else:
'''+''.join('    '+line+'\n' for line in old.splitlines())
replace(old,new)
replace('return 0 if report["status"] == "block_complete" else 1',
        'return 0 if report["status"] in ("block_complete", "capacity_complete") else 1')
ast.parse(text)
(out / 'train_flat_base.py').write_text(text)
(out / 'preparation.json').write_text(json.dumps(dict(
    schema='flat-population-trainer-candidate-preparation-v1', adopted=False,
    input_sha256=hashlib.sha256(base.read_bytes()).hexdigest(),
    entry_sha256=hashlib.sha256(text.encode()).hexdigest(),
    learning_changes='Population only when explicitly migrated; objective, Adam, weights, std, epochs, rates and clocks unchanged; cumulative rollout exposure retained',
    native_steps=0, optimizer_updates=0),indent=2)+'\n')
print(out)
