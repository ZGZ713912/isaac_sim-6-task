"""Prepare an isolated source candidate; do not mutate any native frozen file."""
from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,json,shutil

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
DEST=HERE/'separated_v13_source_candidate'
if DEST.exists():raise SystemExit('Preserve the existing source candidate')
DEST.mkdir()
parent=json.loads((ROOT/'runs/flat_base_20261019/pilot016/progress.json').read_text())
for name in parent['source_sha256']:
 p=DEST/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/name).read_bytes())
for name in ['src/wheeled_algo/__init__.py','src/wheeled_tasks/__init__.py','scripts/export_flat_base.py']:
 p=ROOT/name
 if p.exists():
  target=DEST/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
module=DEST/'src/wheeled_algo/separated_feedback.py'
module.write_text((HERE/'separated_feedback_v3_candidate.py').read_text().replace('Unadopted identity fork','Identity fork'))

def edit(name,old,new):
 p=DEST/name;s=p.read_text();assert s.count(old)==1,(name,old,s.count(old));p.write_text(s.replace(old,new))

path='src/wheeled_algo/command_residual.py'
edit(path,'    if kind not in (None, KIND, FEEDBACK_KIND, TRACKING_KIND):','    from .separated_feedback import SeparatedFeedbackActor, KIND as SEPARATED_KIND\n    if kind not in (None, KIND, FEEDBACK_KIND, TRACKING_KIND, SEPARATED_KIND):')
edit(path,'    if kind in (KIND, FEEDBACK_KIND, TRACKING_KIND):','    if kind == SEPARATED_KIND:\n        mean = SeparatedFeedbackActor(GuardedFeedbackActor(CommandResidualActor(original), tracking_gate=True))\n    if kind in (KIND, FEEDBACK_KIND, TRACKING_KIND, SEPARATED_KIND):')
path='src/wheeled_tasks/chassis/flat_base_continuation.py'
edit(path,'10, 11, 12))','10, 11, 12, 13))')
edit(path,'"original-command-and-gated-feedback-residual-v1", "original-command-and-tracking-feedback-residual-v2"):',
 '''"original-command-and-gated-feedback-residual-v1", "original-command-and-tracking-feedback-residual-v2",
            "original-command-and-separated-tracking-feedback-residual-v3"):''')
edit(path,'"original-command-and-gated-feedback-residual-v1", "original-command-and-tracking-feedback-residual-v2")\n',
 '''"original-command-and-gated-feedback-residual-v1", "original-command-and-tracking-feedback-residual-v2",
                "original-command-and-separated-tracking-feedback-residual-v3")
''')
path='scripts/train_flat_base.py'
edit(path,'''    if plan.get("actor_adaptation") in ("original-command-and-gated-feedback-residual-v1",
                                       "original-command-and-tracking-feedback-residual-v2"):''',
'''    if plan.get("actor_adaptation") in ("original-command-and-gated-feedback-residual-v1",
                                       "original-command-and-tracking-feedback-residual-v2",
                                       "original-command-and-separated-tracking-feedback-residual-v3"):''')
edit(path,'    if plan.get("training_command_reference"):\n        sources.append(ROOT / "scripts/flat_rmcs_command_reference.py")',
'''    if plan.get("actor_adaptation") == "original-command-and-separated-tracking-feedback-residual-v3":
        sources.append(ROOT / "src/wheeled_algo/separated_feedback.py")
    if plan.get("training_command_reference"):
        sources.append(ROOT / "scripts/flat_rmcs_command_reference.py")''')
edit(path,'''        feedback_kinds = ("original-command-and-gated-feedback-residual-v1",
                          "original-command-and-tracking-feedback-residual-v2")''',
'''        feedback_kinds = ("original-command-and-gated-feedback-residual-v1",
                          "original-command-and-tracking-feedback-residual-v2",
                          "original-command-and-separated-tracking-feedback-residual-v3")
        separated_feedback = plan.get("actor_adaptation") == feedback_kinds[2]''')
edit(path,'                    tracking_gate=plan["actor_adaptation"] == feedback_kinds[1]).to(args.device)',
'                    tracking_gate=plan["actor_adaptation"] in feedback_kinds[1:]).to(args.device)')
edit(path,'''                from functools import partial
                configure_command_residual_training = partial(configure_command_residual_training,
                    train_command=plan.get("feedback_training_scope") != "feedback_only")''',
'''                if separated_feedback:
                    from wheeled_algo.separated_feedback import (SeparatedFeedbackActor,
                        configure_separated_feedback_training, extend_separated_feedback_parent)
                    old_actor_names = list(dict(runner.alg.actor.named_parameters()))
                    runner.alg.actor.mlp = SeparatedFeedbackActor(runner.alg.actor.mlp).to(args.device)
                    configure_command_residual_training = configure_separated_feedback_training
                else:
                    from functools import partial
                    configure_command_residual_training = partial(configure_command_residual_training,
                        train_command=plan.get("feedback_training_scope") != "feedback_only")''')
edit(path,'''            if (feedback_residual and info.get("actor_adaptation") in feedback_kinds
                    and info["actor_adaptation"] != plan["actor_adaptation"]):''',
'''            if separated_feedback and info.get("actor_adaptation") not in feedback_kinds[1:]:
                raise ValueError("Separated feedback must extend the tracking parent or resume its exact own state")
            if separated_feedback and info.get("actor_adaptation") == feedback_kinds[1]:
                if not migration:
                    raise ValueError("Separated feedback requires a declared identity-extension plan migration")
                checkpoint, report["separated_feedback_identity_extension"] = extend_separated_feedback_parent(
                    checkpoint, runner.alg.actor, runner.alg.critic, runner.alg.optimizer, old_actor_names)
                report["active_branch_actor_updates"] = info["active_branch_actor_updates"]
                report["feedback_branch_actor_updates"] = info["feedback_branch_actor_updates"]
                report["post_split_actor_updates"] = 0
                report["inherited_shared_feedback_updates"] = info["feedback_branch_actor_updates"]
            elif (feedback_residual and info.get("actor_adaptation") in feedback_kinds
                    and info["actor_adaptation"] != plan["actor_adaptation"]):''')
edit(path,'''                if feedback_residual:
                    report["feedback_branch_actor_updates"] = info["feedback_branch_actor_updates"]
            runner.alg.load(checkpoint, load_cfg=None, strict=True)''',
'''                if feedback_residual:
                    report["feedback_branch_actor_updates"] = info["feedback_branch_actor_updates"]
                if separated_feedback:
                    report["post_split_actor_updates"] = info["post_split_actor_updates"]
                    report["inherited_shared_feedback_updates"] = info["inherited_shared_feedback_updates"]
            runner.alg.load(checkpoint, load_cfg=None, strict=True)''')
edit(path,'''            if report.get("feedback_gate_revision"):
                report.update(initialization="exact_parent_state_with_declared_feedback_gate_function_change",''',
'''            if report.get("separated_feedback_identity_extension"):
                report.update(initialization="exact_parent_mean_with_cloned_low_and_fast_feedback_weights_and_Adam",
                    restore_verification_reference="declared_identity_extended_parent_with_feedback_and_Adam_clones",
                    resumption_semantics="parent mean/std/critic/Adam/RNG/counters retained; new fast feedback clones parent shared feedback including Adam step; fresh physical reset")
            if report.get("feedback_gate_revision"):
                report.update(initialization="exact_parent_state_with_declared_feedback_gate_function_change",''')
edit(path,'''            if feedback_residual:
                report["feedback_branch_actor_updates"] += int(enabled)''',
'''            if feedback_residual:
                report["feedback_branch_actor_updates"] += int(enabled)
            if separated_feedback:
                report["post_split_actor_updates"] += int(enabled)''')
edit(path,'''                    writer.add_scalar("FlatTransfer/feedback_branch_actor_updates", report["feedback_branch_actor_updates"], report["successful_updates_total"])''',
'''                    writer.add_scalar("FlatTransfer/feedback_branch_actor_updates", report["feedback_branch_actor_updates"], report["successful_updates_total"])
                if separated_feedback:
                    writer.add_scalar("FlatTransfer/post_split_actor_updates", report["post_split_actor_updates"], report["successful_updates_total"])''')
path='scripts/export_flat_base.py'
edit(path,'    from wheeled_algo.guarded_feedback_residual import KIND as FEEDBACK_KIND, TRACKING_KIND',
'    from wheeled_algo.guarded_feedback_residual import KIND as FEEDBACK_KIND, TRACKING_KIND\n    from wheeled_algo.separated_feedback import KIND as SEPARATED_KIND')
edit(path,'    if info.get("actor_adaptation") in (KIND, FEEDBACK_KIND, TRACKING_KIND):',
'    if info.get("actor_adaptation") in (KIND, FEEDBACK_KIND, TRACKING_KIND, SEPARATED_KIND):')
# Include maximum-height coverage without changing the running evaluator.
path=DEST/'scripts/evaluate_flat_base.py'
s=(HERE/'evaluate_maximum_height_candidate.py').read_text()
s=s.replace('ROOT = Path(__file__).resolve().parents[2]','ROOT = Path(__file__).resolve().parents[1]')
s=s.replace('sys.path.insert(0, str(ROOT / "scripts"))\n','')
s=s.replace('ROOT / "src/wheeled_algo/command_residual.py", ROOT / "src/wheeled_algo/guarded_feedback_residual.py"]',
'''ROOT / "src/wheeled_algo/command_residual.py", ROOT / "src/wheeled_algo/guarded_feedback_residual.py",
                ROOT / "src/wheeled_algo/separated_feedback.py"]''')
path.write_text(s)
for p in DEST.rglob('*.py'):ast.parse(p.read_text(),filename=str(p))
plan=json.loads((ROOT/'contracts/flat_base_finetune_v12.json').read_text())
plan.update(schema='flat-base-finetune-plan-v13',actor_adaptation='original-command-and-separated-tracking-feedback-residual-v3',training_scope='candidate_parent832_identity_fork_low_and_fast_feedback_weights_Adam_same_PPO_rewards_sampling_clock')
p=HERE/'separated_candidate_v13.json';p.write_text(json.dumps(plan,indent=2)+'\n')
files=[str(p.relative_to(DEST)) for p in DEST.rglob('*.py') if not (ROOT/p.relative_to(DEST)).exists() or p.read_bytes()!=(ROOT/p.relative_to(DEST)).read_bytes()]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
receipt=dict(schema='flat-separated-v13-isolated-source-candidate-v1',recorded_at=datetime.now(timezone.utc).isoformat(),changed_files=sorted(files),candidate_source_sha256={n:sha(DEST/n) for n in sorted(files)},candidate_plan_sha256=sha(p),adopted=False,native_optimizer_updates=0,native_environment_steps=0,behavior_qualified=False)
(HERE/'separated_candidate_v13_source001.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
