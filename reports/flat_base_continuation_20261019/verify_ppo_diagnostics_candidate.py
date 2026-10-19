"""CPU equivalence check for the unadopted logging-only PPO extension."""
from copy import deepcopy
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import torch
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from wheeled_algo.command_residual import CommandResidualActor
from wheeled_algo.guarded_feedback_residual import GuardedFeedbackActor, configure_guarded_feedback_training
from wheeled_algo.onnx_actor_import import RecoveredActor

torch.set_num_threads(1)
HERE = Path(__file__).resolve().parent
OLD = HERE / "training_plan_revision012_source/src/wheeled_algo/anchored_ppo.py"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", type=Path, default=HERE / "anchored_ppo_diagnostics_candidate.py")
parser.add_argument("--output", type=Path, default=HERE / "ppo_diagnostics_candidate_cpu001.json")
args = parser.parse_args()
NEW = args.source.resolve()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TeacherAnchoredPPO


def exact(left, right):
    if torch.is_tensor(left):
        return torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(exact(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(exact(a, b) for a, b in zip(left, right))
    return left == right


def exercise(kind, enabled, coefficient):
    torch.manual_seed(611)
    original = RecoveredActor().eval()
    command = CommandResidualActor(original)
    with torch.no_grad():
        command.residual[2].weight.normal_(std=.05)
    mean = GuardedFeedbackActor(command, tracking_gate=True)
    obs = TensorDict(dict(policy=torch.randn(8, 35), critic=torch.randn(8, 83)), batch_size=[8])
    obs["policy"][:4, :3] = 0.
    obs["policy"][4:, 0] = 1.2
    obs["policy"][6:, 2] = -2.
    groups = dict(actor=["policy"], critic=["critic"])
    actor = MLPModel(obs, groups, "actor", 6, hidden_dims=[256, 128, 64], activation="elu",
        obs_normalization=False, distribution_cfg=dict(class_name="GaussianDistribution", init_std=.04, std_type="scalar"))
    actor.mlp = mean
    critic = MLPModel(obs, groups, "critic", 1, hidden_dims=[256, 128, 64], activation="elu", obs_normalization=False)
    alg = kind(actor, critic, RolloutStorage("rl", 8, 4, obs, [6]),
        num_learning_epochs=2, num_mini_batches=2, learning_rate=1e-4,
        schedule="fixed", use_clipped_value_loss=False)
    alg.optimizer = torch.optim.Adam([
        dict(params=actor.parameters(), lr=1e-4, name="actor"),
        dict(params=critic.parameters(), lr=5e-4, name="critic")])
    alg.configure_teacher(deepcopy(original), coefficient=coefficient,
        action_scales=[.025] * 4 + [.04] * 2, min_vx=.75, min_yaw=1.5)
    configure_guarded_feedback_training(actor, enabled, train_command=False)
    initial = deepcopy(actor.state_dict())
    losses = []
    for cycle in range(2):
        torch.manual_seed(811 + cycle)
        with torch.no_grad():
            for _ in range(4):
                actions = alg.act(obs)
                alg.process_env_step(obs, -(actions - .1).square().sum(-1), torch.zeros(8), {})
            alg.compute_returns(obs)
        torch.manual_seed(821 + cycle)
        losses.append(alg.update())
    return dict(actor=deepcopy(actor.state_dict()), critic=deepcopy(critic.state_dict()),
        optimizer=deepcopy(alg.optimizer.state_dict()), rng=torch.get_rng_state().clone(),
        losses=losses, initial=initial)


old_kind, new_kind = load(OLD, "frozen_flat_ppo"), load(NEW, "candidate_flat_ppo")
rows = []
for enabled in (False, True):
    for coefficient in (0., .02):
        before = exercise(old_kind, enabled, coefficient)
        after = exercise(new_kind, enabled, coefficient)
        checks = {key + "_exact": exact(before[key], after[key]) for key in ("actor", "critic", "optimizer", "rng")}
        checks["old_six_losses_exact"] = all(all(new[k] == v for k, v in old.items())
            for old, new in zip(before["losses"], after["losses"], strict=True))
        extras = set(after["losses"][0]) - set(before["losses"][0])
        checks["six_new_metrics_finite"] = len(extras) == 6 and all(
            torch.isfinite(torch.tensor(loss[k])).item() for loss in after["losses"] for k in extras)
        checks["rates_read_from_actual_adam"] = all(loss["actor_learning_rate"] == 1e-4
            and loss["critic_learning_rate"] == 5e-4 for loss in after["losses"])
        checks["clip_and_gradient_ranges"] = all(0 <= loss["policy_clip_fraction"] <= 1
            and loss["actor_gradient_norm"] >= 0 and loss["critic_gradient_norm"] >= 0 for loss in after["losses"])
        changes = [k for k in after["actor"] if not torch.equal(after["initial"][k], after["actor"][k])]
        checks["only_declared_feedback_may_change"] = bool(changes) == enabled and all(
            k.startswith("mlp.feedback.") for k in changes)
        checks["critic_adam_moments_exercised"] = bool(after["optimizer"]["state"])
        rows.append(dict(actor_enabled=enabled, teacher_coefficient=coefficient, checks=checks,
            passed=all(checks.values()), final_metrics={k: after["losses"][-1][k] for k in sorted(extras)}))
report = dict(schema="flat-ppo-diagnostics-candidate-cpu-v1", recorded_at=datetime.now(timezone.utc).isoformat(),
    source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (OLD, NEW)},
    cases=rows, passed=all(r["passed"] for r in rows), adopted=False,
    native_training_optimizer_updates=0, native_environment_steps=0)
args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
print(json.dumps(dict(passed=report["passed"], cases=len(rows), checks=sum(len(r["checks"]) for r in rows))))
if not report["passed"]:
    raise SystemExit("Logging candidate changed the native CPU PPO result")
