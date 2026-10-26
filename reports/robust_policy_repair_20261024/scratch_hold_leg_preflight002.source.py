"""CPU replay checks the intervention on recorded observations, not dynamics."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch
from tensordict import TensorDict

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports/robust_policy_repair_20261024"
RUN = ROOT / "runs/scratch_standing_20261026/pilot005_static_load"
BASELINE = ROOT / "runs/scratch_standing_20261026/eval005_final256_feedback001"
OUTPUT = REPORTS / "scratch_hold_leg_preflight002.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


assert not OUTPUT.exists()
torch.set_num_threads(2)
checkpoint = RUN / "model_final.pt"
assert sha(checkpoint) == "44ddd2a037dd558bc739bb5721f722199ea1cd36deb1130aad7ac88f9d8cba27"
state = torch.load(checkpoint, map_location="cpu", weights_only=False)
infos = state["infos"]
assert infos["training_origin"] == "scratch" and infos["continuation_forbidden"]
archive = RUN / "source_snapshot"
assert len(infos["source_hashes"]) == 262
assert all(sha(archive / name) == value for name, value in infos["source_hashes"].items())
agent = read(RUN / "agent_config.json")
assert sha(RUN / "agent_config.json") == infos["agent_config_sha256"]
model_name = "src/wheeled_algo/wheel_velocity_increment.py"
model = load_module("hold_frozen_increment", archive / model_name)
helper = load_module("hold_current_probe", ROOT / "scripts/hold_initial_leg_target_probe.py")
result = read(BASELINE / "evaluation/result.json")
assert sha(BASELINE / "evaluation/native_trace.npz") == result["trace_sha256"]
assert sha(BASELINE / "evaluation/feedback_trace.npz") == result["feedback_trace"]["sha256"]
with np.load(BASELINE / "evaluation/feedback_trace.npz", allow_pickle=False) as values:
    frames = torch.from_numpy(values["observation_policy"].copy())
    recorded = torch.from_numpy(values["action_raw_policy"].copy())
assert frames.shape == (1600, 21, 35) and recorded.shape == (1600, 21, 6)
cfg = deepcopy(agent["actor"])
assert cfg.pop("class_name") == "wheeled_algo.wheel_velocity_increment:WheelVelocityIncrementModel"
actor = model.WheelVelocityIncrementModel(TensorDict({"policy": frames[0]}, batch_size=[21]),
    agent["obs_groups"], "actor", 6, **cfg)
actor.load_state_dict(state["actor_state_dict"], strict=True)
actor.eval()
before = {name: value.detach().clone() for name, value in actor.state_dict().items()}
probe = helper.HoldInitialLegTargetProbe(actor).eval()
wrapped = []
with torch.inference_mode():
    for frame in frames:
        wrapped.append(probe(TensorDict({"policy": frame}, batch_size=[21])))
wrapped = torch.stack(wrapped)
native = torch.stack(probe.native_actions)
selected = probe.selected
native_error = float((native - recorded).abs().max())
checks = dict(
    recorded_native_actions_reproduce_on_CPU=torch.allclose(native, recorded, rtol=2e-5, atol=2e-6),
    all1600_first21_inputs_used=probe.calls == 1600,
    first_action_unchanged=torch.equal(wrapped[0], native[0]),
    held_targets_from_each_own_first_action=torch.equal(wrapped[:, selected, :4],
        native[0, selected, :4].expand(1600, -1, -1)),
    every_native_wheel_output_unchanged=torch.equal(wrapped[:, :, 4:], native[:, :, 4:]),
    other_height_outputs_unchanged=torch.equal(wrapped[:, ~selected], native[:, ~selected]),
    actual_later_leg_actions_differ=not torch.equal(wrapped[1:, selected, :4], native[1:, selected, :4]),
    frozen_actor_state_unchanged=all(torch.equal(value, actor.state_dict()[name]) for name, value in before.items()),
    CPU_tests_completed="24 passed" in (REPORTS / "scratch_hold_leg_cpu_tests002.log").read_text(),
    all4_invalid_CLI_configurations_rejected=all(row["exit_code"] == 2 and not row["output_created"]
        for row in read(REPORTS / "scratch_hold_leg_cli_rejections001.json")),
)
paths = [ROOT / name for name in ("scripts/evaluate_scratch_standing.py",
    "scripts/hold_initial_leg_target_probe.py", "tests/test_scratch_hold_initial_leg_target_probe.py")]
frozen = REPORTS / "scratch_hold_leg_preflight002"
frozen.mkdir(exist_ok=False)
for path in paths:
    (frozen / path.name).write_bytes(path.read_bytes())
report = dict(schema="scratch-hold-initial-leg-target-preflight-v1", status="prepared_not_started",
    observed_utc=datetime.now(timezone.utc).isoformat(), checks=checks, all_checks_pass=all(checks.values()),
    checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint), frozen_checkpoint_source_count=262,
    source_sha256={str(path.relative_to(ROOT)): sha(path) for path in paths},
    CPU_replay=dict(source_trace=str(BASELINE / "evaluation/feedback_trace.npz"),
        source_trace_sha256=sha(BASELINE / "evaluation/feedback_trace.npz"), observations=1600 * 21,
        maximum_native_GPU_to_CPU_action_difference=native_error,
        maximum_native_to_wrapped_wheel_difference=float((wrapped[:, :, 4:] - native[:, :, 4:]).abs().max()),
        maximum_later_leg_intervention_raw=float((wrapped[1:, selected, :4] - native[1:, selected, :4]).abs().max()),
        trajectory_prediction=False),
    CPU_tests=dict(path=str(REPORTS / "scratch_hold_leg_cpu_tests002.log"), passed=24),
    CLI_rejections=str(REPORTS / "scratch_hold_leg_cli_rejections001.json"),
    seed=261101, native_validation=False, native_steps=0, new_PPO=0, optimizer_constructed=False,
    adoption_by_training=False, behavior_qualification=False, hardware_ready=False,
    prerequisite="Keep007 running untouched to512 and its own original evaluation; verify actual process/GPU "
        "exit and seal. Run this serial diagnostic only if subsequent evidence still calls for separating "
        "later leg feedback from contact/mechanical coupling; preparation is not automatic adoption or launch.",
    scope="The CPU actor was loaded read-only, outside training, without an optimizer. Original recorded "
        "observations test the action intervention only; they do not predict the altered closed-loop "
        "trajectory or contact benefit. Final requested policy remains direct six-motor RL.",
    preflight_reader_sha256=sha(Path(__file__)))
OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
print(json.dumps(dict(checks=checks, maximum_native_GPU_to_CPU_action_difference=native_error), ensure_ascii=False))
if not report["all_checks_pass"]:
    raise SystemExit(1)
