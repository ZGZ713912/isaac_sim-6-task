"""Authenticate unchanged means/state and the declared native Gaussian revision."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(Path(__file__).parent)]
from export_robust_getup import load_mean
from iteration037_migration_audit import equal
from rsl_rl.modules.distribution import HeteroscedasticGaussianDistribution
from wheeled_algo.robust_getup import SensorConditionedMean, quiet_gate

torch.set_num_threads(1)
parent = ROOT / "runs/unified_robust_20261022/quiet_wheel_velocity038/model_migrated.pt"
checkpoint = ROOT / "runs/unified_robust_20261022/quiet_axis_exploration039/model_migrated.pt"
old = torch.load(parent, map_location="cpu", weights_only=False)
new = torch.load(checkpoint, map_location="cpu", weights_only=False)
for key in ("actor_state_dict", "critic_state_dict", "optimizer_state_dict"):
    assert equal(old[key], new[key]), key
for key in ("environment_rng", "domain_rng", "torch_rng", "cuda_rng"):
    assert equal(old["infos"][key], new["infos"][key]), key
assert old["infos"]["completed_updates"] == new["infos"]["completed_updates"] == 441
differences = {k: [old["infos"]["settings"].get(k), v] for k, v in new["infos"]["settings"].items()
               if old["infos"]["settings"].get(k) != v}
assert set(differences) == {"version", "quiet_exploration_fraction", "quiet_wheel_exploration_fraction"}
old_mean, _ = load_mean(parent, migration_source_root=parent.parent / "source")
new_mean, _ = load_mean(checkpoint)
wrappers = []
for mean, state in ((old_mean, old), (new_mean, new)):
    task = state["infos"]["settings"]
    wrapper = SensorConditionedMean(mean, task["quiet_exploration_fraction"], task.get("quiet_wheel_exploration_fraction"))
    wrapper.load_state_dict({k.removeprefix("mlp."): v for k, v in state["actor_state_dict"].items()}, strict=True)
    wrappers.append(wrapper.eval())
observations = []
for name, stride in (("quiet_velocity_baseline036", 16), ("side_confirmation032", 40)):
    data = np.load(ROOT / f"runs/unified_robust_20261022/{name}/robust/trajectory.npz")
    observations.append(data["observations"][::stride].reshape(-1, 35).copy())
x = torch.from_numpy(np.concatenate(observations))
with torch.inference_mode():
    before, after = (wrapper(x) for wrapper in wrappers)
    assert torch.equal(before[:, 0], after[:, 0])
    inactive = quiet_gate(x)[:, 0] == 0.
    assert torch.equal(before[inactive], after[inactive])
    distribution = HeteroscedasticGaussianDistribution(6, std_type="log", std_range=(.003, .8))
    distribution.update(after)
    encoded_std = after[:, 1].exp()
    assert encoded_std.min() > .003 and encoded_std.max() < .8
    assert torch.equal(distribution.std, encoded_std)
    actions = distribution.sample()
    normal = torch.distributions.Normal(after[:, 0], encoded_std)
    torch.testing.assert_close(distribution.log_prob(actions), normal.log_prob(actions).sum(-1), rtol=0, atol=0)
    torch.testing.assert_close(distribution.entropy, normal.entropy().sum(-1), rtol=0, atol=0)
    old_normal = torch.distributions.Normal(before[:, 0], before[:, 1].exp())
    native_kl = distribution.kl_divergence((before[:, 0], before[:, 1].exp()), distribution.params)
    torch.testing.assert_close(native_kl, torch.distributions.kl_divergence(old_normal, normal).sum(-1), rtol=0, atol=0)
result = dict(schema="quiet039-migration-audit-v1", checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    new_optimizer_updates=0, robust_consumed=865, base_clock=441,
    all_actor_critic_optimizer_rng_exact=True, settings_revision=differences,
    observations=len(x), mean_raw_action_error=0., nonquiet_distribution_exact=True,
    minimum_native_std=float(encoded_std.min()), maximum_native_std=float(encoded_std.max()),
    native_sample_logprob_entropy_kl_exact=True, native_std_clamp_inactive=True,
    deployment_package_replaced=False, hardware_ready=False)
(Path(__file__).parent / "iteration039_migration.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
