"""Quantify fixed quiet sampling variance on actual observations, without physics."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from rsl_rl.modules.distribution import HeteroscedasticGaussianDistribution

ROOT = Path(__file__).resolve().parents[2]
REPORTS = Path(__file__).parent
RUNS = ROOT / "runs/unified_robust_20261022"
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(REPORTS)]
from export_robust_getup import load_mean
from iteration036_compare import read
from wheeled_algo.robust_getup import SensorConditionedMean, quiet_gate


def main():
    torch.set_num_threads(1)
    checkpoint = RUNS / "quiet_all_prediction052/model_migrated.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    mean, info = load_mean(checkpoint)
    original = SensorConditionedMean(mean, quiet_exploration_fraction=.06, quiet_wheel_exploration_fraction=.35)
    original.log_std.data.copy_(state["actor_state_dict"]["mlp.log_std"])
    revised = deepcopy(original)
    revised.quiet_wheel_exploration_fraction = .1
    evidence = read("all_prediction_static052")
    x = torch.from_numpy(evidence[3]["observations"][::20].reshape(-1, 35).copy())
    moving = torch.from_numpy(read("side_confirmation032")[3]["observations"][::20].reshape(-1, 35).copy())
    with torch.inference_mode():
        a, b = original(x), revised(x)
        ordinary = quiet_gate(moving).ravel() == 0
        assert torch.equal(original(moving)[ordinary], revised(moving)[ordinary])
        assert torch.equal(a[:, 0], b[:, 0])
        assert torch.equal(a[:, 1, :4], b[:, 1, :4])
    rows = {}
    for label, raw in (("original035", a), ("candidate010", b)):
        distribution = HeteroscedasticGaussianDistribution(6, std_type="log", std_range=(.003, .8))
        distribution.update(raw)
        torch.manual_seed(54001)
        action = distribution.sample()
        reference = torch.distributions.Normal(raw[:, 0], raw[:, 1].exp())
        torch.testing.assert_close(distribution.log_prob(action), reference.log_prob(action).sum(-1))
        full_quiet = quiet_gate(x).ravel() == 1
        physical = distribution.std[full_quiet] * torch.tensor([.25] * 4 + [10.] * 2)
        rows[label] = dict(physical_target_sigma_axes=physical.mean(0).tolist(),
            forward_rolling_sigma_m_s=float((physical[:, 4:].square().sum(-1).sqrt() * .03).mean()),
            actual_distribution_likelihood_consistent=True)
    result = dict(schema="quiet054-wheel-sampling-probe-v1", checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        real_inputs=len(x), nonquiet_sampling_exact_inputs=int(ordinary.sum()),
        deterministic_mean_and_leg_sampling_exact=True, wheel_quiet_fraction_before=.35, after=.1,
        base_log_std=original.log_std.tolist(), results=rows, native_runs=0, optimizer_updates=0,
        scope="Fixed-observation white exploration sigma only, not measured body speed or evidence of native training improvement. Sensor/actuator domains unchanged.")
    (REPORTS / "quiet_sampling054_probe.json").write_text(json.dumps(result, indent=2) + "\n")
    with SummaryWriter(str(evidence[0].parent), filename_suffix=".quiet_sampling054") as writer:
        for label, row in rows.items():
            writer.add_scalar("Diagnosis/quiet_sampling054/" + label + "/forward_rolling_sigma_m_s",
                              row["forward_rolling_sigma_m_s"], info["completed_updates"])
    print(json.dumps(result))


if __name__ == "__main__":
    main()
