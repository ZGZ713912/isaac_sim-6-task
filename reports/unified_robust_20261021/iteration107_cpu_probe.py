"""CPU scope check for the predeclared one-time40/50-degree bracket."""
from pathlib import Path
import hashlib
import inspect
import json
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
REPORTS = Path(__file__).parent
RUNS = ROOT / "runs/unified_robust_20261022"
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(REPORTS)]
from early_roll075_prototype import EarlyRollPrototype, roll_gate
from export_robust_getup import load_mean
from iteration045_runner import write
from wheeled_algo.evaluation_early_roll import EarlyRollEvaluation
from wheeled_algo.robust_getup import quiet_gate


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    checkpoint = RUNS / "offline_recovery096/model_migrated.pt"
    assert sha(checkpoint) == "eb49ecdf871d5d2ca218586b75072c62f61771fc8431171f363c95d8bbb7a50f"
    mean, _ = load_mean(checkpoint)
    state = {key: value.clone() for key, value in mean.state_dict().items()}
    rng = torch.get_rng_state().clone()
    assert inspect.getsource(EarlyRollPrototype.forward) == inspect.getsource(EarlyRollEvaluation.forward)
    path = RUNS / "high_recenter_full066/robust/trajectory.npz"
    report = json.loads(path.with_name("report.json").read_text())
    assert sha(path) == report["trajectory_sha256"]
    with np.load(path) as data:
        x = torch.from_numpy(data["observations"].reshape(-1, 35)[::23].copy())
    results = {}
    with torch.inference_mode():
        reference = mean(x)
        for full in (40., 50.):
            prototype = EarlyRollPrototype(mean, 20., full)
            native = EarlyRollEvaluation(mean, 20., full)
            original, proposed = roll_gate(mean, x, 20., full)
            changed = proposed.ravel() > original.ravel()
            actual = native(x)
            assert torch.equal(actual, prototype(x))
            assert torch.equal(actual[~changed], reference[~changed])
            assert not bool(((quiet_gate(x).ravel() > 0.) & changed).any())
            assert bool(torch.isfinite(actual).all()) and bool((actual.abs() <= mean.action_bound).all())
            results[str(full)] = dict(prototype_exact=True, unchanged_gate_inputs=int((~changed).sum()),
                extended_authority_inputs=int(changed.sum()), unchanged_gate_outputs_exact=True,
                finite_and_bounded=True, quiet_outputs_exact=True)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(value, mean.state_dict()[key]) for key, value in state.items())
    value = dict(schema="roll107-cpu-scope-v1", checkpoint_sha256=sha(checkpoint),
        recorded_inputs=len(x), source_trajectory_sha256=sha(path), variants=results,
        all_weights_RNG_exact=True, new_macro_updates=0, robust_consumed=1384,
        native_improvement_verified=False)
    assert not (REPORTS / "iteration107_cpu.json").exists()
    write(REPORTS / "iteration107_cpu.json", value)
    print(json.dumps(value), flush=True)


if __name__ == "__main__":
    main()
