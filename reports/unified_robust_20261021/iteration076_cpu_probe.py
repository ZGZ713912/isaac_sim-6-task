"""Exercise initial-state acceptance and corruption rejection without native PPO."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile

import torch

import iteration076_runner as run


def main():
    torch.set_num_threads(1)
    original = torch.load(run.PARENT, map_location="cpu", weights_only=False)
    plan = json.loads((run.REPORTS / "iteration076_plan.json").read_text())
    source_reports = run.REPORTS
    accepted = []
    with tempfile.TemporaryDirectory(prefix="quiet076-guard-fixture-") as folder:
        run.REPORTS = Path(folder) / "reports"
        run.TRAIN = Path(folder) / "synthetic_initial"
        run.REPORTS.mkdir()
        run.TRAIN.mkdir()
        run.write(run.REPORTS / "iteration076_plan.json", plan)
        run.write(run.TRAIN / "progress.json", dict(status="training", new_optimizer_updates=0, trainable_actor_parameters=8106))
        for corruption in (None, "actor", "critic", "Adam", "RNG", "filter_setting", "learning_rate"):
            state = deepcopy(original)
            state["infos"]["settings"] = deepcopy(plan["expected_settings"])
            if corruption == "actor":
                state["actor_state_dict"]["mlp.mean.quiet.2.bias"][0] += .01
            elif corruption == "critic":
                state["critic_state_dict"][next(iter(state["critic_state_dict"]))].flatten()[0] += .01
            elif corruption == "Adam":
                slot = next(v for v in state["optimizer_state_dict"]["state"].values() if "exp_avg" in v)
                slot["exp_avg"].flatten()[0] += .01
            elif corruption == "RNG":
                state["infos"]["torch_rng"][0] ^= 1
            elif corruption == "filter_setting":
                state["infos"]["settings"]["quiet_output_wheel_alpha"] = .5
            elif corruption == "learning_rate":
                state["optimizer_state_dict"]["param_groups"][0]["lr"] = 1e-4
            torch.save(state, run.TRAIN / "model_initial.pt")
            rejected = False
            try:
                run.audit()
            except (AssertionError, ValueError):
                rejected = True
            assert rejected == (corruption is not None), "Unexpected guard result:" + str(corruption)
            accepted.append(corruption + " corruption rejected" if corruption else "exact synthetic initial accepted")
    run.write(source_reports / "iteration076_cpu.json", dict(schema="quiet076-guard-cpu-v1",
        checks=accepted, source_parent_sha256=plan["parent_sha256"], check_count=len(accepted),
        fixture="Synthetic initial with actual074 actor/critic/Adam/RNG; no native process",
        new_macro_updates=0, native_initial_audit_passed=False))
    print(json.dumps(accepted), flush=True)


if __name__ == "__main__":
    main()
