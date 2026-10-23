"""Separate ordinary trajectories, static equality and recovery gains after079."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPORTS = Path(__file__).parent
sys.path[:0] = [str(ROOT / "src"), str(REPORTS)]
from iteration036_compare import read, paired
from iteration045_runner import write


def main():
    before, after = read("high_recenter_full066"), read("dynamic_recovery_full079")
    pairing = paired(before, after)
    lanes = [row["lane"] for row in after[1]["lanes"] if not row["recovery"] and not row.get("push", False)]
    static = [row["lane"] for row in after[1]["lanes"] if row["name"] in ("stand_low", "stand", "stand_high")]
    keys = ("observations", "actions", "motor_position", "motor_velocity", "local_position", "alive")
    static_exact = {key: np.array_equal(before[3][key][:, static], after[3][key][:, static]) for key in keys}
    assert len(lanes) == 84 and len(static) == 12 and all(static_exact.values())
    ordinary = {label: sum(ev[1]["lanes"][lane]["survived"] for lane in lanes)
                for label, ev in (("before", before), ("after", after))}
    motion = {name: {field: dict(before=before[2]["cases"][name][field], after=row[field])
        for field in ("vx_mean_m_s", "yaw_mean_rad_s", "vx_mae_m_s", "yaw_mae_rad_s")}
        for name, row in after[2]["cases"].items() if name.startswith(("arc_", "fast_", "forward", "reverse"))}
    value = dict(schema="recovery079-retention-v1", initial_pairing=pairing,
        ordinary_without_strong_push=dict(episodes=len(lanes), **ordinary),
        ordinary_trajectories_all_exact={key: np.array_equal(before[3][key][:, lanes], after[3][key][:, lanes]) for key in keys},
        static12_trajectories_bitwise_exact=static_exact, motion=motion,
        interpretation="079 improves nominal inverted-asymmetric and left moving falls, while the random140 practical family totals stay unchanged and one random strong-left survival is lost. Ordinary84 survive both; no global trajectory-equivalence claim.",
        retained_as_research_candidate=True, adopted_over065=False,
        next="Resolve quiet filtering and confirm any combined candidate before admission; right moving push remains uncovered",
        robust_consumed=1256, new_macro_updates=0, full_training_admitted=False)
    write(REPORTS / "iteration079_retention.json", value)
    print(json.dumps(value), flush=True)


if __name__ == "__main__":
    main()
