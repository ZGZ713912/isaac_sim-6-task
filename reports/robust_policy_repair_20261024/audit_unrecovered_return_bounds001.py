"""Bound cylinder-orientation uncertainty in failed-episode return replay."""
import hashlib
import json
import math
from pathlib import Path
import sys
import time

sys.path[:0] = ["src", "scripts", str(Path(__file__).resolve().parent)]
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from wheeled_tasks.chassis.unified_getup import joint_tracking_score
from wheeled_tasks.chassis.gap_robustness import leg_quality
from recovery_support_candidate001 import recovery_support_cost

base = Path("runs/robust_policy_repair_20261024")
names = ("diagnosis012_final_arithmetic_nominal", "diagnosis015_bias015_nominal")
rows = []
initial = None
for run in names:
    directory = base / run / "robust"
    report = json.loads((directory / "report.json").read_text())
    task = json.loads((directory.parent / "task_contract.json").read_text())["settings"]
    z = np.load(directory / "trajectory.npz")
    assert hashlib.sha256((directory / "trajectory.npz").read_bytes()).hexdigest() == report["trajectory_sha256"]
    if initial is None:
        initial = z["observations"][0].copy()
    else:
        assert np.array_equal(initial, z["observations"][0])
    writer = SummaryWriter(str(directory / "metrics"), filename_suffix=".unrecovered-return-bounds")
    for lane in report["lanes"]:
        i = lane["lane"]
        if not lane["recovery"] or lane["recovered"]:
            continue
        assert not z["recovered"][:, i].any()
        mask = z["alive"][:, i] & ~z["passive"][:, i]
        ids = np.flatnonzero(mask)
        assert len(ids) and np.all(np.diff(ids) == 1) and ids[0] > 0
        value = lambda k: torch.tensor(z[k][ids, i], dtype=torch.float64)
        gravity = value("gravity")
        upright = ((-gravity[:, 2] - .5) / .5).clamp(0., 1.)
        command, velocity, omega = value("commands"), value("velocity"), value("omega")
        tracking = joint_tracking_score(velocity[:, 0] - command[:, 0], omega[:, 2] - command[:, 1])
        effort, speed = value("motor_effort"), value("motor_velocity")
        previous = value("observations")[:, 22:28][:, [0, 1, 4, 2, 3, 5]]
        difference = value("actions") - previous
        difference[:, [0, 1, 3, 4]] = torch.atan2(
            (difference[:, [0, 1, 3, 4]] * .25).sin(),
            (difference[:, [0, 1, 3, 4]] * .25).cos()) / .25
        quality = leg_quality(value("leg_speed_peak_rad_s"), value("leg_saturation_fraction"),
            value("leg_effort_slew_rms_nm"), torch.ones(len(ids), dtype=torch.bool), task["gap_leg_quality"])
        density = (1. - gravity[:, 2] + upright * (2. * torch.exp(-((value("height") - command[:, 2]) / .08).square()) + 4. * tracking)
            + 2. * value("wheel_contact").mean(-1) - .02 * omega.square().sum(-1)
            - .0003 * effort.square().mean(-1) - .002 * difference.square().mean(-1)
            - task["recovery_leg_speed_cost"] * speed[:, [0, 1, 3, 4]].square().mean(-1)
            - .0002 * (value("impact") - 400.).clamp_min(0.) - .5 - quality)
        reach = recovery_support_cost(value("wheel_clearance"), gravity[:, 2],
                                      torch.ones(len(ids), dtype=torch.bool), 1.)
        gamma, gain = task["reward_gamma"], task["reward_potential_gain"]
        discount = gamma ** torch.arange(len(ids), dtype=torch.float64)
        terminal = any(lane["termination_causes"])
        dense_return = float((discount * density * .005).sum())
        added = float((discount * reach * .005).sum())
        terminal_return = -3. * gamma ** (len(ids) - 1) if terminal else 0.

        def potential_bounds(tick):
            base_value = .5 * (1. - float(z["gravity"][tick, i, 2])) + .5 * min(1., max(0., float(z["height"][tick, i]) / .305))
            clearance = z["wheel_clearance"][tick, i].astype(np.float64)
            half_width, radius = .02500000037252903 / 2., .06
            # For ANY cylinder-axis orientation the vertical extent lies in
            # [half_width, sqrt(radius**2+half_width**2)]. No sphere assumption.
            extent_max = math.hypot(radius, half_width)
            lower = base_value + .3 * np.exp(-np.maximum(0., clearance + extent_max - radius) / .12).mean()
            upper = base_value + .3 * np.exp(-np.maximum(0., clearance + half_width - radius) / .12).mean()
            return float(lower), float(upper)

        p0, p1 = potential_bounds(ids[0] - 1), potential_bounds(ids[-1])
        end = (0., 0.) if terminal else p1
        shaping = [gain * (-p0[1] + gamma ** len(ids) * end[0]),
                   gain * (-p0[0] + gamma ** len(ids) * end[1])]
        old = [dense_return + terminal_return + v for v in shaping]
        new = [v - added for v in old]
        row = dict(run=run, lane=i, name=lane["name"], family=lane["post_settle_family"],
            survived=lane["survived"], termination=lane["termination_causes"],
            active_seconds=len(ids) * .005, native_dense_discounted_return=dense_return,
            terminal_discounted_return=terminal_return, potential_shaping_return_bounds=shaping,
            original_return_bounds=old, candidate_added_discounted_cost=added,
            candidate_return_bounds=new)
        rows.append(row)
        for key, item in row.items():
            if isinstance(item, (float, int)):
                writer.add_scalar("RecoveryReachReturn/" + str(i) + "/" + key, item, 1501)
        print(run, lane["name"], "old", np.round(old, 4), "new", np.round(new, 4), "terminal", terminal)
    writer.close()

out = dict(schema="unrecovered-trajectory-return-bounds-v1", rows=rows,
    scope="Only never-recovered designated recovery lanes are replayed. Quiet/request-standing terms vanish in this subset. Potential is bounded for all cylinder orientations, not approximated as a sphere. No value bootstrap or policy gradient is reconstructed; surviving12s trajectories remain right-censored.",
    equal_initial_actor_inputs=True, proposed_weight=1., optimizer_updates=0,
    native_reward_changed=False, actual_experiment_consumed=2587,
    qualification=None, hardware_started=False, created_unix=time.time())
Path("reports/robust_policy_repair_20261024/unrecovered_return_bounds001.json").write_text(json.dumps(out, indent=2) + "\n")
