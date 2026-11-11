"""foundation-jump-v7: arcs, faster moving jumps, platform lane, measured gas plant (simulator-free)."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from wheeled_tasks.chassis.gas_spring_compensation import GasSpringCompensation
from wheeled_tasks.wheelbipe_port import real2sim
from wheeled_tasks.wheelbipe_port.mixed_curriculum import (
    DRIVE_OFF, FLAT, OVER, TASKS, JumpObserver, PlatformObserver, TaskSchedule, jump_timing, quota_counts,
    reward_names, tasks_for)
from wheeled_tasks.wheelbipe_port.training_profile import phase_for_update, specification

ROOT = Path(__file__).resolve().parents[1]
V7 = "foundation-jump-v7"


def phase(spec, update):
    row = dict(phase_for_update(update, spec))
    row["completed_update"] = update
    return row


def stream_digest(profile):
    spec = specification(profile)
    s = TaskSchedule(256, "cpu", 119, spec)
    digest = hashlib.sha256(s.role.numpy().tobytes())
    for step in range(600):
        update = step*15
        age = torch.full((256,), (step % 300)*.02)
        if step % 300 == 0:
            s.reset(torch.arange(256), update)
        commands, height = s.update_commands(age, phase(spec, update))
        ref = s.reference(age)
        context = s.context(torch.zeros(256, 7), age)
        for value in (commands, height, ref["height"], ref["velocity"], context, s.delay, s.apex):
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def test_historical_profiles_keep_their_exact_command_streams():
    # Recorded from the pre-v7 implementation; v4, v5 and v6 share it.
    expected = "6fbf92f6536c482ab49f2c3bcf181744ead822f0d4ad5867b539f8f8386cf942"
    for profile in ("foundation-jump-v4", "foundation-jump-v5", "foundation-jump-v6"):
        assert stream_digest(profile) == expected
        assert tasks_for(specification(profile)) == TASKS


def test_v7_quotas_and_reward_names():
    spec = specification(V7)
    counts = quota_counts(2048, spec)
    assert counts["platform"] == 307 and sum(counts.values()) == 2048
    assert reward_names(spec)[-3:] == ("platform_clearance", "platform_events", "platform_impact")
    v6 = specification("foundation-jump-v6")
    for key in ("phases", "reward_overrides", "parameter_overrides", "control", "ppo_overrides",
                "leg_target_projection_penalty", "special_mode_starts"):
        assert spec[key] == v6[key], key


def test_arc_commands_stay_inside_the_centripetal_limit():
    spec = specification(V7)
    s = TaskSchedule(2048, "cpu", 5, spec)
    arcs = 0
    for update in (3000, 8000, 15000):
        s.reset(torch.arange(2048), update)
        for k in range(3):
            age = torch.full((2048,), 5.*k)
            commands, _ = s.update_commands(age, phase(spec, update))
            lanes = s.mask("forward") | s.mask("reverse") | s.mask("dash")
            arc = lanes & (commands[:, 2].abs() > 0)
            arcs += int(arc.sum())
            if not arc.any():
                continue
            assert float((commands[arc, 0]*commands[arc, 2]).abs().max()) <= 2.5+1e-5
            assert float(commands[arc, 2].abs().min()) >= .2-1e-6
            assert float(commands[arc, 0].abs().min()) >= .2-1e-6
            assert bool((commands[arc, 2] > 0).any() and (commands[arc, 2] < 0).any())
    assert arcs > 500


def test_moving_jump_speed_range_and_fraction():
    spec = specification(V7)
    s = TaskSchedule(2048, "cpu", 3, spec)
    s.reset(torch.arange(2048), 9000)
    commands, _ = s.update_commands(torch.zeros(2048), phase(spec, 9000))
    jump = s.mask("jump")
    moving = jump & (commands[:, 0].abs() > 0)
    assert abs(int(moving.sum())-.5*int(jump.sum())) <= 1
    speed = commands[moving, 0].abs()
    assert .3-1e-6 <= float(speed.min()) and float(speed.max()) <= 1.+1e-6


def test_platform_lane_heights_unlock_and_modes():
    spec = specification(V7)
    s = TaskSchedule(2048, "cpu", 11, spec)
    lane = s.mask("platform")
    assert {round(float(v), 3) for v in s.rise[lane]} == {.06, .12, .16, .2, .22}
    assert abs(float((s.rise[lane] == .2).float().mean())-.5) < .01
    assert bool((s.rise[~lane] == 0).all())
    s.reset(torch.arange(2048), 4999)
    assert bool((s.mode[lane] == FLAT).all())
    s.reset(torch.arange(2048), 9000)
    limit = s.platform_height_limit(9000)
    assert limit == pytest.approx(.14)
    assert bool((s.mode[lane & (s.rise > limit+1e-6)] == FLAT).all())
    s.reset(torch.arange(2048), 13000)
    over = float((s.mode[lane] == OVER).float().mean())
    assert bool((s.mode[lane] != FLAT).all()) and abs(over-.7) < .08


def test_over_request_puts_the_wheel_at_the_edge_near_apex_at_commanded_speed():
    spec = specification(V7)
    p, j = spec["platform"], spec["jump"]
    s = TaskSchedule(2048, "cpu", 13, spec)
    s.reset(torch.arange(2048), 20000)
    over = s.mask("platform") & (s.mode == OVER)
    speed, start = s.platform_speed[over], s.start_x[over]
    # Ideal constant-speed run from the start; odometry equals distance.
    t_request = s.trigger_distance[over]/speed
    _, _, to_apex = jump_timing(s.apex[over], j)
    wheel_x = start+speed*(t_request+to_apex)+p["wheel_center_offset_x_m"]
    target = p["near_edge_x_m"]-.5*j["wheel_radius_m"]
    assert float((wheel_x-target).abs().max()) <= p["timing_jitter_s"]*1.0+1e-5
    assert float(s.apex[over & (s.rise == .2)].min()) == pytest.approx(.5)
    assert bool((s.delay[s.mask("platform")] > 999.).all())


def test_odometry_trigger_and_step_context():
    spec = specification(V7)
    s = TaskSchedule(512, "cpu", 17, spec)
    s.reset(torch.arange(512), 20000)
    lane = s.mask("platform")
    over, off = lane & (s.mode == OVER), lane & (s.mode == DRIVE_OFF)
    for k in range(200):
        age = torch.full((512,), .02*k)
        s.advance_platform_requests(age, torch.full((512,), .02))
    assert bool((s.delay[over] < 999.).all()) and bool((s.delay[off] > 999.).all())
    age = torch.full((512,), 4.)
    context = s.context(torch.zeros(512, 7), age)
    assert torch.allclose(context[over, 1], s.rise[over])
    assert bool((context[over, 4] == 1.).all()) and bool((context[off, 4] == 0.).all())
    assert bool((context[s.mask("jump"), 1] == 0.).all())


def test_support_height_is_the_box_under_the_point():
    spec = specification(V7)
    p = spec["platform"]
    s = TaskSchedule(256, "cpu", 19, spec)
    lane = s.mask("platform")
    near = p["near_edge_x_m"]
    xy = torch.tensor([[near+.01, 0.], [near-.01, 0.], [near+p["length_m"]+.01, 0.], [near+.5, p["width_m"]]])
    support = s.support_height(xy.expand(256, -1, -1))
    assert torch.allclose(support[lane, 0], s.rise[lane])
    assert bool((support[:, 1:] == 0).all()) and bool((support[~lane] == 0).all())


def test_over_success_requires_landing_on_top():
    spec = specification(V7)
    s = TaskSchedule(64, "cpu", 23, spec)
    s.reset(torch.arange(64), 20000)
    over = s.mask("platform") & (s.mode == OVER)
    s.delay[over] = 0.
    j = spec["jump"]
    mass = torch.full((64,), 21.)
    for on_top in (False, True):
        o = JumpObserver(s, .005, mass)
        o.flown[over] = True
        o.landed[over] = True
        o.peak_height[over] = 1.
        for k in range(80):
            o.observe(torch.full((64,), 1.+.005*k), torch.full((64,), j["standing_height_m"]),
                torch.zeros(64, 3), torch.tensor([[0., 0., -1.]]).expand(64, -1), torch.zeros(64, 2),
                torch.full((64, 2), 50.), torch.zeros(64), horizontal_speed_error=torch.zeros(64),
                leg_lengths=torch.full((64, 2), .26), on_top=torch.full((64,), on_top))
        assert bool(o.succeeded[over].all()) == on_top


def test_platform_observer_strike_and_drive_off_sequence():
    spec = specification(V7)
    p, j = spec["platform"], spec["jump"]
    s = TaskSchedule(64, "cpu", 29, spec)
    s.reset(torch.arange(64), 20000)
    lane = s.mask("platform")
    o = PlatformObserver(s, .005, torch.full((64,), 21.))
    near = p["near_edge_x_m"]
    down = torch.tensor([[0., 0., -1.]]).expand(64, -1)
    def step(x, bottom, force, force_x, height=.305, age=1.):
        xy = torch.stack((torch.full((64, 2), x), torch.tensor([.2, -.2]).expand(64, -1)), -1)
        support = s.support_height(xy)
        o.observe(torch.full((64,), age), xy, torch.full((64, 2), bottom), support, torch.full((64, 2), force),
            torch.full((64, 2), force_x), torch.full((64,), height), torch.zeros(64, 3), down,
            torch.zeros(64), torch.zeros(64))
    o.begin_interval()
    step(near-.06, .02, 40., -30.)  # wheel pushed back by the face, below the top
    assert bool(o.strike_event[lane & (s.mode == OVER)].all())
    assert not bool(o.strike_event[~lane].any())
    o.reset(torch.arange(64))
    step(near+.5, s.rise.max().item(), 40., 0.)  # supported on top (rise 0.22 max; per-lane rise below)
    rise = s.rise.clone()
    o.reset(torch.arange(64))
    for value in (rise, ):
        xy = torch.stack((torch.full((64, 2), near+.5), torch.tensor([.2, -.2]).expand(64, -1)), -1)
        o.observe(torch.ones(64), xy, value[:, None].expand(-1, 2), s.support_height(xy), torch.full((64, 2), 40.),
                  torch.zeros(64, 2), torch.full((64,), .305), torch.zeros(64, 3), down, torch.zeros(64), torch.zeros(64))
    active = lane & (s.mode != FLAT)
    assert bool(o.seen_top[active].all())
    step(near+p["length_m"]+.3, .1, 0., 0.)  # off the far edge, airborne
    assert bool(o.dropped[active].all())
    for k in range(int(j["landing_hold_seconds"]/.005)+2):
        step(near+p["length_m"]+.5, 0., 60., 0.)
    assert bool(o.drop_succeeded[active].all()) and not bool(o.dropped[~active].any())
    terms = o.reward_terms(torch.ones(64), torch.zeros(64, 2, 2), torch.zeros(64, 2), .02)
    assert set(terms) == {"platform_clearance", "platform_events", "platform_impact"}


def test_measured_gas_plant_matches_both_legs():
    spec = real2sim.load("contracts/wheelbipe_real2sim_v3p2_20261112.json")
    assert spec["gas_spring_plant"]["controller_model_sha256"] == "13db9ada6fa0e111a51406fc950ca0b970d1fd3de0d9ceeac07b22053a952aa6"
    runtime = json.loads((ROOT/"model/wheel_leg_v2.0.2/friction_v1/isaac/runtime.json").read_text())
    settings = dict(runtime["v5_control"]["gas_spring_compensation"])
    settings["model_file"] = str(ROOT/spec["gas_spring_plant"]["model_path"])
    settings["model_sha256"] = spec["gas_spring_plant"]["model_sha256"]
    gas = GasSpringCompensation(settings, root=ROOT, device="cpu")
    for row in gas.model["measurement"]["bins"]:
        for side, column, measured in (("left", 0, row["left_nm"]), ("right", 2, row["right_nm"])):
            geometry = gas.model["sides"][side]
            delta = float(np.interp(row["beta"], geometry["beta_deg"], geometry["delta_rad"]))
            q = torch.zeros(1, 4)
            q[0, 1 if side == "left" else 2] = delta
            torque, _ = gas.evaluate(q)
            assert abs(abs(float(torque[0, column]))-measured) < .5
    s = torch.linspace(float(gas.geometry_compression_min), float(gas.geometry_compression_max), 500)
    force = gas.physical_force(s)
    assert bool((force[1:] >= force[:-1]-1e-3).all()) and float(force.min()) > 250.


def test_v7_selects_the_measured_plant_and_per_side_friction():
    spec = real2sim.load(specification(V7)["real2sim"])
    assert spec["leg_friction"]["per_motor"] is True
    assert spec["gas_spring_plant"]["model_path"] == "model/gas_spring_20261112/gas_spring_model.json"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device placement")
def test_platform_schedule_and_observer_on_cuda():
    spec = specification(V7)
    s = TaskSchedule(128, "cuda", 31, spec)
    s.reset(torch.arange(128, device="cuda"), 20000)
    age = torch.zeros(128, device="cuda")
    s.update_commands(age, phase(spec, 20000))
    s.advance_platform_requests(age, torch.full((128,), .1, device="cuda"))
    assert s.context(torch.zeros(128, 7, device="cuda"), age).is_cuda
    assert s.support_height(torch.zeros(128, 2, 2, device="cuda")).is_cuda
    o = PlatformObserver(s, .005, torch.full((128,), 21., device="cuda"))
    terms = o.reward_terms(age, torch.zeros(128, 2, 2, device="cuda"), torch.zeros(128, 2, device="cuda"), .02)
    assert all(v.is_cuda for v in terms.values())
