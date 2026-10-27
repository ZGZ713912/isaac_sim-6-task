"""Physical40deg guidance, unconstrained standing legs and true completion."""
import json
import math
from pathlib import Path

import torch
import pytest

from wheeled_tasks.chassis.standard_prone import initialization_motor_target, physical_knee_degrees, route_potential, stable, support_error, valid_disabled_start, density

ROOT = Path(__file__).resolve().parents[1]
SPEC = json.loads((ROOT / "contracts/wheel_leg_standard_prone_v1.json").read_text())


def test_fixed_common_rotation_moves_the_wheels_toward_support_without_changing_fold():
    from wheeled_algo.encoder_leg_geometry import EncoderLegGeometry
    from wheeled_tasks.chassis.standing_geometry107 import table_path
    bank = json.loads((ROOT / "reports/robust_policy_repair_20261024/pose_bank_height_matched073.json").read_text())
    pose = next(p for p in bank["poses"] if p["name"] == SPEC["initial_target_pose_name"])
    names = ("L_joint1", "LL_joint1", "L_joint3", "R_joint1", "RR_joint1", "R_joint3")
    base = torch.tensor([pose["full_joint_positions"][name] for name in names])
    before = base.clone()
    target = initialization_motor_target(base, SPEC["initial_target_common_rotation_rad"])
    geometry = EncoderLegGeometry(json.loads(table_path().read_text()))
    q = torch.stack((base, target))[:, [0, 1, 3, 4]]
    state = geometry(q, torch.zeros_like(q))
    torch.testing.assert_close(base, before)
    torch.testing.assert_close(state["knee_deg"][0], state["knee_deg"][1], atol=.001, rtol=0.)
    assert state["valid"].all()
    assert state["wheel_body_m"][1, :, 0].mean() > state["wheel_body_m"][0, :, 0].mean()+.08
    assert torch.equal(target[[2, 5]], base[[2, 5]])


@pytest.mark.parametrize("rotation", [[float("nan"), 0.], [4., 0.], [0.]])
def test_invalid_initialization_rotations_fail_before_native_or_training(rotation):
    with pytest.raises(ValueError):
        initialization_motor_target(torch.zeros(6), rotation)


def test_passive_included_angle_respects_opposing_joint_signs_and_model_zero():
    lower = -.08616
    at_lower = torch.tensor([[lower, -lower]])
    torch.testing.assert_close(physical_knee_degrees(at_lower, lower), torch.full((1, 2), 40.))
    at_upper = at_lower + torch.tensor([[math.radians(67.), -math.radians(67.)]])
    torch.testing.assert_close(physical_knee_degrees(at_upper, lower), torch.full((1, 2), 107.))


def potential(knee, alignment=.2, standing=False):
    gravity = torch.tensor([[0., 0., -1.]])
    height = .305 if standing else SPEC["belly_height_m"]
    return route_potential(gravity, torch.tensor([[knee, knee]]), torch.tensor([alignment]), torch.tensor([height]), SPEC)


def test_fold_and_under_com_improvements_receive_dense_credit_from_extended_legs():
    assert potential(107.)[0] < potential(85.)[0] < potential(65.)[0] < potential(43.)[0]
    assert potential(65., .3)[0] < potential(65., .2)[0] < potential(65., .1)[0]
    assert potential(107.)[1]["fold_score"].item() > .01
    assert (potential(106.)[1]["fold_score"]-potential(107.)[1]["fold_score"]).item() > .01


def test_reward_near_stop_does_not_prefer_driving_through_the40deg_stop():
    torch.testing.assert_close(potential(40.)[0], potential(43.)[0])
    assert potential(39.)[0] < potential(40.)[0]


def test_folding_guidance_stays_on_for_belly_grounded_upright_body_but_fades_at_standing_height():
    assert potential(40.)[0] > potential(100.)[0]
    torch.testing.assert_close(potential(40., standing=True)[0], potential(100., standing=True)[0])
    assert not potential(40., standing=True)[1]["folding_guidance_weight"].any()


def test_dense_route_reward_credits_folding_before_rising_and_releases_it_after_lift():
    def route_reward(knee, standing=False):
        scores = potential(knee, standing=standing)[1]
        terms = density(torch.tensor([[0., 0., -1.]]), torch.zeros(1, 3), torch.zeros(1, 3),
            torch.zeros(1, 6), torch.zeros(1), torch.zeros(1), torch.tensor([False]), SPEC, route_scores=scores)
        return terms["folding_progress"]+terms["support_alignment"]
    assert route_reward(43.) > route_reward(85.) > route_reward(107.)
    assert route_reward(40., standing=True) == route_reward(100., standing=True) == 0.


def test_support_capture_releases_folding_before_the_body_reaches_standing_height():
    gravity = torch.tensor([[0., 0., -1.]]).repeat(3, 1)
    _, scores = route_potential(gravity, torch.full((3, 2), 43.),
        torch.tensor([.3, .06, 0.]), torch.full((3,), .215), SPEC)
    weight = scores["folding_guidance_weight"]
    assert weight[0] > weight[1] > weight[2] == 0.
    assert weight[1] < .06
    # At aligned support, extending does not sacrifice a held-fold reward
    # while waiting for the body's dynamics to produce actual upward motion.
    assert potential(43., 0.)[0] == potential(64., 0.)[0]


def test_level_cost_resolves_small_tilt_and_never_rewards_upside_down():
    angles = torch.deg2rad(torch.tensor([0., 2., 5., 10., 18., 180.]))
    gravity = torch.stack((angles.sin(), torch.zeros_like(angles), -angles.cos()), -1)
    height = torch.full_like(angles, .305)
    _, scores = route_potential(gravity, torch.full((6, 2), 64.), torch.zeros(6), height, SPEC)
    terms = density(gravity, torch.zeros(6, 3), torch.zeros(6, 3), torch.zeros(6, 6),
        torch.zeros(6), torch.zeros(6), torch.ones(6, dtype=torch.bool), SPEC, route_scores=scores)
    assert terms["level_error_cost"][0] == 0.
    assert (torch.diff(terms["level_error_cost"]) < 0.).all()
    assert terms["level_error_cost"][4] < -6.


def test_real_supported_leg_extension_geometry_has_dense_credit_without_completion():
    from wheeled_algo.encoder_leg_geometry import EncoderLegGeometry
    from wheeled_tasks.chassis.standing_geometry107 import table_path
    bank = json.loads((ROOT / "reports/robust_policy_repair_20261024/pose_bank_height_matched073.json").read_text())
    names = ("L_joint1", "LL_joint1", "L_joint3", "R_joint1", "RR_joint1", "R_joint3")
    pose = next(p for p in bank["poses"] if p["name"] == SPEC["initial_target_pose_name"])
    fold = initialization_motor_target(torch.tensor([pose["full_joint_positions"][n] for n in names]),
        SPEC["initial_target_common_rotation_rad"])
    stand = bank["poses"][bank["diagnostic_height_starts"]["mapping"]["stand"]]
    target = torch.tensor([stand["full_joint_positions"][n] for n in names])
    q = torch.lerp(fold, target, torch.linspace(0., 1., 11)[:, None])[:, [0, 1, 3, 4]]
    geometry = EncoderLegGeometry(json.loads(table_path().read_text()))(q, torch.zeros_like(q))
    assert geometry["valid"].all()
    # Flat wheel support with the sealed60mm wheel radius. This is a feasible
    # geometry audit, not a claim that all interpolated states are dynamically reachable.
    height = .06-geometry["wheel_body_m"][:, :, 2].mean(-1)
    gravity = torch.tensor([[0., 0., -1.]]).repeat(11, 1)
    error = geometry["wheel_body_m"][:, :, :2].mean(1).norm(dim=-1)
    _, scores = route_potential(gravity, geometry["knee_deg"], error, height, SPEC)
    terms = density(gravity, torch.zeros(11, 3), torch.zeros(11, 3), torch.zeros(11, 6),
        torch.zeros(11), torch.zeros(11), torch.zeros(11, dtype=torch.bool), SPEC,
        route_scores=scores, recovered=torch.zeros(11, dtype=torch.bool))
    assert (torch.diff(sum(terms.values())) > 0.).all()


def test_unrecovered_time_cost_ends_only_after_actual_completion_and_favors_faster_recovery():
    times = torch.arange(1000)*.005
    def cost(completion):
        terms = density(torch.tensor([[0., 0., -1.]]).repeat(len(times), 1),
            torch.zeros(len(times), 3), torch.zeros(len(times), 3), torch.zeros(len(times), 6),
            torch.zeros(len(times)), torch.zeros(len(times)), torch.zeros(len(times), dtype=torch.bool),
            SPEC, recovered=times >= completion)
        return terms["unrecovered_time_cost"]
    early, late = cost(1.5), cost(3.)
    assert (early[times < 1.5] == -1.).all() and not early[times >= 1.5].any()
    assert early.sum()*.005 > late.sum()*.005


def test_horizontal_alignment_above_the_com_does_not_count_as_support_below_it():
    com = torch.tensor([[0., 0., .3], [0., 0., .3]])
    wheel = torch.tensor([[0., 0., .1], [0., 0., .5]])
    error = support_error(wheel, com, SPEC["support_min_below_com_m"])
    assert error[0] == 0. and error[1] > .2


def test_rising_toward305mm_does_not_lose_dense_reward_before_completion_credit():
    heights = torch.tensor([SPEC["belly_height_m"], .20, .23, .26, .285, .305])
    gravity = torch.tensor([[0., 0., -1.]]).expand(len(heights), -1)
    _, scores = route_potential(gravity, torch.full((len(heights), 2), 43.),
        torch.zeros_like(heights), heights, SPEC)
    terms = density(gravity, torch.zeros(len(heights), 3), torch.zeros(len(heights), 3),
        torch.zeros(len(heights), 6), torch.zeros_like(heights), torch.zeros_like(heights),
        torch.zeros_like(heights, dtype=torch.bool), SPEC, route_scores=scores)
    # Completion has not happened in any row. This checks the transition goal,
    # independently of the sparse stable-standing bonus and potential shaping.
    assert (torch.diff(sum(terms.values())) > 0.).all()


def test_height_credit_does_not_prefer_overshooting305mm():
    gravity = torch.tensor([[0., 0., -1.]]).repeat(2, 1)
    _, scores = route_potential(gravity, torch.full((2, 2), 64.), torch.zeros(2),
                               torch.tensor([.305, .42]), SPEC)
    terms = density(gravity, torch.zeros(2, 3), torch.zeros(2, 3), torch.zeros(2, 6),
                    torch.zeros(2), torch.zeros(2), torch.zeros(2, dtype=torch.bool), SPEC, route_scores=scores)
    assert sum(terms.values())[0] > sum(terms.values())[1]


def test_user_allowed10cm_rise_translation_has_no_position_penalty_but_larger_travel_does():
    excursion = torch.tensor([0., .05, .10, .12, .20])
    terms = density(torch.tensor([[0., 0., -1.]]).repeat(5, 1), torch.zeros(5, 3),
        torch.zeros(5, 3), torch.zeros(5, 6), torch.zeros(5), excursion,
        torch.zeros(5, dtype=torch.bool), SPEC)
    cost = terms["in_place_cost"]
    assert (cost[:3] == 0.).all()
    assert cost[2] > cost[3] > cost[4]


def test_raising_from_folded_belly_to_unfolded_standing_increases_route_potential():
    assert potential(40., .02)[0] < potential(100., .02, standing=True)[0]


def test_upright_attitude_without_height_contact_and_quiet_motion_cannot_complete():
    gravity = torch.tensor([[0., 0., -1.]]).repeat(4, 1)
    height = torch.tensor([.305, .23, .305, .305])
    velocity = torch.zeros(4, 3)
    velocity[3, 0] = .2
    contact = torch.ones(4, 2, dtype=torch.bool)
    contact[2, 0] = False
    result = stable(gravity, height, torch.zeros(4, 3), velocity, contact, torch.zeros(4, 4), SPEC)
    assert result.tolist() == [True, False, False, False]


def test_disabled_start_rejects_front_inverted_standing_and_body_not_grounded():
    gravity = torch.tensor([[0., 0., -1.], [1., 0., 0.], [0., 0., 1.], [0., 0., -1.], [0., 0., -1.]])
    height = torch.tensor([.173, .173, .173, .305, .173])
    force = torch.tensor([100., 100., 100., 100., 0.])
    assert valid_disabled_start(gravity, height, force, SPEC).tolist() == [True, False, False, False, False]


def test_cold_trial_budget_declares_recovery_deadline_and_full_post_completion_standing():
    evaluation = SPEC["evaluation"]
    assert evaluation["seconds"] == evaluation["recovery_deadline_seconds"]+evaluation["post_completion_standing_seconds"]


def test_prone_geometry_and_exploration_survive_the_real_target_decoder():
    from wheeled_tasks.chassis.v5_control import V5Control
    cfg = json.loads((ROOT / "contracts/scratch_standing_base_mechanism_v2.json").read_text())
    bundle = ROOT / cfg["asset_directory"]
    manifest, mechanism = [json.loads((bundle / n).read_text()) for n in ("manifest.json", "model_spec.json")]
    poses = json.loads((ROOT / "reports/robust_policy_repair_20261024/pose_bank_height_matched073.json").read_text())["poses"]
    pose = next(p for p in poses if p["name"] == SPEC["reset_pose_name"])
    q = torch.tensor([[pose["full_joint_positions"][name] for name in V5Control.ACTIVE]])

    def control(clip):
        return V5Control(manifest, mechanism, {"monomial_coefficients_n": []}, {"actuators": {"wheel": {}}},
                         dict(cfg["v5_control"], action_clip=clip), "cpu")

    old, new = control(3.), control(SPEC["leg_action_clip"])
    delta = q-new.nominal
    action = torch.zeros_like(q)
    action[:, list(new.LEGS)] = torch.atan2(delta[:, list(new.LEGS)].sin(), delta[:, list(new.LEGS)].cos())/.25
    exploration = torch.tensor([[.2, .2, 0., -.2, -.2, 0.]])
    torch.testing.assert_close(old.decode(action, q)[0], old.decode(action+exploration, q)[0])
    torch.testing.assert_close(new.decode(action, q)[0], q[:, list(new.LEGS)], atol=2e-6, rtol=0.)
    assert (new.decode(action+exploration, q)[0]-new.decode(action, q)[0]).abs().min() > .01
