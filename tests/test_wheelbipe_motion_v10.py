"""Motion objective discrimination, deployable history and direction diagnostics."""
from copy import deepcopy

import pytest
import torch

from wheeled_tasks.wheelbipe_port.mixed_curriculum import TaskSchedule, quota_counts, validate
from wheeled_tasks.wheelbipe_port.motion import (
    PolicyHistory, actor_from_state_dict, motion_statistics, motion_terms, observation_contract, replace_motion_terms,
)
from wheeled_tasks.wheelbipe_port.scenario_evaluation import scenario_grid,scenario_override
from wheeled_tasks.wheelbipe_port.training_profile import configure_profile,phase_for_update,specification
from test_wheelbipe_foundation_profile import fake_cfg


def test_v10_inherits_all_v9_plant_quota_clock_and_nontranslation_contracts():
    new,old = specification("foundation-jump-v10"),specification("foundation-jump-v9")
    changed = {"schema","status","parameter_overrides","motion","motion_reward_weights","reward_policy","motion_notes"}
    assert {k:v for k,v in new.items() if k not in changed} == {k:v for k,v in old.items() if k not in changed}
    assert quota_counts(4096,new) == quota_counts(4096,old)
    # A new objective must not silently change the command distribution.
    for update in (0,2000,6000,12000,15000,20000):
        a,b = [TaskSchedule(100,"cpu",211,s) for s in (new,old)]
        ids = torch.arange(100)
        a.reset(ids,update); b.reset(ids,update)
        phase = dict(phase_for_update(update,new),completed_update=update)
        for age in (0.,1.,4.,5.1):
            age = torch.full((100,),age)
            ca,ha = a.update_commands(age,phase); cb,hb = b.update_commands(age,phase)
            torch.testing.assert_close(ca,cb,rtol=0.,atol=0.)
            torch.testing.assert_close(ha,hb,rtol=0.,atol=0.)


def test_opt_in_dimensions_reward_units_and_nonzero_low_speed_commands():
    cfg = fake_cfg();configure_profile(cfg,"foundation-jump-v10")
    assert cfg.observation_space == 175 and not cfg.use_frame_stack
    assert cfg.port_motion["error_scale_m_s"] == .25
    assert cfg.stand_still_deadzone_threshold == 1.e-6
    assert cfg.rewards["motion_tracking"] == 2. and cfg.rewards["motion_error"] == -1.
    assert observation_contract(specification("foundation-jump-v10"))["oldest_frame_age_s"] == .08
    assert observation_contract(specification("foundation-jump-v9"))["actor_dim"] == 35
    old = fake_cfg();configure_profile(old,"foundation-jump-v9")
    assert old.port_motion is None and "motion_tracking" not in old.rewards
    scaled = specification("foundation-jump-v10",10000)
    assert scaled["motion"] == specification("foundation-jump-v10")["motion"]
    assert scaled["slope"]["unlock"]["full_update"] == 4250


def paid(speed,cmd,eligible=None):
    speed,cmd = torch.as_tensor(speed),torch.as_tensor(cmd)
    eligible = torch.ones_like(speed,dtype=torch.bool) if eligible is None else eligible
    t = motion_terms(cmd,speed,eligible,specification("foundation-jump-v10")["motion"])
    return 2*t["motion_tracking"]-t["motion_error"]


def test_objective_rejects_observed_half_speed_and_backward_equilibria():
    assert paid(.5,.5)-paid(.22344,.5) > 2.
    assert paid(0.,0.)-paid(-.4237,0.) > 2.
    assert paid(-.5,-.5) > paid(-.6794,-.5) > paid(.5,-.5)
    # Sub-deadzone positive commands remain positive, not zero references.
    assert paid(.05,.05) > paid(0.,.05) > paid(-.05,.05)


def test_objective_is_signed_symmetric_and_cost_does_not_flatten_at_large_error():
    cmd = torch.tensor([.05,.5,1.,2.7])
    speed = torch.tensor([-.1,.22,.7,-2.],requires_grad=True)
    torch.testing.assert_close(paid(speed,cmd),paid(-speed,-cmd))
    grad = torch.autograd.grad(paid(speed,cmd).sum(),speed)[0]
    assert (grad > 0.).all() and grad[-1] > 3.
    masked = paid(speed,cmd,torch.zeros(4,dtype=torch.bool))
    torch.testing.assert_close(masked,torch.zeros_like(masked))


def test_jump_drop_exclusions_retain_legacy_payment_and_ordinary_terms_are_not_doubled():
    eligible=torch.tensor([True,False])
    terms=dict(track_lin_vel_xy=torch.tensor([.8,.7]),track_lin_vel_xy_square=torch.tensor([.1,.2]))
    out=replace_motion_terms(terms,torch.zeros(2),torch.zeros(2),eligible,specification("foundation-jump-v10")["motion"])
    torch.testing.assert_close(out["track_lin_vel_xy"],torch.tensor([0.,.7]))
    torch.testing.assert_close(out["track_lin_vel_xy_square"],torch.tensor([0.,.2]))
    torch.testing.assert_close(out["motion_tracking"],torch.tensor([1.,0.]))
    torch.testing.assert_close(out["motion_error"],torch.zeros(2))


def test_history_appends_once_preserves_order_and_does_not_alias_returned_frames():
    history = PolicyHistory(2,5,35,"cpu")
    first = history.update(torch.ones(2,35),0)
    second = history.update(torch.full((2,35),2.),1)
    torch.testing.assert_close(first,torch.ones(2,175))
    torch.testing.assert_close(second.reshape(2,5,35)[:,0],torch.full((2,35),2.))
    torch.testing.assert_close(second.reshape(2,5,35)[:,1:],torch.ones(2,4,35))
    third = history.update(torch.full((2,35),3.),1).reshape(2,5,35)
    torch.testing.assert_close(third[:,1:],torch.ones(2,4,35))
    history.reset(torch.tensor([0]))
    reset = history.update(torch.full((2,35),4.),1).reshape(2,5,35)
    torch.testing.assert_close(reset[0],torch.full((5,35),4.))
    torch.testing.assert_close(reset[1,1:],torch.ones(4,35))
    after = history.update(torch.full((2,35),5.),2).reshape(2,5,35)
    torch.testing.assert_close(after[1,1],torch.full((35,),4.))
    with pytest.raises(ValueError):history.update(torch.zeros(2,175),3)


def test_direction_and_height_statistics_expose_signed_bias_without_cancellation():
    cmd = torch.tensor([0.,.5,-.5]*3)
    speed = cmd-.28
    height = torch.tensor([.23]*3+[.305]*3+[.42]*3)
    eligible = torch.ones(9,dtype=torch.bool);eligible[-1] = False
    result = motion_statistics(cmd,speed,height,eligible,speed,speed)
    for direction in ("stand","forward","reverse"):
        for band in ("low","mid","high"):
            prefix=f"Motion/{direction}/{band}/"
            count = result[prefix+"sample_envs"]
            assert count == (0 if direction == "reverse" and band == "high" else 1)
            if count: assert result[prefix+"signed_error_mean_m_s"] == pytest.approx(-.28)
    assert result["Motion/forward/low/actual_mean_m_s"] == pytest.approx(.22)


def state_dict(dim):
    layers=[]
    for a,b in ((dim,256),(256,128),(128,64),(64,6)):
        layers += [torch.nn.Linear(a,b),torch.nn.ELU()]
    model = torch.nn.Sequential(*layers[:-1])
    return model,{"actor."+k:v for k,v in model.state_dict().items()}


def test_old_actor_cannot_be_padded_into_v10_and_new_actor_parity_is_exact():
    _,old = state_dict(35)
    with pytest.raises(ValueError,match="175D"):actor_from_state_dict(old,175)
    model,new = state_dict(175)
    with pytest.raises(ValueError,match="35D"):actor_from_state_dict(new,35)
    actor=actor_from_state_dict(new,175)
    sample=torch.randn(9,175)
    torch.testing.assert_close(actor(sample),model(sample),rtol=0.,atol=0.)


def test_motion_grid_contains_matched_forward_reverse_zero_and_exact_heights():
    spec=specification("foundation-jump-v10")
    grid=scenario_grid("motion")
    assert len(grid)==39
    for height in (.23,.305,.42):
        speeds={c["command"][0] for c in grid if c["height"]==height}
        assert 0. in speeds and {x for x in speeds if x>0.} == {-x for x in speeds if x<0.}
    for cell in grid:
        patch,count=scenario_override(spec,cell,16)
        merged=deepcopy(spec);merged.update(patch);validate(merged)
        assert quota_counts(count,merged)[cell["task"]]==16
        assert patch["evaluation"]["duration_s"]==10.


@pytest.mark.parametrize("patch",[{"history_frames":1},{"history_order":"oldest_first"},
    {"tracking_sigma_m_s":float("nan")},{"error_scale_m_s":0.}])
def test_invalid_motion_contract_is_rejected(patch):
    spec=specification("foundation-jump-v10");spec["motion"].update(patch)
    with pytest.raises(ValueError):validate(spec)
