"""V9 geometry, command isolation, event accounting and interleaved retention."""
from copy import deepcopy
import math

import pytest
import torch

from wheeled_tasks.wheelbipe_port.mixed_curriculum import TaskSchedule, quota_counts, validate
from wheeled_tasks.wheelbipe_port.terrain_curriculum import STAND, SPIN, DRIVE, JUMP
from wheeled_tasks.wheelbipe_port.terrain_objectives import (
    PositionHold, StepDownObserver, BumpObserver, deadband_square, roll_pitch_deg,
    slope_geometry, slope_height, slope_leg_lengths,
)
from wheeled_tasks.wheelbipe_port.training_profile import specification, phase_for_update, configure_profile
from test_wheelbipe_platform_v7 import stream_digest
from test_wheelbipe_foundation_profile import fake_cfg


def schedule(count=2048, update=20000, device="cpu"):
    s = TaskSchedule(count, device, 213, specification("foundation-jump-v9"))
    s.reset(torch.arange(count, device=device), update)
    return s


def command(s, t):
    p = dict(phase_for_update(s.update, s.spec), completed_update=s.update)
    return s.update_commands(torch.full_like(s.height, t), p)


def test_v1_through_v8_keep_exact_command_streams():
    expected = {1: "d1c56848ac479a58f8c05e007a097da4dc1109ddd7ed7dc84d86f09cbac9cd9e",
                7: "8be7ae8ff2f3ea6b79468322275d2b376ff41bb2f25fc69fcb11f8ca5feb06d7",
                8: "8be7ae8ff2f3ea6b79468322275d2b376ff41bb2f25fc69fcb11f8ca5feb06d7"}
    for v in range(1, 9):
        assert stream_digest(f"foundation-jump-v{v}") == expected.get(v, "6fbf92f6536c482ab49f2c3bcf181744ead822f0d4ad5867b539f8f8386cf942")


def test_contract_quotas_clocks_weights_and_original_absolute_milestones():
    s, old = specification("foundation-jump-v9"), specification("foundation-jump-v8")
    assert list(quota_counts(2048, s).values()) == [184,164,164,103,102,62,102,266,102,225,225,164,123,62]
    for key in ("phases", "special_mode_starts", "jump", "platform", "control", "real2sim", "reward_overrides"):
        assert s[key] == old[key]
    assert s["total_updates"] == 20000
    cfg = fake_cfg(); configure_profile(cfg, "foundation-jump-v9")
    assert cfg.rewards["level_roll"] == -.5 and cfg.rewards["step_down_hop"] == -2.
    scaled = specification("foundation-jump-v9", 10000)
    assert scaled["slope"]["unlock"]["full_update"] == 4250
    assert scaled["step_down"]["speed_ramp"] == [1500,4000]
    assert scaled["slope"]["spin"]["late_update"] == 4500
    assert scaled["slope"]["jump"]["probability_ramp"] == [5500,7500]
    assert scaled["balls"]["coverage_ramp"] == [6000,7500]
    assert scaled["bump"]["unlock"]["full_update"] == 7000


def test_slope_wxyz_geometry_plane_and_clearance():
    from wheeled_tasks.wheelbipe_port.model import CollisionSupport
    cfg = specification("foundation-jump-v9")["slope"]
    theta = torch.linspace(0., math.radians(12), 100, dtype=torch.float64)
    yaw = torch.linspace(-math.pi, math.pi, 100, dtype=torch.float64)
    normal, q, center = slope_geometry(theta, yaw, cfg)
    w,x,y,z = q.unbind(-1)
    rotated_z = torch.stack((2*(x*z+w*y), 2*(y*z-w*x), 1-2*(x*x+y*y)), -1)
    torch.testing.assert_close(rotated_z, normal, atol=1e-12, rtol=0.)
    zero = torch.zeros(100, 2, dtype=torch.float64)
    z0 = slope_height(zero, theta, yaw, cfg)
    d = torch.stack((yaw.cos(), yaw.sin()), -1)
    torch.testing.assert_close(slope_height(d, theta, yaw, cfg), z0-theta.tan())
    torch.testing.assert_close(z0-4*theta.sin(), torch.full_like(theta, .05))
    torch.testing.assert_close(center+normal*.3, torch.stack((z0*0,z0*0,z0), -1))


def test_slope_reset_lengths_preserve_plane_difference():
    g = torch.Generator().manual_seed(17)
    n = 2000
    theta = torch.rand(n, generator=g)*math.radians(12)
    yaw = torch.rand(n, generator=g)*2*math.pi
    psi = torch.rand(n, generator=g)*2*math.pi
    height = .28+.10*torch.rand(n, generator=g)
    lengths = slope_leg_lengths(height, theta, psi, yaw)
    assert lengths.min() >= .17 and lengths.max() <= .37
    # Some extreme height/angle pairs hit the declared reset clamp. Only the
    # unclamped configurations can exactly match both side planes.
    interior = ((lengths > .170001) & (lengths < .369999)).all(-1)
    expected = -.437*theta.tan()*torch.sin(psi-yaw)
    torch.testing.assert_close(-(lengths[:,0]-lengths[:,1])[interior], expected[interior], atol=2e-6, rtol=0.)


@pytest.mark.parametrize("update", [0,2499,3000,9000,9999,10999,11999,14000,15000,20000])
def test_unlocks_and_non_jump_control_slots(update):
    s = schedule(update=update); t = s.terrain
    for task,start in (("slope",2500),("step_down",3000),("bump",10000),("balls",12000)):
        if update < start:
            assert not t.active[s.mask(task)].any()
            assert (t.start_x[s.mask(task)] == -40).all()
    if update >= 15000:
        assert t.active[t.terrain_lane].all()
    if update < 11000:
        assert not (t.mode[s.mask("slope")] == JUMP).any()
    age = torch.full_like(s.height, 2.)
    context = s.context(torch.zeros(len(s.role),7), age)
    other = t.terrain_lane & ~(s.mask("slope") & (t.mode == JUMP))
    assert (context[other] == 0).all()
    jump = s.mask("slope") & (t.mode == JUMP)
    assert (context[jump,4] == 1).all() and (context[jump,1] == 0).all()
    torch.testing.assert_close(context[jump,5], s.apex[jump])


def test_spin_drive_reverse_and_shuttle_commands():
    s = schedule(); t = s.terrain
    spin = s.mask("slope") & (t.mode == SPIN)
    for age, active in ((1., False),(2., True),(6.49, True),(6.5, False),(7., False)):
        c,h = command(s, age)
        assert (c[spin,:2] == 0).all()
        assert (c[spin,2] != 0).all() == active
    assert (t.sign[spin] > 0).any() and (t.sign[spin] < 0).any()
    reverse = s.mask("bump") & (t.speed < 0)
    assert reverse.any() and (t.yaw[reverse] == math.pi).all()
    shuttle = s.mask("balls") & (t.mode == DRIVE)
    a = command(s, .5)[0][shuttle,0].clone()
    b = command(s, 2.)[0][shuttle,0].clone()
    torch.testing.assert_close(a, -b)


def test_position_hold_settles_invalidates_and_preserves_jump_anchor():
    s = schedule(100); h = PositionHold(s)
    lane = s.mask("slope")
    xy, c = torch.zeros(100,2), torch.zeros(100,3)
    for _ in range(14): h.update(xy, c, .02)
    assert not h.valid.any()
    h.update(xy, c, .02); assert h.valid[lane].all()
    xy[:,0] = .1; h.update(xy,c,.02)
    torch.testing.assert_close(h.reward_terms()["position_drift"][lane], torch.full((int(lane.sum()),),1.96))
    h.update(xy,c,.02,request=lane,blocked=lane)
    xy[:,0] = .2
    h.update(xy,c,.02,jump_return=lane)
    torch.testing.assert_close(h.anchor[lane,0], torch.full((int(lane.sum()),), .1))
    c[:,0] = .1; h.update(xy,c,.02)
    assert not h.valid.any()


def step_tick(s, o, *, x=1., support=True, tilt=0., ratio=1., nonwheel=0., age=1.):
    n = len(s.role)
    xy = torch.zeros(n,2,2); xy[:,:,0] = x
    surface = s.terrain.terrain_height[:,None].expand(-1,2) if support else torch.zeros(n,2)
    v = lambda value: torch.full((n,),value)
    o.observe(v(age),xy,surface,surface,torch.full((n,2),100.),s.height,v(0.),v(tilt),v(0.),
        v(.5*ratio),v(.5),v(nonwheel))


@pytest.mark.parametrize("tilt,ratio,force,passed", [(0.,1.,0.,True),(13.,1.,0.,False),(0.,.4,0.,False),(0.,1.,3.,False)])
def test_step_down_requires_real_top_edge_landing_and_calm(tilt, ratio, force, passed):
    s = schedule(100); o = StepDownObserver(s,.005,torch.full((100,),21.44))
    lane = s.mask("step_down")
    step_tick(s,o)
    for i in range(100):
        step_tick(s,o,x=2.2,support=False,tilt=tilt,ratio=ratio,nonwheel=force,age=1.+i*.005)
    assert o.seen_top[lane].all() and o.landed[lane].all()
    assert o.success[lane].all() == passed
    if force > 2: assert o.stumble[lane].all()
    o.edge_left.zero_()
    terms = o.reward_terms(torch.zeros(100),torch.zeros(100),torch.ones(100)*.5,torch.ones(100)*.5,torch.ones(100)*.5,.02)
    torch.testing.assert_close(terms["step_down_hop"][lane], torch.full((int(lane.sum()),),.0625))


@pytest.mark.parametrize("ratio,stall,passed", [(1.,False,True),(.5,False,False),(1.,True,False)])
def test_bump_events_and_floor_height_reference(ratio,stall,passed):
    s = schedule(100); o = BumpObserver(s,.005)
    ids = s.mask("bump").nonzero().flatten()
    # Force one known, axis-aligned first strip and keep the remaining strips distant.
    s.terrain.bump_yaw[ids] = 0.
    s.terrain.strip_starts[ids,0] = 1.5
    s.terrain.strip_widths[ids,0] = .3
    xy = torch.zeros(100,2,2); xy[:,:,0] = 1.6
    z = torch.zeros(100); cmd = torch.ones(100)*.5
    o.observe(z+1,xy,z+4,z,cmd*ratio,cmd,z,z)
    xy[:,:,0] = 1.6 if stall else 2.1
    o.observe(z+(10 if stall else 2),xy,z+4,z,cmd*ratio,cmd,z,z)
    assert (o.success_count[ids] == 1).all() == passed
    if not passed: assert (o.failure_count[ids] == 1).all()
    assert (s.support_height(xy)[ids] == 0).all()
    assert (s.terrain.bump_strip_height(torch.zeros_like(xy))[ids] == 0).all()


def test_deadband_and_attitude_use_true_gravity():
    value = deadband_square(torch.tensor([2.,6.,12.,20.]),2.,2.,25.)
    torch.testing.assert_close(value, torch.tensor([0.,4.,25.,25.]))
    r,p = roll_pitch_deg(torch.tensor([[0.,0.,-1.]]))
    assert r.item() == p.item() == 0.


@pytest.mark.parametrize("update", [0,3000,9000,12000,15000,20000])
def test_fixed_interleaved_quotas_and_easy_difficulty_retention(update):
    s = schedule(update=update); t = s.terrain
    roles = s.role.clone(); fractions = quota_counts(2048,s.spec)
    samples = {name:[] for name in ("rpm","speed","coverage","jump")}
    for _ in range(50):
        s.reset(torch.arange(2048),update)
        assert torch.equal(s.role,roles)
        assert all(int(s.mask(name).sum()) == count for name,count in fractions.items())
        spin = s.mask("slope") & (t.mode == SPIN)
        samples["rpm"].append(t.spin_rpm[spin].clone())
        samples["speed"].append(t.speed[s.mask("step_down")].clone())
        samples["coverage"].append(t.coverage[s.mask("balls")].clone())
        samples["jump"].append((t.mode[s.mask("slope")] == JUMP).float())
    if update >= 15000:
        assert (torch.cat(samples["rpm"]) < 50).float().mean() >= .1
        assert (torch.cat(samples["speed"]) < .8).float().mean() >= .25
        assert (torch.cat(samples["coverage"]) < .2).float().mean() >= .1
        # Binomial sampling tolerance: the configured probability is exactly .30.
        assert torch.cat(samples["jump"]).mean() < .32
        assert ((torch.rad2deg(t.slope_angle[s.mask("slope")])-3.).abs() < 1e-5).float().mean() >= 1/8
        assert (t.terrain_height[s.mask("bump")] == .03).float().mean() >= 1/6


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_schedule_and_observers_stay_on_device():
    s = schedule(100,device="cuda:0")
    command(s,2.)
    assert s.support_height(torch.zeros(100,2,2,device=s.device)).is_cuda
    assert PositionHold(s).anchor.is_cuda
    assert StepDownObserver(s,.005,torch.full((100,),21.,device=s.device)).force_peak.is_cuda
