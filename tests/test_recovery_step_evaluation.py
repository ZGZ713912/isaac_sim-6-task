"""Recovery speed diagnostics preserve the reference and reject mixed scopes."""
import pytest
import torch
from torch import nn

from wheeled_algo.evaluation_recovery_step import RecoveryStepEvaluation, recovery_step_variants
from wheeled_algo.robust_getup import RobustGetupActor


def actor():
    torch.manual_seed(88)
    mean = RobustGetupActor(nn.Linear(35, 6), canonical_recovery=True, leg_action_clip=16.,
        recovery_leg_step=.3, cyclic_leg_targets=True, smooth_knee_targets=True,
        stance_aware_recovery=True, recovery_shape_aware=True)
    with torch.no_grad():
        mean.recovery[-1].weight.uniform_(-.2, .2)
    return mean


def frames():
    obs = torch.randn(64, 35, generator=torch.Generator().manual_seed(880)) * .01
    obs[:, :3] = 0.
    obs[:, 3] = 1.525
    obs[:, 7:10] = torch.tensor([0., 0., -1.])
    obs[:, 10:14] = 0.
    obs[32:, 7:10] = torch.tensor([0., .9949874, .1])
    return obs


@pytest.mark.parametrize("step", [.3, .2, .15])
def test_original_identity_wheels_and_zero_authority_exact_without_mutating_reference(step):
    mean, obs = actor(), frames()
    before = {k: v.clone() for k, v in mean.state_dict().items()}
    rng = torch.get_rng_state()
    model = RecoveryStepEvaluation(mean, step)
    old, new = mean(obs), model(obs)
    assert torch.equal(new[:, 4:], old[:, 4:])
    assert torch.equal(new[:32], old[:32])
    assert mean.recovery_leg_step == .3
    assert all(torch.equal(v, mean.state_dict()[k]) for k, v in before.items())
    assert all(torch.equal(v, model.mean.state_dict()[k]) for k, v in before.items())
    assert torch.equal(rng, torch.get_rng_state())
    if step == .3:
        assert torch.equal(new, old)
    else:
        assert not torch.equal(new[32:, :4], old[32:, :4])
        circular = torch.atan2(((new-old)[:, :4]*.25).sin(), ((new-old)[:, :4]*.25).cos())
        assert float(circular.abs().max()) <= (.3-step)*.25 + 2e-6
    assert torch.isfinite(new).all() and (new.abs() <= mean.action_bound).all()


@pytest.mark.parametrize("step", [True, 0., .049, .301, float("nan"), float("inf")])
def test_invalid_steps_rejected(step):
    with pytest.raises(ValueError):
        RecoveryStepEvaluation(actor(), step)


def test_unsupported_original_actor_rejected():
    mean = actor()
    mean.smooth_knee_targets = False
    with pytest.raises(ValueError):
        RecoveryStepEvaluation(mean, .2)
    with pytest.raises(ValueError):
        RecoveryStepEvaluation(nn.Linear(35, 6), .2)


@pytest.mark.parametrize("values", [[], [.2,.2], [.2,.15,.1], [.3], [.04], [True], [float("nan")]])
def test_grid_rejects_invalid_alternatives(values):
    with pytest.raises(ValueError):
        recovery_step_variants(values)


def test_grid_always_contains_original_control():
    assert list(recovery_step_variants([.2,.15]).values()) == [.3,.2,.15]


@pytest.mark.parametrize("extra,text", [(["--domain-scale", ".35"], "nominal domain0"),
    (["--quiet-wheel-output-alphas", ".25"], "only one action-filter"),
    (["--seeds", "1", "2"], "one seed"),
    (["--friction-min", ".3"], "no other intervention")])
def test_bad_scope_rejected_before_checkpoint_or_native(monkeypatch,capsys,tmp_path,extra,text):
    from evaluate_robust_getup_batch import main
    import sys
    output = tmp_path/"not-created"
    monkeypatch.setattr(sys,"argv",["evaluate","--checkpoint",str(tmp_path/"missing.pt"),
        "--output",str(output),"--variants","robust","--replicas","4",
        "--case-names","moving_push_left","moving_push_right","inverted_asymmetric",
        "--recovery-leg-steps",".2",".15",*extra])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2 and text in capsys.readouterr().err
    assert not output.exists()
