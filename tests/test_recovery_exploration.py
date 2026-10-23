"""Sampling changes must preserve the mean, ordinary scope and Gaussian law."""
from copy import deepcopy
import math

import pytest
import torch
from torch import nn

from wheeled_algo.evaluation_recovery_noise import SampledRecoveryEvaluation,recovery_noise_variants
from wheeled_algo.recovery_exploration import RecoveryNoiseMean
from wheeled_algo.robust_getup import SensorConditionedMean,tilt_gate


class Mean(nn.Linear):
    stance_aware_recovery=True
    recovery_shape_aware=True

    def __init__(self):
        super().__init__(35,6)


def frames():
    x=torch.randn(60,35,generator=torch.Generator().manual_seed(92))*.01
    x[:,:3]=0.;x[:,3]=1.525;x[:,7:10]=torch.tensor([0.,0.,-1.]);x[:,10:14]=0.
    x[20:40,9]=-.3
    x[40:,9]=-.6;x[40:,10]=1.8
    return x


@pytest.mark.parametrize("fraction",[1.,.5,.25,.1])
def test_mean_state_rng_wheels_and_zero_recovery_exact(fraction):
    mean=Mean();x=frames();original=SensorConditionedMean(mean,.06,.35)
    state=deepcopy(original.state_dict());rng=torch.get_rng_state()
    candidate=RecoveryNoiseMean(mean,.06,.35,fraction)
    before,after=original(x),candidate(x)
    assert set(candidate.state_dict())==set(state)
    assert all(torch.equal(v,candidate.state_dict()[k]) for k,v in state.items())
    assert torch.equal(before[:,0],after[:,0]) and torch.equal(before[:,1,4:],after[:,1,4:])
    gate=tilt_gate(x,True,True).ravel()
    assert torch.equal(before[gate==0],after[gate==0])
    assert torch.equal(rng,torch.get_rng_state())
    if fraction==1.:
        assert torch.equal(before,after)
    else:
        full=gate==1.
        torch.testing.assert_close(after[full,1,:4].exp(),before[full,1,:4].exp()*fraction)
        assert (after[:,1,:4]<=before[:,1,:4]).all()


def test_native_distribution_logprob_and_gradient_use_the_same_reduced_std():
    from rsl_rl.modules.distribution import HeteroscedasticGaussianDistribution
    model=RecoveryNoiseMean(Mean(),.06,.35,.25);x=frames()
    output=model(x)
    native=HeteroscedasticGaussianDistribution(6,.5,(.003,.8),"log");native.update(output)
    mean,log_std=output.unbind(-2);std=log_std.clamp(math.log(.003),math.log(.8)).exp()
    expected=torch.distributions.Normal(mean,std)
    actions=mean.detach()+.03
    torch.testing.assert_close(native._distribution.log_prob(actions),expected.log_prob(actions),rtol=0,atol=0)
    torch.testing.assert_close(native._distribution.entropy(),expected.entropy(),rtol=0,atol=0)
    loss=-native._distribution.log_prob(actions).mean();loss.backward()
    assert model.log_std.grad is not None and torch.isfinite(model.log_std.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.mean.parameters())


def test_private_common_draws_match_native_formula_and_preserve_global_rng():
    mean=Mean();x=frames();settings=dict(quiet_exploration_fraction=.06,quiet_wheel_exploration_fraction=.35)
    log_std=torch.tensor([-.6,-.7,-.8,-.9,-.7,-.7]);rng=torch.get_rng_state()
    models=[SampledRecoveryEvaluation(mean,settings,log_std,f,211492) for f in [1.,.5,.25]]
    generator=torch.Generator().manual_seed(211492)
    for _ in range(3):
        epsilon=torch.randn((len(x),6),generator=generator)
        for model in models:
            output=model.sampler(x);std=output[:,1].clamp(math.log(.003),math.log(.8)).exp()
            assert torch.equal(model(x),output[:,0]+epsilon*std)
    assert len({m.sampling_receipt()["generator_state_sha256"] for m in models})==1
    assert all(m.sampling_receipt()["calls"]==3 for m in models)
    assert torch.equal(rng,torch.get_rng_state())
    with pytest.raises(ValueError):models[0](x[:2])


@pytest.mark.parametrize("fraction",[True,0.,.099,1.001,float("nan"),float("inf")])
def test_invalid_fraction_rejected(fraction):
    with pytest.raises(ValueError):RecoveryNoiseMean(Mean(),.06,.35,fraction)


@pytest.mark.parametrize("values",[[],[.25,.25],[.25,.5,.75],[1.],[.09],[True],[float("nan")]])
def test_invalid_grid_rejected(values):
    with pytest.raises(ValueError):recovery_noise_variants(values)


@pytest.mark.parametrize("extra,text",[(["--domain-scale",".35"],"nominal domain0"),
    (["--seeds","1","2"],"one seed"),(["--friction-min",".3"],"no other intervention"),
    (["--recovery-leg-steps",".2"],"only one action-filter"),
    (["--recovery-noise-seed","-1"],"no other intervention")])
def test_invalid_cli_scope_rejected_before_checkpoint(monkeypatch,capsys,tmp_path,extra,text):
    import sys
    from evaluate_robust_getup_batch import main
    output=tmp_path/"not-created"
    monkeypatch.setattr(sys,"argv",["evaluate","--checkpoint",str(tmp_path/"missing.pt"),
        "--output",str(output),"--variants","robust","--case-names","left_asymmetric","right_asymmetric",
        "inverted_asymmetric","moving_push_left","moving_push_right","--recovery-leg-noise-fractions",".5",".25",*extra])
    with pytest.raises(SystemExit) as error:main()
    assert error.value.code==2 and text in capsys.readouterr().err and not output.exists()
