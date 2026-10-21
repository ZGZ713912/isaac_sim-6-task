import pytest
import torch
from torch import nn

from flat_action_relaxation_probe import ActionRelaxationProbe, VARIANTS


class Mean(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.arange(6, dtype=torch.float32))

    def forward(self, obs):
        return obs[:, :6] + self.weight


@pytest.mark.parametrize('variant', VARIANTS)
def test_previous_applied_policy_order_and_only_selected_axes(variant):
    mean = Mean()
    before = {k: v.detach().clone() for k, v in mean.state_dict().items()}
    obs = torch.zeros(3, 35)
    obs[:, 22:28] = torch.arange(6, dtype=torch.float32) + 10.
    obs[:, 28:] = 999.  # Context entries are not previous actions.
    probe = ActionRelaxationProbe(mean, variant)
    actual = probe(obs)
    for k, alpha in enumerate(VARIANTS[variant]):
        assert actual[:, k].tolist() == pytest.approx([k + (1-alpha)*10.] * 3)
    assert all(torch.equal(v, before[k]) for k, v in mean.state_dict().items())


def test_control_bitwise_identity_and_no_hidden_state():
    mean = Mean()
    probe = ActionRelaxationProbe(mean, 'control')
    obs = torch.randn(7, 35)
    assert torch.equal(probe(obs), mean(obs))
    assert torch.equal(probe(obs), probe(obs))
    assert set(probe.state_dict()) == {'alpha', 'mean.weight'}


def test_relaxation_keeps_equilibria_and_gradients():
    mean = Mean()
    obs = torch.zeros(1, 35)
    obs[:, 22:28] = mean(obs).detach()
    probe = ActionRelaxationProbe(mean, 'all')
    assert torch.equal(probe(obs), mean(obs))
    probe(obs).sum().backward()
    assert mean.weight.grad.tolist() == [.25] * 6
    with pytest.raises(ValueError, match='frozen'):
        ActionRelaxationProbe(mean, 'tuned_after_results')
