import torch
from wheeled_tasks.chassis.getup_feedback import GetupFeedback


def feedback(enabled):
    return GetupFeedback(2, "cpu", torch.tensor(enabled), dict(pd_feedback_period_ms=[1, 5],
        pd_feedback_delay_ms=[0, 3], pd_encoder_bias_rad=.003), torch.Generator().manual_seed(43))


def test_nominal_pd_feedback_remains_current_and_exact():
    domain = feedback([False, False])
    domain.reset(torch.arange(2), torch.zeros(2, 6), torch.zeros(2, 6))
    for tick in range(12):
        q, dq = torch.full((2, 6), float(tick)), torch.full((2, 6), float(-tick))
        actual_q, actual_dq = domain.observe(q, dq)
        torch.testing.assert_close(actual_q, q)
        torch.testing.assert_close(actual_dq, dq)


def test_pd_delay_is_causal_and_sample_hold_does_not_refresh_early():
    domain = feedback([True, True])
    domain.reset(torch.arange(2), torch.zeros(2, 6), torch.zeros(2, 6))
    domain.period[:] = torch.tensor([1, 3])
    domain.delay[:] = torch.tensor([2, 0])
    domain.bias.zero_()
    sequence = []
    for tick in range(7):
        q, _ = domain.observe(torch.full((2, 6), float(tick)), torch.zeros(2, 6))
        sequence.append(q[:, 0].tolist())
    assert [row[0] for row in sequence] == [0, 0, 0, 1, 2, 3, 4]
    assert [row[1] for row in sequence] == [0, 0, 0, 3, 3, 3, 6]
    other = domain.q[0].clone()
    domain.reset(torch.tensor([1]), torch.full((1, 6), 19.), torch.zeros(1, 6))
    torch.testing.assert_close(domain.q[0], other)
    assert domain.q_history[:, 1].min() == 19.
