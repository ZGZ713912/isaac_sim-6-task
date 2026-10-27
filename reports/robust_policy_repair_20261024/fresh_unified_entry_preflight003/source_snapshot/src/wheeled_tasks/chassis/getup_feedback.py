"""Causal low-level encoder sampling/holding for robustness training."""
import torch


class GetupFeedback:
    def __init__(self, count, device, enabled, cfg, rng):
        self.enabled, self.cfg, self.rng = enabled, cfg, rng
        self.rows = torch.arange(count, device=device)
        self.period = torch.ones(count, device=device, dtype=torch.long)
        self.delay = torch.zeros_like(self.period)
        self.bias = torch.zeros(count, 6, device=device)
        self.q_history = torch.zeros(8, count, 6, device=device)
        self.dq_history = torch.zeros_like(self.q_history)
        self.q = self.bias.clone()
        self.dq = self.bias.clone()
        self.cursor = self.tick = 0

    def reset(self, ids, q, dq):
        for field, key in ((self.period, "pd_feedback_period_ms"), (self.delay, "pd_feedback_delay_ms")):
            lo, hi = self.cfg[key]
            draw = torch.randint(lo, hi + 1, (len(ids),), device=q.device, generator=self.rng)
            nominal = 1 if key == "pd_feedback_period_ms" else 0
            field[ids] = torch.where(self.enabled[ids], draw, nominal)
        self.bias[ids] = (torch.rand(len(ids), 6, device=q.device, generator=self.rng) * 2 - 1) * self.cfg["pd_encoder_bias_rad"]
        self.bias[ids] *= self.enabled[ids, None]
        self.bias[ids[:, None], q.new_tensor([2, 5], dtype=torch.long)] = 0.
        self.q_history[:, ids], self.dq_history[:, ids] = q[None], dq[None]
        self.q[ids], self.dq[ids] = q, dq

    def observe(self, q, dq):
        self.q_history[self.cursor].copy_(q)
        self.dq_history[self.cursor].copy_(dq)
        read = (self.cursor - self.delay) % len(self.q_history)
        sample = self.tick % self.period == 0
        self.q.copy_(torch.where(sample[:, None], self.q_history[read, self.rows] + self.bias, self.q))
        self.dq.copy_(torch.where(sample[:, None], self.dq_history[read, self.rows], self.dq))
        self.cursor = (self.cursor + 1) % len(self.q_history)
        self.tick += 1
        return self.q, self.dq
