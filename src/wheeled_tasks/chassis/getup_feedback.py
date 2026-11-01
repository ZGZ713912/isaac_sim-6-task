"""Causal low-level encoder sampling/holding for robustness training."""
import math
import torch

from .physical_history import delayed_sample, split_delay_ms


class GetupFeedback:
    def __init__(self, count, device, enabled, cfg, rng, dt=.001):
        self.enabled, self.cfg, self.rng = enabled, cfg, rng
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("PD feedback requires positive finite physical dt")
        self.dt = dt
        self.interpolate_delay = dt != .001 and cfg["pd_feedback_delay_ms"][1] > 0
        self.rows = torch.arange(count, device=device)
        self.period = torch.ones(count, device=device, dtype=torch.long)
        self.delay = torch.zeros_like(self.period)
        self.delay_fraction = torch.zeros(count, device=device)
        self.bias = torch.zeros(count, 6, device=device)
        self.q_history = torch.zeros(max(8, math.ceil(cfg["pd_feedback_delay_ms"][1]/(dt*1000.))+2), count, 6, device=device)
        self.dq_history = torch.zeros_like(self.q_history)
        self.q = self.bias.clone()
        self.dq = self.bias.clone()
        self.cursor = self.tick = 0

    def reset(self, ids, q, dq):
        for field, key in ((self.period, "pd_feedback_period_ms"), (self.delay, "pd_feedback_delay_ms")):
            lo, hi = self.cfg[key]
            draw = torch.randint(lo, hi + 1, (len(ids),), device=q.device, generator=self.rng)
            if key == "pd_feedback_period_ms":
                # A physical integrator cannot refresh feedback between ticks.
                ticks = torch.ceil(draw/(self.dt*1000.)).long().clamp_min(1)
                field[ids] = torch.where(self.enabled[ids], ticks, 1)
            else:
                whole, fraction = split_delay_ms(draw, self.dt)
                field[ids] = torch.where(self.enabled[ids], whole, 0)
                self.delay_fraction[ids] = torch.where(self.enabled[ids], fraction, 0.)
        self.bias[ids] = (torch.rand(len(ids), 6, device=q.device, generator=self.rng) * 2 - 1) * self.cfg["pd_encoder_bias_rad"]
        self.bias[ids] *= self.enabled[ids, None]
        self.bias[ids[:, None], q.new_tensor([2, 5], dtype=torch.long)] = 0.
        self.q_history[:, ids], self.dq_history[:, ids] = q[None], dq[None]
        self.q[ids], self.dq[ids] = q, dq

    def observe(self, q, dq):
        self.q_history[self.cursor].copy_(q)
        self.dq_history[self.cursor].copy_(dq)
        sample = self.tick % self.period == 0
        if self.interpolate_delay:
            q_sample = delayed_sample(self.q_history, self.cursor, self.delay, self.delay_fraction, self.rows)
            dq_sample = delayed_sample(self.dq_history, self.cursor, self.delay, self.delay_fraction, self.rows)
        else:
            read = (self.cursor-self.delay) % len(self.q_history)
            q_sample, dq_sample = self.q_history[read, self.rows], self.dq_history[read, self.rows]
        self.q.copy_(torch.where(sample[:, None], q_sample + self.bias, self.q))
        self.dq.copy_(torch.where(sample[:, None], dq_sample, self.dq))
        self.cursor = (self.cursor + 1) % len(self.q_history)
        self.tick += 1
        return self.q, self.dq
