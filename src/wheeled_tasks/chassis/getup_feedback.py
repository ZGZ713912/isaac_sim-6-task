"""Causal low-level encoder sampling/holding for robustness training."""
import math
import torch


class GetupFeedback:
    def __init__(self, count, device, enabled, cfg, rng, dt=.001):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("Feedback physical period must be positive and finite")
        self.dt = dt
        self.enabled, self.cfg, self.rng = enabled, cfg, rng
        self.rows = torch.arange(count, device=device)
        self.period = torch.ones(count, device=device, dtype=torch.long)
        self.delay = torch.zeros_like(self.period)
        self.period_ticks = self.period.clone()
        self.delay_ticks = self.delay.clone()
        self.delay_fraction = torch.zeros(count, device=device)
        self.interpolate_delay = dt != .001 and cfg["pd_feedback_delay_ms"][1] > 0
        self.bias = torch.zeros(count, 6, device=device)
        history = max(8, math.ceil(cfg["pd_feedback_delay_ms"][1]*.001/dt)+2)
        self.q_history = torch.zeros(history, count, 6, device=device)
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
        if self.dt != .001:
            # Milliseconds are not solver ticks. Sub-step delays use only the
            # current and preceding physical samples; never a future sample.
            age = self.delay[ids].float()*(.001/self.dt)
            self.delay_ticks[ids] = age.floor().long()
            self.delay_fraction[ids] = age-age.floor()
            # A sampled encoder cannot refresh faster than native physics.
            self.period_ticks[ids] = (self.period[ids].float()*(.001/self.dt)).ceil().long().clamp_min(1)
        self.bias[ids] = (torch.rand(len(ids), 6, device=q.device, generator=self.rng) * 2 - 1) * self.cfg["pd_encoder_bias_rad"]
        self.bias[ids] *= self.enabled[ids, None]
        self.bias[ids[:, None], q.new_tensor([2, 5], dtype=torch.long)] = 0.
        self.q_history[:, ids], self.dq_history[:, ids] = q[None], dq[None]
        self.q[ids], self.dq[ids] = q, dq

    def observe(self, q, dq):
        self.q_history[self.cursor].copy_(q)
        self.dq_history[self.cursor].copy_(dq)
        delay = self.delay if self.dt == .001 else self.delay_ticks
        period = self.period if self.dt == .001 else self.period_ticks
        read = (self.cursor - delay) % len(self.q_history)
        q_delayed, dq_delayed = self.q_history[read, self.rows], self.dq_history[read, self.rows]
        if self.interpolate_delay:
            older = (read-1) % len(self.q_history)
            q_delayed = q_delayed.lerp(self.q_history[older, self.rows], self.delay_fraction[:, None])
            dq_delayed = dq_delayed.lerp(self.dq_history[older, self.rows], self.delay_fraction[:, None])
        sample = self.tick % period == 0
        self.q.copy_(torch.where(sample[:, None], q_delayed + self.bias, self.q))
        self.dq.copy_(torch.where(sample[:, None], dq_delayed, self.dq))
        self.cursor = (self.cursor + 1) % len(self.q_history)
        self.tick += 1
        return self.q, self.dq
