"""Read-only first-episode reward payments and physical-phase accounting."""
from __future__ import annotations

import torch
from torch.nn.functional import one_hot


class FlatRewardAudit:
    def __init__(self, groups, device, policy_dt, warmup_seconds):
        self.names = list(dict.fromkeys(groups))
        self.group_ids = torch.tensor([self.names.index(g) for g in groups], device=device)
        self.dt, self.warmup_ticks = policy_dt, round(warmup_seconds / policy_dt)
        self.frames = torch.zeros(2, len(self.names), dtype=torch.int64, device=device)
        self.phases = torch.zeros(2, len(self.names), 5, dtype=torch.int64, device=device)
        self.terms, self.payments = None, None

    def observe(self, components, data, active):
        terms = tuple(sorted(components))
        if self.terms is None:
            self.terms = terms
            self.payments = torch.zeros(2, len(self.names), len(terms) + 2, dtype=torch.float64, device=active.device)
        if self.terms != terms:
            raise ValueError("Reward terms changed during a fixed replay")
        density = torch.stack([components[n] for n in terms], -1).double()
        actual = data["reward"].double()[:, None]
        paid = density * self.dt
        values = torch.cat((paid, actual, actual - paid.sum(-1, keepdim=True)), -1)
        phase = one_hot(data["physical_phase"].long(), 5)
        for index, mask in enumerate((active, active & (data["episode_ticks"] > self.warmup_ticks))):
            # Inactive lanes cannot contaminate a first-episode receipt, even
            # when their later, unobserved simulation produces nonfinite data.
            self.payments[index].index_add_(0, self.group_ids, torch.where(mask[:, None], values, 0.))
            self.frames[index].index_add_(0, self.group_ids, mask.long())
            self.phases[index].index_add_(0, self.group_ids, phase * mask[:, None])

    def report(self):
        if self.terms is None:
            raise ValueError("Cannot audit a replay without observed reward payments")
        counts, phases, payments = self.frames.cpu(), self.phases.cpu(), self.payments.cpu()
        result = {}
        for window, index in (("whole_episode", 0), ("post_warmup", 1)):
            result[window] = {}
            for group, name in enumerate(self.names):
                frames = int(counts[index, group])
                mean = payments[index, group] / max(frames, 1)
                per_step = dict(zip(self.terms, mean[:-2].tolist()))
                result[window][name] = dict(frames=frames, observed_sim_seconds=frames * self.dt,
                    component_reward_per_policy_step=per_step,
                    component_reward_per_sim_second={k: v / self.dt for k, v in per_step.items()},
                    actual_reward_per_policy_step=float(mean[-2]),
                    residual_reward_per_policy_step=float(mean[-1]),
                    physical_phase_fraction={str(p): float(phases[index, group, p]) / max(frames, 1) for p in range(5)})
        return result
