"""Switch native domains before sampling the first action of an empty rollout.

RSL's one continuous learn call retains its global iteration count and ETA.
Its local observation still belongs to the previous environment at a domain
boundary, so replace it BEFORE native PPO records or samples that action.
"""
from __future__ import annotations

import torch


class RolloutDomainSwitch:
    def __init__(self, runner, *, completed, desired, current, rebuild):
        self.runner = runner
        self.completed, self.desired = completed, desired
        self.current, self.rebuild = current, rebuild
        self.native_act = runner.alg.act
        self.hook = self.act
        self.switches = 0
        runner.alg.act = self.hook

    def identity(self):
        alg = self.runner.alg
        return tuple(id(v) for v in (alg, alg.actor, alg.critic, alg.optimizer, alg.storage))

    def act(self, observations):
        desired = self.desired(self.completed())
        if desired != self.current():
            if self.runner.alg.storage.step != 0:
                raise RuntimeError("A domain may change only before an empty rollout")
            before, writer = self.identity(), self.runner.logger.writer
            # The caller samples under inference_mode. Construct long-lived
            # native buffers in the same ordinary mode as initial startup.
            with torch.inference_mode(False):
                self.rebuild(*desired)
            if self.identity() != before or self.runner.logger.writer is not writer:
                raise RuntimeError("Domain switch replaced the live learner or logger")
            # Discard unfinished OLD-domain episode totals, never the global
            # time/step counters used for the full-run ETA and TensorBoard.
            for name in ("cur_reward_sum", "cur_episode_length", "cur_ereward_sum", "cur_ireward_sum"):
                value = getattr(self.runner.logger, name, None)
                if value is not None:
                    value.zero_()
            observations = self.runner.env.get_observations().to(self.runner.device)
            self.switches += 1
        return self.native_act(observations)

    def close(self):
        if self.runner.alg.act is self.hook:
            self.runner.alg.act = self.native_act
