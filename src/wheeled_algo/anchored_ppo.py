# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# Copyright (c) 2026, this project's contributors
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Visible-error-gated high-command source retention for flat continuation.

RSL-RL 5.5.1 feedforward PPO with a frozen, command-masked teacher loss.

The minibatch update follows installed RSL-RL's PPO implementation. The added
loss uses current training observations, never evaluation traces. Sampling,
log probabilities, GAE, value targets and Adam remain native RSL-RL paths.
Unsupported extensions fail explicitly instead of silently being ignored.
"""
from __future__ import annotations

from importlib.metadata import version
import math
import torch
from torch import nn
from rsl_rl.algorithms import PPO


class TeacherAnchoredPPO(PPO):
    def configure_teacher(self, teacher, *, coefficient, action_scales, min_vx, min_yaw, min_visible_wheel_error_m_s=None):
        if version("rsl-rl-lib") != "5.5.1":
            raise ValueError("The teacher update is pinned to RSL-RL 5.5.1")
        if (self.actor.is_recurrent or self.critic.is_recurrent or self.rnd or self.symmetry
                or self.is_multi_gpu or self.use_mixed_precision or self.schedule != "fixed"
                or self.actor.obs_normalization or self.actor.obs_groups != ["policy"]):
            raise ValueError("Teacher PPO supports only the frozen flat feedforward interface")
        if (not all(math.isfinite(x) for x in (coefficient, min_vx, min_yaw))
                or coefficient < 0 or min_vx <= 0 or min_yaw <= 0):
            raise ValueError("Invalid teacher loss settings")
        if (min_visible_wheel_error_m_s is not None and
                (type(min_visible_wheel_error_m_s) not in (int, float)
                 or not math.isfinite(min_visible_wheel_error_m_s) or min_visible_wheel_error_m_s <= 0)):
            raise ValueError("Visible wheel-error threshold must be finite and positive")
        scale = torch.tensor(action_scales, dtype=torch.float32, device=self.device)
        if scale.shape != (6,) or not torch.isfinite(scale).all() or not (scale > 0).all():
            raise ValueError("Teacher loss requires six finite positive action scales")
        self.teacher = teacher.to(self.device).eval().requires_grad_(False)
        self.teacher_coefficient = coefficient
        self.teacher_action_scales = scale
        self.teacher_min_vx, self.teacher_min_yaw = min_vx, min_yaw
        self.teacher_min_visible_wheel_error_m_s = min_visible_wheel_error_m_s

    def teacher_selection(self, frame):
        selected = (frame[:, 0].abs() >= self.teacher_min_vx) | (frame[:, 2].abs() >= self.teacher_min_yaw)
        if self.teacher_min_visible_wheel_error_m_s is not None:
            vx, yaw = frame[:, 0], frame[:, 2]
            expected = torch.stack((vx - .4373 * yaw / 2, -vx - .4373 * yaw / 2), -1) / .6
            wheel_error = .6 * (frame[:, 20:22] - expected).abs().amax(-1)
            selected &= wheel_error >= self.teacher_min_visible_wheel_error_m_s
        return selected

    def teacher_loss(self, observations):
        frame = observations["policy"]
        selected = self.teacher_selection(frame)
        # Multiplication by a mask avoids a GPU synchronization / dynamic
        # shape on every minibatch. It also gives exactly zero low-speed loss.
        with torch.no_grad():
            target = self.teacher(frame)
        error = self.actor.mlp(frame) - target
        weights = selected.to(error.dtype)
        count = weights.sum().clamp_min(1.)
        loss = (.5 * (error / self.teacher_action_scales).square().mean(-1) * weights).sum() / count
        rmse = ((error.square().mean(-1) * weights).sum() / count).sqrt()
        return loss, rmse, weights.mean()

    def update(self):
        if not hasattr(self, "teacher"):
            raise RuntimeError("A frozen teacher must be configured before PPO updates")
        totals = dict(value=0., surrogate=0., entropy=0., teacher=0., teacher_action_rmse=0., teacher_fraction=0.,
            approx_policy_kl=0., policy_clip_fraction=0., actor_gradient_norm=0., critic_gradient_norm=0.,
            actor_learning_rate=0., critic_learning_rate=0.)
        for batch in self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs):
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)
            self.actor(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[0], stochastic_output=True)
            log_prob = self.actor.get_output_log_prob(batch.actions)
            values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
            entropy = self.actor.output_entropy
            ratio = torch.exp(log_prob - torch.squeeze(batch.old_actions_log_prob))
            with torch.no_grad():
                log_ratio = log_prob - torch.squeeze(batch.old_actions_log_prob)
                approx_policy_kl = ((ratio - 1.) - log_ratio).mean()
                policy_clip_fraction = ((ratio - 1.).abs() > self.clip_param).float().mean()
            surrogate = -torch.squeeze(batch.advantages) * ratio
            clipped = -torch.squeeze(batch.advantages) * torch.clamp(ratio, 1. - self.clip_param, 1. + self.clip_param)
            surrogate_loss = torch.max(surrogate, clipped).mean()
            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_loss = torch.max((values - batch.returns).square(), (value_clipped - batch.returns).square()).mean()
            else:
                value_loss = (batch.returns - values).square().mean()
            anchor, rmse, fraction = self.teacher_loss(batch.observations)
            loss = (surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()
                    + self.teacher_coefficient * anchor)
            self.optimizer.zero_grad()
            loss.backward()
            actor_gradient_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            critic_gradient_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            project = getattr(getattr(self.actor, 'mlp', None), 'project_parameters_', None)
            if project is not None and any(p.requires_grad for p in self.actor.parameters()):
                project()
            values_to_log = dict(zip(totals, (value_loss, surrogate_loss, entropy.mean(), anchor, rmse, fraction,
                approx_policy_kl, policy_clip_fraction, actor_gradient_norm, critic_gradient_norm,
                self.optimizer.param_groups[0]["lr"], self.optimizer.param_groups[-1]["lr"]), strict=True))
            for key, value in values_to_log.items():
                totals[key] += value.item() if torch.is_tensor(value) else float(value)
        obs = self.storage.observations.flatten(0, 1)
        self.actor.update_normalization(obs)
        self.critic.update_normalization(obs)
        self.storage.clear()
        count = self.num_learning_epochs * self.num_mini_batches
        return {key: value / count for key, value in totals.items()}
