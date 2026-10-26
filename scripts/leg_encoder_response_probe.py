"""Read-only isolation of the four leg encoders' neural target response.

Retain current posture, positions, history and all native wheel outputs. This
diagnostic is never a training policy, action filter or deployment candidate.
"""
import math

import torch


def validate_scale(scale):
    if (isinstance(scale, bool) or not isinstance(scale, (int, float))
            or not math.isfinite(scale) or scale not in (0., .5, 1.)):
        raise ValueError('Declare a leg encoder response scale of0,0.5or1')
    return float(scale)


def responses(actor, observation, scale):
    scale = validate_scale(scale)
    if actor.training or actor.is_recurrent:
        raise ValueError('Require the frozen feed-forward inference actor')
    frame = observation['policy']
    if frame.ndim != 2 or frame.shape[1] != 35 or not torch.isfinite(frame).all():
        raise ValueError('Require finite original35D observations')
    selected = torch.isclose(frame[:, 3], frame.new_tensor(.305 * 5.), rtol=0., atol=1e-6)
    native = actor(observation)
    if native.shape != (len(frame), 6) or not torch.isfinite(native).all():
        raise ValueError('Require a finite six-action native actor result')
    if scale == 1.:
        return native, native, native, selected
    neutral = observation.clone()
    neutral['policy'][selected, 16:20] = 0.
    anchor = actor(neutral)
    if anchor.shape != native.shape or not torch.isfinite(anchor).all():
        raise ValueError('Require a finite neutral-leg-encoder actor result')
    output = native.clone()
    output[selected, :4] = anchor[selected, :4] + scale * (native[selected, :4] - anchor[selected, :4])
    if not torch.equal(output[:, 4:], native[:, 4:]) or not torch.equal(output[~selected], native[~selected]):
        raise RuntimeError('Leg encoder diagnosis changed wheels or other heights')
    return output, native, anchor, selected


class LegEncoderResponseProbe(torch.nn.Module):
    def __init__(self, actor, scale):
        super().__init__()
        if actor.training or actor.is_recurrent:
            raise ValueError('Require the frozen feed-forward inference actor')
        self.actor, self.scale = actor, validate_scale(scale)
        self.selected = None
        self.calls = 0
        self.native_actions, self.neutral_legs = [], []
        self.eval()

    def forward(self, observation):
        if len(observation['policy']) < 21:
            raise ValueError('Require the original21 standing cases')
        output, native, anchor, selected = responses(self.actor, observation, self.scale)
        if self.calls == 0:
            if selected[:21].nonzero().flatten().tolist() != list(range(7, 14)):
                raise ValueError('Expected the original seven middle-height cases')
            if not torch.equal(output, native):
                raise ValueError('First action must remain native at the zero-velocity reset')
            self.selected = selected.detach().clone()
        elif selected.shape != self.selected.shape or not torch.equal(selected, self.selected):
            raise ValueError('Fixed-case height population changed')
        self.native_actions.append(native[:21].detach().cpu().clone())
        self.neutral_legs.append(anchor[:21, :4].detach().cpu().clone())
        self.calls += 1
        return output

    def finish(self, output, writer, completed, digest):
        from pathlib import Path
        import numpy as np

        if self.calls != 1600 or len(self.native_actions) != self.calls:
            raise ValueError('Require every original8s/200Hz actor decision')
        path = Path(output) / 'unmodified_leg_encoder_actions.npz'
        if path.exists():
            raise ValueError('Preserve the earlier encoder diagnostic')
        np.savez_compressed(path, native_action_policy=torch.stack(self.native_actions).numpy(),
            neutral_leg_action_policy=torch.stack(self.neutral_legs).numpy(),
            selected=self.selected[:21].cpu().numpy())
        metrics = dict(actor_calls=self.calls, selected_original_cases=int(self.selected[:21].sum()),
            response_scale=self.scale, native_wheel_output_max_error=0., first_action_native=1.)
        for name, value in metrics.items():
            writer.add_scalar('LegEncoderProbe/' + name, value, completed)
        writer.flush()
        return dict(schema='scratch-leg-encoder-response-counterfactual-v1',
            actor_calls=self.calls, native_actor_calls_per_decision=1 + int(self.scale != 1.),
            response_scale=self.scale, selected_height_m=.305, selected_case_indices=list(range(7, 14)),
            neutral_features=[16, 17, 18, 19], leg_policy_columns=[0, 1, 2, 3], wheel_policy_columns=[4, 5],
            formula='legs=F(neutral_leg_velocity)+scale*(F(current)-F(neutral_leg_velocity)); wheels=F(current)',
            retained_feedback='current IMU, gravity, positions, previous action, commands; native current wheel outputs',
            original_actions=dict(path=str(path), sha256=digest(path)), first_action_unchanged=True,
            other_heights_policy_unchanged=True, new_PPO=0, optimizer_constructed=False,
            learned_policy=False, adoption_by_training=False, export_forbidden=True, behavior_qualification=False,
            scope='Read-only isolation of direct neural leg-velocity response. No target hold, physics, PD, reset, '
                  'mechanical protection or strict100% support acceptance changes; no policy qualification.')
