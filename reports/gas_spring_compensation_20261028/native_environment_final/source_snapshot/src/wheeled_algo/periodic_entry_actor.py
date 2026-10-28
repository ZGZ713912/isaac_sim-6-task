"""Periodic continuation of an existing learned entry NN near encoder cuts."""
import math

import torch

from .gap_entry_handoff import handoff_entry_gate
from .gap_entry_preservation import EntryPreservingGapActor
from .getup_symmetry import reflect_actions, reflect_observation
from .paired_recovery import wrap
from .quiet_output_actor import QuietOutputGetupActor
from .recovery_coordinates import smooth_recovery_legs

VERSION = "periodic-chart-v1"
SEAM_WIDTH_RAD = .35


def periodic_entry_logits(head, frame, feature_scale):
    """Average equivalent local charts only within .35 rad of +/-pi.

    The 16 charts cover four existing measured encoder angles. Their weights
    form a partition of unity, meet at equal weights on each periodic cut,
    and vanish smoothly outside the seam region. There are no learned additions,
    timers, hidden states, privileged observations, or pose targets here.
    """
    angles = frame[:, 10:14]
    phase = ((angles.abs()-(math.pi-SEAM_WIDTH_RAD))/SEAM_WIDTH_RAD).clamp(0., 1.)
    weights = .5*phase.square()*(3.-2.*phase)
    bits = frame.new_tensor([[float(bool(mask & (1 << j))) for j in range(4)] for mask in range(16)])
    alternate = angles-2.*math.pi*angles.sign()
    chart_angles = angles[:, None, :]+bits[None, :, :]*(alternate-angles)[:, None, :]
    expanded = frame[:, None, :].expand(-1, 16, -1)
    charts = torch.cat((expanded[:, :, :10], chart_angles, expanded[:, :, 14:]), -1)
    values = head(torch.tanh(charts.reshape(-1, frame.shape[-1])*feature_scale)).reshape(-1, 16, 6)
    probability = torch.where(bits[None, :, :] > 0., weights[:, None, :], 1.-weights[:, None, :]).prod(-1)
    # A separate original call and explicit selection preserve exact old values
    # outside the seam, independent of GEMM batch-size rounding.
    base = head(torch.tanh(frame*feature_scale))
    blended = base+(probability[:, :, None]*(values-base[:, None, :])).sum(1)
    return torch.where(weights.amax(-1, keepdim=True) > 0., blended, base)


def periodic_canonical_entry_logits(head, frame, feature_scale):
    weight = ((frame[:, 8:9]/.15+1.)*.5).clamp(0., 1.)
    direct = periodic_entry_logits(head, frame, feature_scale)
    reflected = reflect_actions(periodic_entry_logits(head, reflect_observation(frame), feature_scale))
    return direct+weight*(reflected-direct)


class PeriodicEntryGapActor(EntryPreservingGapActor):
    """Same actor and weights; periodic evaluation of its existing entry head."""
    def forward(self, obs):
        old = QuietOutputGetupActor.forward(self, obs)
        gate = handoff_entry_gate(obs)
        frame = torch.cat((obs[:, :14], torch.zeros_like(obs[:, 14:16]), obs[:, 16:]), -1)
        raw = (periodic_canonical_entry_logits(self.entry_recovery, frame, self.feature_scale)
               if self.canonical_recovery else periodic_entry_logits(self.entry_recovery, frame, self.feature_scale))
        legs = smooth_recovery_legs(raw[:, :4], obs[:, 22:26], obs[:, 10:14],
                                   self.recovery_leg_step, self.recovery_knee_center,
                                   previous_common_version=self.recovery_previous_common_version)
        delta = wrap(legs*.25-obs[:, 10:14])
        legs = (obs[:, 10:14]+.12*torch.tanh(delta/.12))/.25
        fallen = torch.cat((legs, 9.*torch.tanh(raw[:, 4:6])), -1)
        blended = old+gate*(fallen-old)
        result = torch.where(gate > 0., blended, old)
        result = torch.maximum(torch.minimum(result, self.action_bound), -self.action_bound)
        delta = wrap(result[:, :4]*.25-obs[:, 10:14])
        magnitude = delta.abs()
        shaped = delta.sign()*(.06+.06*torch.tanh((magnitude-.06)/.06))
        upright = ((-obs[:, 9:10]-.5)/.2).clamp(0., 1.)
        blended = delta+upright*(shaped-delta)
        changed = (upright > 0.) & (magnitude > .06)
        legs = torch.where(changed, (obs[:, 10:14]+blended)/.25, result[:, :4])
        legs = torch.maximum(torch.minimum(legs, self.action_bound[:4]), -self.action_bound[:4])
        legs = self.finalize_leg_pairs(legs)
        return torch.cat((legs, result[:, 4:]), -1)


def exact_state_equal(a, b):
    """Exact nested checkpoint equality, including the saved RNG arrays."""
    import numpy as np
    if type(a) is not type(b):
        return False
    if isinstance(a, torch.Tensor):
        return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return a.dtype == b.dtype and np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(exact_state_equal(v, b[k]) for k, v in a.items())
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(exact_state_equal(x, y) for x, y in zip(a, b, strict=True))
    return a == b


def validate_periodic_entry_checkpoint(state, source):
    """The zero-update probe may change exactly one policy function setting."""
    from copy import deepcopy
    info, prior = state['infos'], source['infos']
    record = info.get('entry_encoder_migration', {})
    if (record.get('schema') != 'periodic-entry-migration-v1'
            or record.get('new_PPO') != 0
            or record.get('learning_clock') != info['completed_updates']
            or info['completed_updates'] != prior['completed_updates']
            or prior['settings'].get('gap_entry_encoder_version') is not None):
        raise ValueError('Periodic entry requires a zero-update migration from an original encoder head')
    if not exact_state_equal({k: v for k, v in state.items() if k != 'infos'},
                             {k: v for k, v in source.items() if k != 'infos'}):
        raise ValueError('Periodic entry migration changed weights, optimizer or saved state')
    expected = deepcopy(prior)
    expected['settings']['gap_entry_encoder_version'] = VERSION
    expected['source_sha256']['src/wheeled_algo/periodic_entry_actor.py'] = info['source_sha256'].get(
        'src/wheeled_algo/periodic_entry_actor.py')
    expected['entry_encoder_migration'] = record
    expected['hardware_ready'] = False
    if not exact_state_equal(info, expected):
        raise ValueError('Periodic entry migration changed unrelated checkpoint metadata')
