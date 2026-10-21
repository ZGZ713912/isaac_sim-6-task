"""Diagnostic command variant: keep a straight heading through a zero request.

This wraps the frozen reference for selected replay lanes only. It is not a
training contract, an exported actor, or a transcription of current RMCS.
"""
import torch


class StopHeadingMemory:
    def __init__(self, reference, selected):
        if selected.shape != reference.heading.shape or selected.dtype != torch.bool:
            raise ValueError('Heading-memory lanes require one boolean per environment')
        cfg = reference.cfg
        if not cfg['heading_hold_enabled'] or not cfg['heading_hold_gain'] > 0:
            raise ValueError('Heading memory requires the existing heading controller')
        self.reference = reference
        self.selected = selected.clone()
        self.base_update = reference.update
        self.held_lane_ticks = 0

    def update(self, requested, yaw, ticks):
        ref = self.reference
        # Compute from the pre-update state. The original update owns validation,
        # reset state, captures, height limits and repeated-tick handling.
        retain = (self.selected & (ticks != ref.last_tick) & ref.heading_valid
                  & (requested[:, 0] == 0.) & (requested[:, 1] == 0.))
        result = self.base_update(requested, yaw, ticks)
        ids = retain.nonzero(as_tuple=True)[0]
        error = ref.heading[ids] - yaw[ids]
        error = torch.atan2(error.sin(), error.cos())
        cfg = ref.cfg
        cap = min(cfg['heading_hold_yaw_rate_max_rad_s'], cfg['yaw_rate_max_rad_s'],
                  2 * cfg['wheel_speed_limit_rad_s'] * cfg['wheel_radius_m'] / cfg['wheel_track_m'])
        result[ids, 1] = (cfg['heading_hold_gain'] * error).clamp(-cap, cap)
        ref.heading_valid[ids] = True
        self.held_lane_ticks += len(ids)
        return result
