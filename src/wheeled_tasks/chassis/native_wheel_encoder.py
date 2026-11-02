"""Continuous output-shaft encoder counts from PhysX's4-pi joint coordinate.

Only the simulation feedback producer performs this lifting, at every physics
step. The policy observer still requires continuous driver angles and retains
its jump/finiteness checks. No position, velocity or torque is written to PhysX.
"""
import math

import torch


def encoder_contract():
    return dict(schema="wheel-leg-native-wheel-encoder-v1", period_rad=4.*math.pi,
                sampling="every_physics_step_before_sensor_transport",
                output="continuous_output_shaft_radians_float64", reset="selected_episode_lanes")


class NativeWheelEncoder:
    def __init__(self, positions, contract):
        if contract != encoder_contract() or positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError("Native wheel encoder needs its explicit4-pi physical-rate contract")
        self.previous = positions.to(torch.float64).clone()
        self.angle = self.previous.clone()
        self.wraps = torch.zeros_like(positions, dtype=torch.long)

    def reset(self, ids, positions):
        self.previous[ids] = positions.to(torch.float64)
        self.angle[ids] = positions.to(torch.float64)
        self.wraps[ids] = 0

    def update(self, positions):
        value = positions.to(torch.float64)
        delta = value-self.previous
        # PhysX revolute coordinates wrap at +/-2pi. A full2pi encoder jump is
        # deliberately NOT repaired; the policy's source guard must reject it.
        turns = torch.round(delta/(4.*math.pi))
        self.angle += delta-turns*(4.*math.pi)
        self.wraps += turns.abs().to(torch.long)
        self.previous.copy_(value)
        return self.angle
