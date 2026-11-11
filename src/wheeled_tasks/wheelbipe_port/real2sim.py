"""Measured leg plant for the 200 Hz Wheelbipe port (real2sim v3, 2026-11-10).

Only the plant changes: DM velocity filtering, feedback age, actuation delay
and velocity quantization on the leg PD path, measured moving friction, the
high-beta gas-spring tail and leg armature spread. Controller law, gains and
observation layout are untouched, so exported actors stay deployable.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
CONTRACT = "contracts/wheelbipe_real2sim_v3_20261110.json"
MAX_FEEDBACK_TICKS = 2.


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path=CONTRACT, root=ROOT):
    root = Path(root)
    spec = json.loads((root/path).read_text())
    if spec.get("schema") != "wheelbipe-real2sim-v3":
        raise ValueError("Unsupported real2sim contract")
    for name in ("leg_friction", "gas_spring_plant"):
        row = spec[name]
        if digest(root/row["model_path"]) != row["model_sha256"]:
            raise ValueError(f"real2sim {name} model differs from its contract hash")
    for name, row in spec["leg_feedback"]["real_parameters"].items():
        if isinstance(row, dict):
            low, high = row["range"]
            if not (math.isfinite(low) and low <= row["nominal"] <= high and low >= 0.):
                raise ValueError("Invalid real2sim range: "+name)
    friction = spec["leg_friction"]
    for joint in ("hip", "knee"):
        low, high = friction["range_nm"][joint]
        if not 0. < low <= friction["nominal_nm"][joint] <= high <= friction["native_effort_limit_nm"]-40.:
            raise ValueError("Leg friction range exceeds the native effort limit")
    return spec


def sample(shape, nominal, low, high, amplitude, device):
    """amplitude=0 keeps nominal; 1 is uniform over [low, high]."""
    nominal = torch.as_tensor(nominal, device=device, dtype=torch.float32)
    low = torch.as_tensor(low, device=device, dtype=torch.float32)
    high = torch.as_tensor(high, device=device, dtype=torch.float32)
    u = low+(high-low)*torch.rand(shape, device=device)
    return nominal+amplitude*(u-nominal)


def friction_settings(spec):
    row = spec["leg_friction"]
    return dict(enabled=True, model_path=row["model_path"], model_sha256=row["model_sha256"])


def sample_friction(spec, count, amplitude, device):
    """(count, 4) Coulomb torques in hip, knee, hip, knee order."""
    row = spec["leg_friction"]
    order = ("hip", "knee", "hip", "knee")
    return sample((count, 4), [row["nominal_nm"][j] for j in order],
                  [row["range_nm"][j][0] for j in order], [row["range_nm"][j][1] for j in order],
                  amplitude, device)


def sample_armature_scale(spec, count, amplitude, device):
    row = spec["leg_armature"]
    bound = torch.tensor([row["hip_fraction"], row["knee_fraction"]]*2, device=device)
    return 1.+amplitude*bound*torch.empty(count, 4, device=device).uniform_(-1., 1.)


class LegFeedbackTiming:
    """Per-environment DM feedback and actuation timing on four leg axes.

    Call update() once per physics tick before the PD, actuate() on the PD
    output, and observe() at the policy boundary. All tensors are N x 4 in
    hip, knee, hip, knee order.
    """

    def __init__(self, spec, count, device, physics_dt, amplitude):
        if abs(physics_dt-spec["physics_dt_s"]) > 1e-9:
            raise ValueError("real2sim v3 timing is calibrated for 200 Hz physics only")
        p = spec["leg_feedback"]["real_parameters"]
        m = spec["leg_feedback"]["sim_mapping"]
        draw = lambda row: sample((count, 1), row["nominal"], *row["range"], amplitude, device)
        self.tau_ms, self.age_ms, self.act_ms = (draw(p[k]) for k in
            ("velocity_filter_tau_ms", "feedback_age_ms", "actuation_delay_ms"))
        tau = (self.tau_ms+m["filter_tau_offset_ms"]).clamp_min(.5)*1e-3
        self.alpha = 1.-torch.exp(-physics_dt/tau)
        self.feedback_ticks = ((self.age_ms+m["zoh_equivalent_ms"])*1e-3/physics_dt).clamp(0., MAX_FEEDBACK_TICKS)
        self.observation_ticks = (self.age_ms*1e-3/physics_dt).clamp(0., 1.)
        self.actuation_ticks = (self.act_ms*1e-3/physics_dt).clamp(0., 1.)
        self.quantum = float(p["velocity_quantum_rad_s"])
        self.filtered = torch.zeros(count, 4, device=device)
        self.history_q = torch.zeros(3, count, 4, device=device)  # index 0 newest
        self.history_v = torch.zeros_like(self.history_q)
        self.previous_torque = torch.zeros(count, 4, device=device)
        self.pending = torch.ones(count, dtype=torch.bool, device=device)

    def reset(self, env_ids):
        # State is re-seeded lazily from the first post-reset sample, after
        # every subclass has finished writing its reset pose.
        self.pending[env_ids] = True

    def quantize(self, value):
        return torch.round(value/self.quantum)*self.quantum

    @staticmethod
    def interpolate(history, ticks):
        index = ticks.floor().clamp(0., MAX_FEEDBACK_TICKS-1.)
        weight = ticks-index
        first = torch.where(index < .5, history[0], history[1])
        second = torch.where(index < .5, history[1], history[2])
        return (1.-weight)*first+weight*second

    def _seed(self, q, dq):
        ids = self.pending.nonzero().flatten()
        if len(ids):
            self.filtered[ids] = dq[ids]
            self.history_q[:, ids] = q[ids]
            self.history_v[:, ids] = dq[ids]
            self.previous_torque[ids] = 0.
            self.pending[ids] = False

    def update(self, q, dq):
        """True leg state -> (q_meas, dq_meas) used by the PD."""
        self._seed(q, dq)
        self.filtered += self.alpha*(dq-self.filtered)
        self.history_q = torch.cat((q[None], self.history_q[:-1]))
        self.history_v = torch.cat((self.filtered[None], self.history_v[:-1]))
        return (self.interpolate(self.history_q, self.feedback_ticks),
                self.quantize(self.interpolate(self.history_v, self.feedback_ticks)))

    def observe(self, dq):
        """Policy-boundary DM velocity from the current true state, uncommitted."""
        peek = self.filtered+self.alpha*(dq-self.filtered)
        w = self.observation_ticks
        value = self.quantize((1.-w)*peek+w*self.history_v[0])
        return torch.where(self.pending[:, None], self.quantize(dq), value)

    def actuate(self, torque):
        b = self.actuation_ticks
        applied = (1.-b)*torque+b*self.previous_torque
        self.previous_torque = torque.clone()
        return applied

    def summary(self):
        stats = lambda x: [float(x.min()), float(x.mean()), float(x.max())]
        return dict(velocity_filter_tau_ms=stats(self.tau_ms), feedback_age_ms=stats(self.age_ms),
                    actuation_delay_ms=stats(self.act_ms), feedback_delay_ticks=stats(self.feedback_ticks),
                    actuation_delay_ticks=stats(self.actuation_ticks), velocity_quantum_rad_s=self.quantum)
