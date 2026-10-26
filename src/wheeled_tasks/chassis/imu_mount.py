"""Bounded rigid IMU misalignment and a moving sensor mount, in sensor axes.

R_BS maps sensor vectors into body axes. Moving-mount gyro includes relative
angular velocity, rather than independently perturbing gravity and gyro.
The sensor contract stores angular velocity multiplied by 0.5.
"""
import math

import torch


def euler_matrix(angles):
    """Rz(yaw) Ry(pitch) Rx(roll), sensor-to-body convention."""
    r, p, y = angles.unbind(-1)
    cr, sr, cp, sp, cy, sy = r.cos(), r.sin(), p.cos(), p.sin(), y.cos(), y.sin()
    return torch.stack((cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr,
                        sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr,
                        -sp, cp*sr, cp*cr), -1).reshape(-1, 3, 3)


class ImuMount:
    def __init__(self, count, device, cfg, rng):
        for key in ("fixed_degrees", "wobble_degrees", "wobble_hz"):
            values = cfg[key]
            if len(values) != (3 if key == "fixed_degrees" else 2) or any(not math.isfinite(v) for v in values):
                raise ValueError("Invalid IMU mount configuration: " + key)
        if (any(not 0. <= v <= 10. for v in cfg["fixed_degrees"])
                or not 0. <= cfg["wobble_degrees"][0] <= cfg["wobble_degrees"][1] <= 3.
                or not 0. < cfg["wobble_hz"][0] <= cfg["wobble_hz"][1] <= 20.):
            raise ValueError("IMU mount perturbation exceeds the bounded pilot domain")
        self.cfg, self.rng, self.device = cfg, rng, device
        self.fixed = torch.zeros(count, 3, device=device)
        self.amplitude = torch.zeros(count, 2, device=device)
        self.frequency = torch.ones(count, 2, device=device)
        self.phase = torch.zeros(count, 2, device=device)
        self.start_seconds = torch.zeros(count, device=device)

    def reset(self, ids, enabled, seconds=0.):
        def uniform(bounds, columns):
            return bounds[0] + (bounds[1]-bounds[0]) * torch.rand(len(ids), columns, device=self.device, generator=self.rng)
        bounds = self.fixed.new_tensor(self.cfg["fixed_degrees"]) * (math.pi/180.)
        self.fixed[ids] = (2.*uniform([0., 1.], 3)-1.) * bounds * enabled[:, None]
        self.amplitude[ids] = uniform(self.cfg["wobble_degrees"], 2) * (math.pi/180.) * enabled[:, None]
        self.frequency[ids] = uniform(self.cfg["wobble_hz"], 2)
        self.phase[ids] = uniform([-math.pi, math.pi], 2)
        self.start_seconds[ids] = seconds

    def apply(self, sensors, seconds, ids=None):
        fixed, amplitude, frequency, phase = self.fixed, self.amplitude, self.frequency, self.phase
        start = self.start_seconds
        if ids is not None:
            fixed, amplitude, frequency, phase = (v[ids] for v in (fixed, amplitude, frequency, phase))
            start = start[ids]
        theta = phase + (2.*math.pi*(seconds-start[:, None])) * frequency
        angle = amplitude * theta.sin()
        rate = amplitude * (2.*math.pi*frequency) * theta.cos()
        a, b = angle.unbind(-1)
        da, db = rate.unbind(-1)
        zero = torch.zeros_like(a)
        # Rwiggle = Rx(a) Ry(b). Its angular velocity in fixed-mount axes is
        # [da, cos(a)*db, sin(a)*db]. Both measurements use the same rotation.
        rx = euler_matrix(torch.stack((a, zero, zero), -1))
        ry = euler_matrix(torch.stack((zero, b, zero), -1))
        rigid = euler_matrix(fixed)
        rotation = rigid @ rx @ ry
        relative = (rigid @ torch.stack((da, a.cos()*db, a.sin()*db), -1).unsqueeze(-1)).squeeze(-1)
        inverse = rotation.transpose(-1, -2)
        result = sensors.clone()
        result[:, :3] = .5 * (inverse @ (2.*sensors[:, :3] + relative).unsqueeze(-1)).squeeze(-1)
        result[:, 3:6] = (inverse @ sensors[:, 3:6].unsqueeze(-1)).squeeze(-1)
        return result

    def parameters(self):
        return dict(imu_fixed_rpy_rad=self.fixed, imu_wobble_amplitude_rad=self.amplitude,
                    imu_wobble_frequency_hz=self.frequency, imu_wobble_phase_rad=self.phase)

    def set_profile(self, lane, profile):
        """Explicit diagnostic cases; never inferred from the true body state."""
        for key, size, limit in (("fixed_degrees", 3, 10.), ("amplitude_degrees", 2, 3.),
                                 ("frequency_hz", 2, 20.), ("phase_rad", 2, math.pi)):
            value = profile[key]
            if (len(value) != size or any(not math.isfinite(v) or abs(v)>limit for v in value)
                    or key == "frequency_hz" and any(v <= 0. for v in value)
                    or key == "amplitude_degrees" and any(v < 0. for v in value)):
                raise ValueError("Invalid diagnostic IMU profile: " + key)
        self.fixed[lane] = self.fixed.new_tensor(profile["fixed_degrees"])*(math.pi/180.)
        self.amplitude[lane] = self.amplitude.new_tensor(profile["amplitude_degrees"])*(math.pi/180.)
        self.frequency[lane] = self.frequency.new_tensor(profile["frequency_hz"])
        self.phase[lane] = self.phase.new_tensor(profile["phase_rad"])
