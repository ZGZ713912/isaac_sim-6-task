"""Wheel centers and folding direction from four motor encoders alone.

Inputs use the calibrated model angles, in policy order: left hip/auxiliary,
right hip/auxiliary. No body pose, passive encoder or simulator FK is read.
"""
import torch

from .rolling_velocity import RollingBaseVelocity


class EncoderLegGeometry(RollingBaseVelocity):
    def __init__(self, table, relative_margin_rad=.001):
        super().__init__(table)
        if not 0. <= relative_margin_rad <= .001:
            raise ValueError("Encoder geometry margin must be within0..1mrad")
        self.relative_margin_rad = relative_margin_rad

    def forward(self, positions, velocities):
        if positions.shape != velocities.shape or positions.ndim != 2 or positions.shape[1] != 4:
            raise ValueError("Need four aligned hip/auxiliary motor angles and velocities")
        centers, rates, knees, knee_rates, lengths, length_rates, validity = [], [], [], [], [], [], []
        for side in range(2):
            i = 2 * side
            get = lambda key: getattr(self, f"side{side}_{key}")
            knots = get("delta_rad")
            difference = positions[:, i+1] - positions[:, i]
            delta = torch.atan2(difference.sin(), difference.cos())
            valid = (torch.isfinite(positions[:, i:i+2]).all(-1)
                     & torch.isfinite(velocities[:, i:i+2]).all(-1)
                     & (delta >= knots[0]-self.relative_margin_rad)
                     & (delta <= knots[-1]+self.relative_margin_rad))
            upper = torch.searchsorted(knots, delta.contiguous()).clamp(1, len(knots)-1)
            lower = upper-1
            width = knots[upper]-knots[lower]
            blend = ((delta-knots[lower])/width).clamp(0., 1.)
            points = get("wheel_center_b_m")
            point = points[lower] + blend[:, None]*(points[upper]-points[lower])
            derivative = (points[upper]-points[lower])/width[:, None]
            axis, origin = get("hip_axis_b"), get("hip_origin_b_m")
            angle = positions[:, i]-get("hip_reference_rad")
            cosine, sine = angle.cos()[:, None], angle.sin()[:, None]

            def rotate(value):
                return (cosine*value + sine*torch.linalg.cross(axis.expand_as(value), value)
                        + (1.-cosine)*axis*(axis*value).sum(-1, keepdim=True))

            relative = rotate(point-origin)
            delta_rate = velocities[:, i+1]-velocities[:, i]
            velocity = (torch.linalg.cross(axis.expand_as(relative), relative)*velocities[:, i:i+1]
                        + rotate(derivative)*delta_rate[:, None])
            beta = get("passive_inner_deg")
            knee = beta[lower] + blend*(beta[upper]-beta[lower])
            knee_rate = (beta[upper]-beta[lower])/width*delta_rate
            length = relative.norm(dim=-1)
            centers.append(relative+origin)
            rates.append(velocity)
            knees.append(knee)
            knee_rates.append(knee_rate)
            lengths.append(length)
            length_rates.append((relative*velocity).sum(-1)/length.clamp_min(1e-9))
            validity.append(valid)
        # Invalid readings remain explicitly marked; a caller must not treat
        # clamped interpolation outside the closed-chain branch as valid FK.
        return dict(wheel_body_m=torch.stack(centers, 1), wheel_body_velocity_m_s=torch.stack(rates, 1),
            knee_deg=torch.stack(knees, 1), knee_rate_deg_s=torch.stack(knee_rates, 1),
            leg_length_m=torch.stack(lengths, 1), leg_length_rate_m_s=torch.stack(length_rates, 1),
            valid=torch.stack(validity, 1))
