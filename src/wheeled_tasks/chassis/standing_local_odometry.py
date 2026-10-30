"""Local standing odometry from deployable encoders/IMU, with no control output.

Wheel angles are already continuous OUTPUT-SHAFT radians from the feedback
driver. Never unwrap motor single-turn packets at the actor's lower rate.
Subtract a per-request origin before converting a long-lived double angle to
the policy's float32 sensor stream. Geometry and filtering see that same noisy,
delayed stream as the other policy sensors; simulator position is never read.
"""
from copy import deepcopy
import json
import math

import torch

from wheeled_algo.encoder_leg_geometry import EncoderLegGeometry
from wheeled_algo.rolling_velocity import RollingBaseVelocity
from wheeled_algo.standing_displacement import WheelAngleCoordinate
from .standing_geometry107 import EXTENDED_SHA256, table_path
from .standing_sensor_uncertainty import BoundedSensorCoordinate


def observation_contract():
    return dict(schema="wheel-leg-local-standing-odometry-v1", actor_dim=35,
        displacement_slots=[14, 15], validity_slot=31, scale_per_m=20., limit_m=1.,
        filter_seconds=.05, capture_seconds=.2, command_threshold=[.001, .001, .001],
        capture_gyro_max_rad_s=.5, capture_leg_speed_max_rad_s=3.,
        capture_wheel_speed_max_rad_s=2.5, minimum_encoder_stance_height_m=.20,
        upright_gravity_z_max=-math.cos(math.radians(15.)), severe_gravity_z=-.5,
        wheel_increment_margin_rad=.02, wheel_increment_speed_allowance=3.,
        geometry_table_sha256=EXTENDED_SHA256,
        source="same_policy_encoder_IMU_stream_after_sensor_delay_and_noise",
        wheel_input="driver_continuous_output_shaft_angles; subtract_float64_request_origin_before_float32",
        reference="capture_once_after_contiguous_sensor_standing; preserve_during_height_change_and_drift",
        reset="episode_or_disabled_or_motion_request_or_severe_tilt",
        invalid="zero_features_and_validity; retain_reference; wheel_jump_fault_until_next_request",
        control_outputs=False, privileged_position_or_contact=False)


def validate_contract(contract):
    if contract != observation_contract():
        raise ValueError("Local standing odometry requires its complete versioned sensor contract")
    return deepcopy(contract)


class LocalStandingOdometry:
    def __init__(self, count, device, dt, contract):
        self.cfg = validate_contract(contract)
        if not math.isfinite(dt) or dt <= 0.:
            raise ValueError("Local standing odometry needs a positive policy period")
        self.dt = dt
        table = json.loads(table_path().read_text())
        coordinate = WheelAngleCoordinate()
        coordinate.geometry = RollingBaseVelocity(table)
        self.coordinate = BoundedSensorCoordinate(coordinate).to(device).eval()
        self.legs = EncoderLegGeometry(table).to(device).eval()
        self.source_origin = torch.zeros(count, 2, dtype=torch.float64, device=device)
        self.source_active = torch.zeros(count, dtype=torch.bool, device=device)
        self.source_fault = torch.zeros_like(self.source_active)
        self.source_last = self.source_origin.clone()
        self.source_speed = torch.zeros(count, 2, device=device)
        self.source_tick = torch.full((count,), -1, dtype=torch.long, device=device)
        self.anchor = torch.zeros(count, 2, device=device)
        self.filtered = torch.zeros_like(self.anchor)
        self.reference_active = torch.zeros_like(self.source_active)
        self.capture_age = torch.zeros(count, device=device)
        self.last_tick = self.source_tick.clone()
        self.displacement = torch.zeros_like(self.anchor)
        self.valid = torch.zeros_like(self.source_active)

    def reset(self, ids):
        for field in ("source_origin", "source_last", "source_speed", "anchor", "filtered",
                      "capture_age", "displacement"):
            getattr(self, field)[ids] = 0.
        for field in ("source_active", "source_fault", "reference_active", "valid"):
            getattr(self, field)[ids] = False
        self.source_tick[ids] = self.last_tick[ids] = -1

    def requested(self, frame, enabled):
        command = frame[:, :3]
        return (enabled & torch.isfinite(command).all(-1)
                & (command.abs() < command.new_tensor(self.cfg["command_threshold"])).all(-1))

    def local_wheel_angles(self, frame, continuous_wheels, enabled, tick):
        """Prepare sensor positions BEFORE the shared domain delay/noise draw."""
        requested = self.requested(frame, enabled)
        wheel = continuous_wheels.to(torch.float64)
        speed = frame[:, 20:22] * 10.
        finite = torch.isfinite(wheel).all(-1) & torch.isfinite(speed).all(-1)
        start = requested & ~self.source_active & finite
        self.source_origin[start] = wheel[start]
        self.source_fault[~requested] = False
        fresh = requested & self.source_active & (self.source_tick != tick)
        seconds = (tick-self.source_tick).clamp_min(1).to(speed.dtype)*self.dt
        bound = (self.cfg["wheel_increment_margin_rad"] + self.cfg["wheel_increment_speed_allowance"]
                 * seconds[:, None] * torch.maximum(speed.abs(), self.source_speed.abs()))
        jump = ((wheel-self.source_last).abs() > bound).any(-1)
        self.source_fault |= requested & (~finite | (fresh & jump))
        self.source_active = requested & (self.source_active | start)
        save = requested & finite & (self.source_tick != tick)
        self.source_last[save] = wheel[save]
        self.source_speed[save] = speed[save]
        self.source_tick[save] = tick
        usable = self.source_active & ~self.source_fault & finite
        local = torch.where(usable[:, None], wheel-self.source_origin, 0.)
        return local.to(frame.dtype)

    def update(self, frame, nominal_policy_legs, enabled, tick):
        """Replace only two reserved position slots and one validity slot."""
        requested = self.requested(frame, enabled) & self.source_active
        positions = torch.cat((frame[:, 10:14]+nominal_policy_legs, frame[:, 14:16]), -1)
        gravity, gyro = frame[:, 7:10], frame[:, 4:7]*2.
        velocities = frame[:, 16:22]*10.
        norm = gravity.norm(dim=-1)
        imu_valid = (torch.isfinite(gravity).all(-1) & torch.isfinite(gyro).all(-1)
                     & (norm > .5) & (norm < 1.5))
        gz = gravity[:, 2]/norm.clamp_min(1e-9)
        coordinate, coordinate_valid = self.coordinate(positions, gravity)
        leg = self.legs(positions[:, :4], velocities[:, :4])
        # Encoder-only support geometry excludes the belly-down folded stance.
        # This is not a simulator height/contact or recovery-completion flag.
        stance_height = -leg["wheel_body_m"][:, :, 2] + .06
        standing = (imu_valid & (gz < self.cfg["upright_gravity_z_max"])
                    & leg["valid"].all(-1)
                    & (stance_height > self.cfg["minimum_encoder_stance_height_m"]).all(-1))
        valid = (standing & coordinate_valid[:, 0] & ~self.source_fault
                 & torch.isfinite(velocities).all(-1))
        eligible = (requested & valid & (gyro.norm(dim=-1) < self.cfg["capture_gyro_max_rad_s"])
                    & (velocities[:, :4].abs().amax(-1) < self.cfg["capture_leg_speed_max_rad_s"])
                    & (velocities[:, 4:].abs().amax(-1) < self.cfg["capture_wheel_speed_max_rad_s"]))
        fresh = self.last_tick != tick
        reset = ~requested | (imu_valid & (gz > self.cfg["severe_gravity_z"]))
        self.reference_active[reset] = False
        self.capture_age[reset] = 0.
        self.filtered[reset] = 0.
        waiting = fresh & ~self.reference_active & ~reset
        self.capture_age[waiting] = torch.where(eligible[waiting], self.capture_age[waiting]+self.dt, 0.)
        capture = (waiting & eligible
                   & (self.capture_age+1e-7 >= self.cfg["capture_seconds"]))
        self.anchor[capture] = coordinate[capture]
        self.reference_active[capture] = True
        self.filtered[capture] = 0.
        use = requested & valid & self.reference_active & ~reset
        advance = use & fresh
        delta = (coordinate-self.anchor).clamp(-self.cfg["limit_m"], self.cfg["limit_m"])
        alpha = self.dt/(self.cfg["filter_seconds"]+self.dt)
        self.filtered[advance] += alpha*(delta[advance]-self.filtered[advance])
        self.displacement = torch.where(use[:, None], self.filtered, 0.)
        self.valid = use
        self.last_tick[fresh] = tick
        result = frame.clone()
        result[:, 14:16] = self.cfg["scale_per_m"]*self.displacement
        result[:, self.cfg["validity_slot"]] = use.to(frame.dtype)
        return result

    def metrics(self):
        return {"StandingOdometry/valid_fraction": self.valid.float().mean(),
                "StandingOdometry/reference_fraction": self.reference_active.float().mean(),
                "StandingOdometry/wheel_fault_fraction": self.source_fault.float().mean(),
                "StandingOdometry/displacement_abs_m": self.displacement.abs().mean(),
                "StandingOdometry/displacement_peak_m": self.displacement.abs().max()}
