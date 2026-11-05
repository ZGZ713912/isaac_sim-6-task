"""Explicit zero-PPO paired-shaft rollover trajectory, in physical leg coordinates."""
from dataclasses import dataclass
import math


MOTOR_SIGNS = (1., 1., -1., -1.)  # LH, LK, RH, RK; mirrored CAD axes.


@dataclass
class Sweep:
    direction: int
    speed: float = 4.
    profile: str = "direct"
    prepare_seconds: float = .5
    acceleration: float = 20.
    maximum_angle: float = 2.*math.pi
    angle: float = 0.
    velocity: float = 0.
    elapsed: float = 0.
    braking: bool = False
    captured: bool = False

    def __post_init__(self):
        if self.direction not in (-1, 1) or self.profile not in ("direct", "braked"):
            raise ValueError("Unknown physical direction or sweep profile")
        if not all(math.isfinite(v) and v > 0 for v in
                   (self.speed, self.prepare_seconds, self.acceleration, self.maximum_angle)):
            raise ValueError("Sweep bounds must be finite and positive")

    def step(self, dt, tilt_deg):
        if not math.isfinite(dt) or not 0 < dt <= .02 or not math.isfinite(tilt_deg):
            raise ValueError("Invalid controller sample")
        self.elapsed += dt
        if self.elapsed <= self.prepare_seconds + 1e-12:
            return self.angle
        # The milestone only stops the scripted sweep. It is not task completion.
        self.captured |= tilt_deg < 15.
        self.braking |= self.profile == "braked" and tilt_deg < 75.
        goal = 0. if self.captured else (.5 if self.braking else self.speed)
        previous = self.velocity
        change = max(-self.acceleration*dt, min(self.acceleration*dt, goal-previous))
        self.velocity = previous+change
        self.angle = min(self.maximum_angle, self.angle + (previous+self.velocity)*.5*dt)
        if self.angle >= self.maximum_angle:
            self.velocity = 0.
        return self.angle

    def targets(self, initial):
        if len(initial) != 4:
            raise ValueError("Need LH,LK,RH,RK encoder-radian targets")
        return [q+s*self.direction*self.angle for q, s in zip(initial, MOTOR_SIGNS)]
