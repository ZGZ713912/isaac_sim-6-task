"""Reproducible effective wheel/ground friction and frictionless body rollers."""
import numpy as np


class TraditionalContactDomain:
    ground_coefficient = 1.
    body_roller_coefficient = 0.
    other_body_coefficient = .8
    combine_mode = "multiply"

    def __init__(self, wheel_ground_friction_range=(.2, .4), seed=42):
        band = np.asarray(wheel_ground_friction_range, dtype=float)
        if band.shape != (2,) or not np.isfinite(band).all() or not 0. <= band[0] <= band[1] <= 1.:
            raise ValueError("Expected an ordered wheel/ground friction interval in [0,1]")
        if type(seed) is not int or seed < 0:
            raise ValueError("Friction seed must be a nonnegative integer")
        self.band, self.seed = band, seed

    def sample(self, case):
        if type(case) is not int or case < 0:
            raise ValueError("Friction case must be a nonnegative integer")
        # Case-local RNG makes candidate comparisons independent of trial order,
        # early termination and wall-clock time.
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, case]))
        return float(rng.uniform(*self.band))

    def body_coefficient(self, name, wheel_coefficient):
        if name in ("L_link3", "R_link3"):
            return wheel_coefficient
        if name == "base_link":
            return self.body_roller_coefficient
        return self.other_body_coefficient

    def summary(self):
        return dict(wheel_ground_friction_range=self.band.tolist(), seed=self.seed,
            distribution="uniform per episode reset; same sample for left/right wheels",
            candidate_comparison="same case index uses the same draw for every candidate",
            ground_coefficient=self.ground_coefficient, combine_mode=self.combine_mode,
            body_roller_coefficient=self.body_roller_coefficient,
            body_roller_scope="all base_link collision hulls including bottom, front and rear",
            other_body_coefficient=self.other_body_coefficient,
            static_dynamic_relation="equal; effective coefficient is robot times ground")
