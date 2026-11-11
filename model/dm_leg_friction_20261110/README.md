# Left-leg moving friction, horizontal fixture, 2026-11-10

Source identification: `RMCS/.claude/report/leg_id_horizontal_gap_20261110`
(`turnfric.txt`, mode 1 common-angle steady rotation of the whole leg, hip fixed, gas
spring installed, beta about 71.5/71.8 deg, knee rotor 51-76 C).

Only the local Coulomb plateau is admitted. The 0.05 rad/s linear band is numerical
regularization, not an identified viscous/Stribeck law or static breakaway torque.
No controller compensation is added.

| motor | measured (Nm) | nominal | random range |
|---|---|---|---|
| left_hip | 2.54-3.21 | 2.70 | 2.3-3.3 |
| left_knee | 2.95-4.18 (5.3 hot) | 3.40 | 2.9-4.6 |

The right side repeats the left values under an explicit symmetry assumption; no right
leg data was collected. Beta is fixed near 72 deg, so pose and speed dependence are not
identified, and these values are a prior outside that domain. Standing bearing load may
raise the real friction; that requires the vertical/loaded datasets.

Supersedes `model/dm_leg_friction_20261031/model.json` (hip 2.02, knee 2.05 Nm), which
understated the measured plateau by +34 % hip and +66 % knee.
