# Leg moving friction, left 2026-11-10 + right 2026-11-11

Left values are unchanged from `dm_leg_friction_20261110`. The right leg is now measured
(`RMCS/.claude/report/leg_id_right_20261111`, `turnfric.txt`, `gas.txt`) instead of mirrored.

| motor | common rotation (Nm) | knee open/close (Nm) | plateau | rotor temp |
|---|---|---|---|---|
| left_hip | 2.40-3.21 | - | 2.70 | 22-72 C |
| left_knee | 2.95-4.18 | 2.87-5.10 | 3.40 | 22-95 C |
| right_hip | 1.73-1.96 | - | 1.85 | 46-91 C |
| right_knee | 1.48-1.95 | 3.27-4.05 | 1.75 | 48-92 C |

The right leg was measured 20-30 C hotter, which lowers moving friction; its random ranges
reach upward for colder operation. Only the local Coulomb plateau is admitted; the 0.05 rad/s
band is numerical regularization.
