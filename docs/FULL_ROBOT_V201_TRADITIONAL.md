# Full v2.0.1 traditional-control research plant

This independent worktree adds a native plant for the RMCS traditional-control
tuning project. Existing RL environments and training launchers are unchanged.

## Asset and provenance

- Source: `model/wheel_leg_v2.0.1/轮腿整车URDFv2.0.1/urdf/轮腿整车URDFv2.0.1.urdf`.
  SHA256: `da8136d5654de4e2b718e8213ba436159720f0b9adc462df1a0756cd7854f037`.
- Generated bundle: `model/full_robot_v201_traditional_v1`; every generated file
  is covered by `manifest.json`. Original CAD masses, COMs and inertias remain;
  21 bodies total 21.44174 kg, including both gimbal bodies.
- Six external loop joints and two physical spring sliders close the mechanism.
  Loop pin topology is transported from the reviewed v7 datum only after each
  zero-pose anchor agrees with v2.0.1 within 0.1 mm. Old mass and qualification
  results are not transported.
- Non-wheel collision meshes use convex hulls, wheel collision is cylindrical,
  and self collision is disabled. Contact material is an explicit research input.
- The 15.62 identified wheel response supplies independent reflected inertia,
  passive loss and effective delay. Gas-spring force knots reuse the previous
  measured axial-force curve, linear interpolation and endpoint hold, with
  assumed left/right symmetry. Neither is a new full-vehicle identification.

## Native plant and shared controller

`src/wheeled_tasks/chassis/traditional_plant.py` creates one free-body PhysX
articulation after `AppLauncher`. `step()` accepts six canonical motor torques and
advances exactly 1 ms of simulated time. An optional physical integration step
divides 1 ms into integer substeps; the six motor commands remain held while the
physical springs and identified actuator response update at each substep.

The ground is a native infinite `UsdGeom.Plane`; its display quad has no collision.
Contacts are read from the PhysX contact view with contact processing enabled.
No target-pose teleportation occurs during stepping. Forward kinematics is used
for diagnostics only; it does not overwrite the native bodies.

The controller lives in RMCS, loaded through the same C ABI by the external
`wheelleg_controller_tuning` runner. Its packet contains encoders and synthetic
IMU only. Native velocity, contact forces and loop residuals are evaluation
signals. The two gimbal axes have a separate explicit zero-reference PD hold;
they remain dynamic and are not six-axis chassis controller outputs.

`traditional_model.py` generates equilibrium postures, constrained virtual-work
gravity/spring feedforward and a height-indexed engineering DARE seed. The
rolling-pendulum seed is not an identified full-articulation stability proof.

## Validation and limits

Seven CPU geometry/virtual-work tests pass with:

```sh
PYTHONPATH=src /home/noir/Documents/workspace/wheelleg_controller_tuning/.venv/bin/python \
  -m pytest tests/test_full_robot_v201_traditional.py -q
```

Native tests and parameter searches run from the tuning repository. Their JSON
receipts record model/table/code/library hashes, settings, simulation ticks and
wall time; NPZ retains each millisecond and TensorBoard events derive from that
trace. No local TensorBoard server is launched. Current results and remaining
work are recorded in the tuning repository's `docs/V201_IMPLEMENTATION_STATUS.md`.

Mass variation currently changes native masses, not inertia tensors, and must be
reported as that intervention. Leg motor inertia/delay, electrical-loss constants,
gimbal control, tire contact, CAD/API calibration and terrain remain research
limitations. An initialized standing run is separate from self-righting, jumping,
robustness, hardware timing and robot validation. No new PPO training or hardware
qualification is supplied by these files.
