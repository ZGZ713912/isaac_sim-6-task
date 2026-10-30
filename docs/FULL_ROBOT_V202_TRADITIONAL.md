# v2.0.2 traditional-control plant

The active traditional-control pipeline uses the supplied
`model/wheel_leg_v2.0.2/isaac/robot.usda` directly. Its manifest and all dependencies
are authenticated before pose generation or native simulation. The copied source
package is sealed by `model/wheel_leg_v2.0.2/source_snapshot.json`; the original
workspace remains untouched. The worktree branch is `dev/wheelleg-v202-traditional`.

The base frame is already X forward, Y left, Z up. No further root rotation,
mesh rotation, inertia rewrite, pin transport, or knee-limit regeneration is
performed. The 21 original bodies retain 21.44174 kg total mass. The supplied
asset has 18 moving tree joints, six external spherical loop joints, two physical
spring sliders, and two fixed-zero gimbal joints. Gimbal mass is retained; this
fixture is the supplied six-motor chassis variant.

`traditional_asset.py` validates model identity, frame, manifest, spring/encoder
bindings and wheel CAD inertia. `full_robot_asset.py` now supplies only generic
CPU geometry. `traditional_model.py` regenerates equilibria, constrained virtual
work and engineering DARE seeds using v2.0.2 transforms and side-specific knee
zeros. The resulting seven-height contract is
`contracts/full_robot_v202_traditional_postures_v1.json`.

`traditional_plant.py` loads native PhysX after AppLauncher. One controller tick
advances 1 ms simulation time, optionally split into physical substeps. Native
state is reset once per episode; no pose writes occur during stepping. The
shared C++ controller is hosted by the tuning repository, not reimplemented here.
Ground-truth velocity, contact forces, measured leg lengths and mechanical-limit
excesses are evaluator signals. Only encoders and synthetic IMU enter control.

The plant loads v2.0.2's bound wheel model so reflected armature is applied once.
It invokes the existing physical gas-force component with the bound v2.0.2 curve
and declared C1 endpoint trend. The native fixed gimbal joints replace the retired
model's extra PD motors. Native COM, inertia, wheel armature, knee limits and
spring travel are read back and checked; all limits retain their supplied values.

The working height interval is 0.24–0.35 m. This controller interval is narrower
than the full mechanical knee domain; it does not redefine the mechanical stops.
A native infinite plane is used for ground collision, and self collision remains
disabled as declared in the supplied asset.

Run from `wheelleg_controller_tuning`:

```sh
.venv/bin/python -m wheelleg_tuning.cli prepare
.venv/bin/python -m wheelleg_tuning.cli evaluate --output results/v202_baseline.json
```

Geometry tests:

```sh
PYTHONPATH=src /home/noir/Documents/workspace/wheelleg_controller_tuning/.venv/bin/python \
  -m pytest tests/test_full_robot_v202_traditional.py -q
```

Nine tests cover authenticated topology, canonical URDF transforms, CAD mass
properties, USD limits, closed poses across five heights, virtual work, the DARE
seed and rejected out-of-limit inputs. Native behavior evidence and remaining
work are in the tuning repository's `docs/V202_IMPLEMENTATION_STATUS.md`.

The previous v2.0.1 source/bundle/table/builder/test entry points were removed from
this worktree. Their history remains in Git. Existing RL environments, active
training and the original workspace are not switched by this task. No new PPO,
hardware motion, real-time deadline qualification or electrical compliance is
implied by successful asset integration.
