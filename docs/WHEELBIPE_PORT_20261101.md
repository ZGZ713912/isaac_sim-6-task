# Wheelbipe V14.2 → wheel-leg v2.0.2

Branch: `dev/wheelbipe-v202-port`. Base: unified branch `6ba08234`.
Reference: `third_party/wheelbipe/provenance.json`.

The requested experiment reuses Wheelbipe's latest ordinary V14.2 PPO
training, with a new actor, critic and empty Adam. The small pilot and the
20,000-update full run have separate output directories and seeds. A pilot
checkpoint is never a donor for full training.

## Port boundary

Preserve the reference's environment, reward formulas, command schedules,
35-dimensional actor contract, 24-step PPO rollout, ELU 256/128/64 networks,
adaptive 1e-4 learning rate, and 50/200 Hz policy/physics clocks.

Replace geometry, mass/COM/inertia, six loop constraints, reset IK, the
40°–110° included-angle domain, measured 15.62 wheel response, DM moving
friction and the measured gas-force/encoder-feedforward model. P/D stays
160/2.5 and the spring scale stays one. Wheel software torque is 4.5 Nm;
leg electrical torque is clipped to 40 Nm before adding physical loss.
The two gimbal bodies retain their CAD mass with their joints fixed at zero;
gimbal-only commands require a separately identified gimbal actuator.

## Execution and evidence

1. Authenticate the copied source and original v2.0.2 assets; check reset
   kinematics and joint/body correspondence.
2. Run native closed-chain, action, observation, finite-value, actuator and
   checkpoint checks on Kaiser. Exercise both flat and rough entry points.
3. Train a small fresh policy and verify actor/critic/Adam changes and
   deterministic replay. A running script alone is insufficient.
4. Only after these checks, run the reference's full 20,000 updates from
   scratch, then export and numerically compare ONNX against the actor.
5. Report all capabilities from `docs/WHEEL_LEG_ACCEPTANCE.md`, including
   untested or failing ones. Software migration, learned behavior and hardware
   acceptance are separate results.

Training metrics use RSL-RL TensorBoard events on Kaiser. Local 6007 is
reserved; no local TensorBoard is started. This experiment has no hardware IO.

## Implementation and current evidence

The original 187 source files remain byte-for-byte authenticated. The adapter
uses the original ordinary `OnPolicyRunner` and PPO configuration, not our
previous custom trainer. Eighteen joints and 21 bodies are exposed by a name
correspondence without rotating a CAD axis. The original reset IK is replaced
with a measured CAD lookup; 840 midpoint/rotation cases close all six loops
within 1.33 micrometres. Two CPU tests additionally exercise independent left
and right lengths, rotations and reversed native joint order.

The mirrored passive-knee reward coordinates and limits use the actual
40°–110° CAD branch. The wheel-axis alignment reference uses our 0.1833 m
hip offset. These are geometry substitutions; reference reward weights are
still unchanged. All source actuator/spring/sensor randomizers are disabled
and replaced once by the explicit v2 groups. The critic receives the actual
160/2.5 feedback gains and calibrated delays, rather than zero-gain native
effort actuators. Feedback PD, gas feedforward and physical spring force each
execute once per 5 ms physics tick.

Initial local CPU native checks passed on flat (8 environments, 500 policy
steps, 85 resets, policy-boundary loop peak 2.108 mm) and rough (4 environments,
200 policy steps, 16 resets, policy-boundary peak 2.022 mm). Both used finite
raw states and observations and bounded delivered electrical torque. These
checks predate the final mirrored-knee substitution and per-physics-step peak
monitor; the frozen candidate is rechecked before remote PPO. They establish
engineering wiring only, not learned balance, self-righting or deployment.

Kaiser has a separate relocated Sim 5.1 / Lab 2.3.2 / RSL-RL 3.0.1 runtime at
`/home/kaiser/wheelbipe-sim51/env`; its existing Sim 6 runtime and TensorBoard
service remain independent. Remote GPU preflight, small PPO, full training
and final ONNX parity are pending.

CLI entry points are `scripts/train_wheelbipe_port.py` (preflight/train/evaluate)
and `scripts/export_wheelbipe_port_onnx.py`. Output directories must be new;
training rejects `--checkpoint`. A final export requires a completed
20,000-update receipt and matching checkpoint hash. Diagnostic pilot exports
must explicitly use `--allow-pilot`.
