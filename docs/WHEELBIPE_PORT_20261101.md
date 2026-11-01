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

Status: source snapshot recorded; implementation and native pilot pending.
