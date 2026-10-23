# Unified full-RL standing observation v3

This candidate connects the 200 Hz unified self-righting actor to the existing
WheelLegController. It is opt-in through
`rmcs_ws/src/rmcs_bringup/config/wheel-leg-infantry-unified-rl.yaml` and
`unified_full_rl: true`. The active infantry profile and its model are unchanged.
The candidate's `unified_candidate.onnx` is deliberately absent: the research
actor has not passed the behavioral admission gate.

## Runtime contract

WheelLegController owns calibrated sensing, standing geometry/reference state,
the coherent 35-value frame, applied-action history, mechanical guards, drive
handshake and 1000 Hz host torque PD. RlBridge copies one frame without another
transform and runs asynchronous inference. Each positive producer sequence is
consumed once; duplicate frames do not refresh action age. Producer sequence
rollback latches rejection until a session reset.

The actor has 35 float32 features and six raw actions at 200 Hz, with
`paired_branch107_v1` winding and model SHA
`0ac82045ab584f20091538aa782d93ea545382dc701faca1f8ae92aeb7cd1997`.
The required `rmcs_observation_contract_sha256` metadata is
`db0fc6121c73121dcd39eef0524ecfeab4b3aa69aa7f5700d4205499af2402a9`.
Model loading rejects missing/mismatched hashes or policy frequency. The
controller also checks the frame paths, action decoding, frequency, torque
mode and absence of scripted recovery/heading/blend modifiers.

| Feature slots | Content |
| --- | --- |
| 0–3 | vx, vy, yaw rate, height ×5 |
| 4–9 | angular velocity ×0.5, projected gravity |
| 10–13 | wrapped model leg offsets |
| 14–15 | standing wheel-coordinate displacement ×20 |
| 16–21 | six model joint velocities ×0.1 |
| 22–27 | previous physically applied raw action |
| 28–34 | fixed ordinary-motion context [1,0,0,0,0,0,0] |

Standing geometry uses the frozen closed-chain lookup, wrapped leg channels,
continuous wheel angles and IMU gravity. It captures an eligible stationary
reference, filters with the training 150 ms constant, and clears it on RL
entry/reset, movement request, or valid severe tilt. Invalid geometry yields
zero displacement while preserving the reference. The declared 50 ms capture
comparison uses training float32 arithmetic; its first eligible sample is tick
11 (55 ms), not tick 10. No world-position sensor is assumed.

Mode 2 completes the drive handshake with zero actuator effort. Mode 3 enters
RL after a fresh valid inference at any body attitude. There is no scripted leg
positioning or recovery handoff in this mode. Motor faults, calibrated hinge
limits, stale/skewed sensing, action timeout and torque caps remain enforced.
The first passive action history comes from measured feasible leg coordinates;
subsequent frames use the action that reached the host PD computation.

## Offline verification (2026-10-23)

The repository `.script/build-rmcs` built rmcs_rl/rmcs_core in an isolated
worktree. All 38 individual gtests passed, including legacy policy/recovery
tests, coherent frame tests, invalid external-frame configuration tests, the
new observation helper and the production controller/bridge/executor using
memory-only sensors. The integration test requires
`RMCS_UNIFIED_TEST_PROFILE` pointing at a separately prepared diagnostic model
profile; it skips without that explicit fixture.

35,158 native-derived/synthetic frames (35,137 valid) matched the independent
Torch helper with full-frame error at most 4.77e-7 and passive-history error
1.91e-6. Reset/invalid/duplicate inputs and additional whole leg turns were
included. A diagnostic graph of research checkpoint 079 matched 2,048 recorded
Torch inputs within 3.34e-6 raw action. The production memory integration
checked inverted/side entry, rearm/reset, stale-sensor stop and exact serialized
applied-action history. These tests provide interface evidence, not closed-loop
self-righting or hardware qualification. No CAN or real robot was used.

Detailed machine-readable evidence is in the training repository's
`reports/unified_robust_20261021/runtime085_*.json`; the diagnostic graph is
explicitly unqualified and is not a delivery export.
