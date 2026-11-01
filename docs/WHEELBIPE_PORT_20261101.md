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
adaptive 1e-4 learning rate, and 50 Hz policy clock. The reference uses
200 Hz physics; this port explicitly uses 1000 Hz after the numerical
failures below. The main unified v4 defaults are unaffected.

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
within 1.33 micrometres. Three CPU tests additionally exercise independent
left/right lengths, reversed native joint order and CAD collision support.

The mirrored passive-knee reward coordinates and limits use the actual
40°–110° CAD branch. The wheel-axis alignment reference uses our 0.1833 m
hip offset. These are geometry substitutions; reference reward weights are
still unchanged. All source actuator/spring/sensor randomizers are disabled
and replaced once by the explicit v2 groups. The critic receives the actual
160/2.5 feedback gains and calibrated delays, rather than zero-gain native
effort actuators. Feedback PD, gas feedforward and physical spring force each
execute once per physics tick (1 ms in the admitted GPU pilot).

Initial local CPU native checks passed on flat (8 environments, 500 policy
steps, 85 resets, policy-boundary loop peak 2.108 mm) and rough (4 environments,
200 policy steps, 16 resets, policy-boundary peak 2.022 mm). Both used finite
raw states and observations and bounded delivered electrical torque. These
checks predate the final mirrored-knee substitution and per-physics-step peak
monitor; the frozen candidate is rechecked before remote PPO. They establish
engineering wiring only, not learned balance, self-righting or deployment.

Kaiser has a separate relocated Sim 5.1 / Lab 2.3.2 / RSL-RL 3.0.1 runtime at
`/home/kaiser/wheelbipe-sim51/env`; its existing Sim 6 runtime and TensorBoard
service remain independent.

### Numerical integration study and PPO admission

`reports/wheelbipe_port_20261101/numerical_study001.json` retains failed cases
and hashes of the full Kaiser receipts. The imported 200 Hz integration
cannot be admitted for this explicit, stiffer PD and CAD loop model: the
32-environment GPU 64/32 case reached 23.899 mm; a same-budget CPU case
reached 3.115 mm. Increasing solver iterations or restoring the original
12/6 solver did not solve this. None of these failures started full PPO.

At 50 Hz policy / 500 Hz physics, GPU native 32×300 policy steps passed with
a 1.881 mm peak. Real PPO at 128 environments then reached 3.733 mm in its
first update, so the requested full run was withheld. This also demonstrated
why finite state or a low-amplitude random-action probe alone is inadequate.
At 50 Hz policy / 1000 Hz physics, the 32×300 native probe reached 0.422 mm.
The first 128-environment GPU pilot at that clock failed at update 17 with
a 6.094 mm loop peak. Its reset poses exposed a second migration difference:
the reference root height placed our larger CAD collision geometry as much
as 14.15 cm below the ground. Reset now raises the root only when necessary,
using independently checked CAD collision support and 2 mm clearance. It
preserves the sampled orientation, joint pose, velocities and positive drop
height; there are no dynamic pose writes or relaxed loop limits. Even with
this reset correction, the 200 Hz GPU probe failed at 16.546 mm.

The corrected, frozen `47968e5d` GPU pilot completed all 30 fresh PPO updates
at 128 environments, with the 3 mm bound checked at every physics tick.
Its maximum loop gap was 1.601 mm over 14,400 physics ticks (24 policy steps
per update, 20 physics ticks per policy step), with finite policy tensors.
All 9 actor/std and 8 critic tensors changed; Adam started empty and finished
with 17 states. The training call took 425.865 seconds. Solver settings are 64/32;
external forces are integrated with the original Sim 5.1 default (`false`).
Rendering stays at 50 Hz, independent of finer feedback/integration. No old
checkpoint is migrated between clocks.

The local two-environment/two-update CPU runner check started with empty
Adam and changed all 9 actor/std and 8 critic tensors; Adam acquired 17
states. Its diagnostic ONNX export passed 1024 random-input parity samples
with maximum absolute error 2.98e-7. This tiny run verifies software wiring,
not learned ability. Cold evaluation entry points also completed seven
0.8-second cases; these are too short and too early for behavior acceptance.

The GPU pilot ONNX passes 1024 random-input comparisons with maximum
absolute error 9.5367e-7. Its checkpoint SHA-256 is
`387017a0836baee582daab529f118ef925d3b4e985c54369244628c5fe0632b2`.
The raw pilot and parity receipts are committed under
`reports/wheelbipe_port_20261101/`. These establish genuine PPO and export
wiring, not learned task capabilities. Reference reward weights are unchanged.

### Full-run ownership and progress

`scripts/run_wheelbipe_campaign.py` authenticates the admitted pilot,
implementation, collision geometry, ground plane, checkpoint and ONNX hashes.
It runs rough-terrain and full-population preflights, then a separate fresh
20,000-update flat run (512 environments, seed 109, save every 500 updates).
Kaiser's WSL has approximately 16 GiB RAM, so environment count is explicit
rather than inferred from the GPU's 24 GiB VRAM. The update budget and original
PPO settings are preserved; this is not a claim of matching the sample count
of an experiment with a different population.

The campaign exports the completed final checkpoint to ONNX, checks parity,
copies preprocessing parameters, runs seven cold nominal command diagnostics
and writes all 13 permanent capability statuses. Any physical, optimizer,
source-hash or export failure stops the campaign. It never silently resumes
or extends a failed training run. A process lock prevents a duplicate owner.

Current owner source is `b5337fb9`, identical to the admitted pilot's core.
The first rough preflight stalled before stepping because of missing
`libGLU.so.1`; it was stopped and its failure retained. A new independent
output directory, `full_flat_512_20000_1000hz_seed109_002`, uses the existing
Kaiser sysroot in this experiment's library path. No global runtime or other
training service was modified. The rough GPU preflight completed 200 policy /
4,000 physics steps with a 0.455 mm peak. The 512-environment preflight
completed 100 policy / 2,000 physics steps with a 0.490 mm peak. Both exited
normally. The subsequent fresh full run has completed its first 10 updates
(4,800 physics ticks), with a 1.271 mm cumulative loop peak. All 17
actor/std/critic tensors changed in the first update and Adam gained 17
states. Its initialization SHA differs from the pilot. The full ONNX is
pending completion of all 20,000 updates.

`reports/wheelbipe_port_20261101/kaiser_full_launch001.json` freezes the
observed receipts and their SHA-256, owner PID 854206, worker PID 855685,
the exact command/source, and TensorBoard scalar evidence. PIDs are historical
locators, not permission to signal a future process with the same number.
The independent training owner is in tmux `wheelbipe-full-002`.

Remote TensorBoard uses a separate service on port 6008. Local
`http://127.0.0.1:6008/` is only an SSH forward; existing 6006/6007 services
are untouched. The training owner and TensorBoard have independent lifetimes.
All 13 permanent capabilities remain unqualified until measured with their
full acceptance protocols; see the committed capability-status report.

The chat follow-up `wheelbipe-onnx` checks every 30 minutes, commits progress
at new 500-update checkpoint milestones, retrieves and verifies final outputs,
and stops after final delivery. It stays quiet during ordinary progress.
The remote training/export owner continues independently of that follow-up.

CLI entry points are `scripts/train_wheelbipe_port.py` (preflight/train/evaluate)
and `scripts/export_wheelbipe_port_onnx.py`. Output directories must be new;
training rejects `--checkpoint`. A final export requires a completed
20,000-update receipt and matching checkpoint hash. Diagnostic pilot exports
must explicitly use `--allow-pilot`.
