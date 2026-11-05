# Disabled recovery placement window

`wheel_leg_self_righting_passive_window_v1.json` extends the disabled placement
window of the recovery-focus return-horizon contract. Its minimum is 1.0 seconds,
maximum 1.2 seconds, and settled hold 0.05 seconds. The existing angular-speed
and motor-speed check remains, including its maximum-time override. This does
not certify static equilibrium. COM linear velocity and contact remain
diagnostics; they do not enter the actor or the enable state machine.

Only recovery lanes use these timers. Physical simulation, passive mechanical
loads, feedback, communication, wheel response and motor histories continue
throughout the disabled interval. Existing zero motor effort, previous-action
reset and original enable behavior remain. There is no extra pose reset,
teleport or history flush after placement. Auxiliary standing roles are unchanged.

The active recovery deadline, completion and stable hold, mechanical and impact
protections, original evaluation cases, and all subsequent acceptance requirements
remain unchanged. The source contract retains the same return weights, decoder,
feedback clock, reward, initialization and curriculum. A new experiment uses a
fresh actor, critic and empty optimizer; it cannot extend an earlier run.

The native probe reports original enable events and the state at the end of
their disabled physical interval. Those observations establish wiring and timer
behavior only. Observer on/off checks must extend beyond the maximum disabled
window; use 1.5 seconds for this contract. These short probes are not recovery
trials. The complete cold matrix keeps its 24-second world budget, leaving room
for the original 20-second active window. It does not execute a WheelBipe handoff.

Record every disabled physical step as well as the full active window. Report
disabled, active and whole-trajectory per-body ground impulse separately, with
each positive world-Z resultant integrated over a sliding 20 ms window before
grouping. Delaying enable must not conceal placement loads. An active scorer
pass does not replace this whole-trajectory audit or authorize a low-impulse
claim. Failure flags and null handoff times retain their original meanings.
