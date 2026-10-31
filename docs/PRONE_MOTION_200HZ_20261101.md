# 200Hz physics and feedback PD, 100Hz policy

`train_prone_motion.py --physics-hz 200 --position-iterations 64 --velocity-iterations 32`
selects a scoped simulation approximation. The original v4 task/control files
remain frozen. The resolved run binds a separate authenticated control with
5ms physics, 5ms fresh feedback PD, 10ms policy, and two native solver/feedback
cycles per policy sample. Each 64-step rollout still spans 0.64 simulated
seconds per environment. PPO budgets, rewards, gamma/GAE horizons, observations,
motor limits, measured wheel models, gas springs, friction and curriculum are
unchanged. The option rejects fewer than 64 position / 32 velocity iterations;
the earlier 12/6 coarse probes exceeded the existing 3mm closure guard.

Milliseconds are converted using the actual physical period. Encoder feedback
and torque delays use causal interpolation between adjacent physical samples
when a delay is fractional in solver ticks. Encoder sampling/holding periods
round upward to whole physical steps, with at least one step. Consequently a
nominal 1ms encoder period refreshes every 5ms in this approximation. The
historical 1ms path retains its existing integer delay and sample/hold behavior.
The identified left/right wheel aggregate delays remain 5.5/3.5ms and already
use fractional physical ticks. Motor time constants, gimbal time, passive
settling, impact windows and policy sensor ages retain seconds as their units.

`SubstepControlAudit` now instruments the actual robust task override, recording
one target decode and two PD/solver/feedback updates per policy sample. Native
probe and training context receipts persist this measured audit after stepping.
The two-domain zero-PPO probe replays measured wheel and spring paths at the
selected physical rate and checks native solver readback. These are wiring
checks, not learned behavior or hardware qualification.

5ms contact samples cannot resolve the user's 2ms loss tolerance. Coarse cold
acceptance records physical durations correctly and withholds full contact
acceptance; final qualification requires a separate 1kHz physical evaluation.
Curriculum admission retains its existing native geometry, task and motion
criteria and is not hardware acceptance.

Example fresh run (no checkpoint, teacher or optimizer restore):

```sh
scripts/train_prone_motion.sh --task WheelLeg-ProneMotion-v4 \
  --physics-hz 200 --position-iterations 64 --velocity-iterations 32 \
  --num-envs 256 --max-iterations 10000 --seed 106 --save-interval 250 \
  --domain-randomization curriculum --device cuda:0 --headless \
  --output /absolute/path/to/new/run
```

TensorBoard runs on Kaiser and reads the remote event files. Local access uses
the existing 6006 SSH forward; local port6007 belongs to another experiment.
