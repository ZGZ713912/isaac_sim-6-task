# Inverted to stationary prone

`wheel_leg_rollover_learning_v1.json` defines a separate skill. It stops at a
normally-up, ground-supported low prone posture. This is not default standing,
full seven-family recovery, or a WheelBipe16001 handoff qualification.

The native plant, gas spring, motor friction, original mechanical/impact guards,
encoder-relative range-v2 actions and35D encoder/IMU interface remain in use.
The actor controls left common rotation, left signed motor-relative increment,
right common rotation, right relative increment, and both wheel velocities.
Mechanical knee angle is not the auxiliary motor's angle. Positive physical
common rotation uses opposite numerical signs on the mirrored left/right sides.

Training births use only the authenticated bank's inverted training split.
Both mechanical knees and fore/aft orbital positions vary independently;
all four front/rear combinations remain represented. Holdouts never enter
training. Body pose continues evolving while legs align. Alignment is a soft
reward and never gates actual orientation progress. The110degree shape is
guidance while inverted, not a final equality or a prerequisite for flipping.

Reward density (per second) combines a time cost, continuous normally-up
orientation, soft knee/alignment/coordinated-motion guidance while inverted,
and a slow, low normally-up capture score. Braking grows near normally-up.
Effort, target changes, shaft speed and excess20ms ground impulse are secondary
costs. The impulse is integrated for each body before reducing to part groups;
normal leg support is permitted. A bounded discounted state potential rewards
progress without paying accumulated rotations. A one-time10point capture bonus
contains the existing native3point completion term exactly once.

Capture requires every completed native1ms frame to satisfy tilt<=15degrees,
root-link height0.12..0.24m, root-COM speed<=0.08m/s, angular speed<=0.35rad/s,
four leg motor speeds<=1rad/s, and total positive ground-normal force>=20percent
of weight, continuously for1s. Chassis and legs may carry the robot; both wheels
need not touch. The native policy-boundary completion bridge then needs one
10ms check. Mechanical or impact faults reject simultaneous completions.
This height band is a prone capture region, not a claim of exact standard-prone
joint matching or the nominal0.305m standing reset posture.

The reverse takeover course uses real physical prefixes of1.7,1.2,0.6,then0s.
At least every fourth lane always has a zero prefix. A prefix moves each leg
from its own encoder angle, combining same-direction motion with circular
alignment feedback and bounded knee preparation. At release, only the actor's
three raw-action history buffers are zeroed. Physical state, wheel delay,
encoder/communication history, motor filters, gimbal and native clock remain
the same objects and are compared before/after. No post-prefix state write,
teacher network, imitation loss, or policy donor is used.

`RecoveryRolloutMask` excludes complete prefix intervals from PPO, normalizer
updates and GAE, with a separate bootstrap mask for the returned observation.
Only actor-controlled capture counts.64 ended episodes born in the current
stage and at least75percent successes unlock the next shorter prefix. Old-stage
episodes provide no advancement credit. Stage, outcomes and independent RNG
are saved. CLI update counts and reward do not advance the course. The current
entry supports fresh starts and in-process trainctl pause; checkpoint resume
for this new course is rejected until its full restore path is implemented.

Use the regular train_prone_motion launcher with this plan, explicit
1000Hz physics/200Hz PD/100Hz policy, eager backend, fresh networks and a finite
CLI update budget. The two-domain zero-PPO probe includes an explicitly shorter
30ms prefix fixture to test release wiring. It does not qualify behavior.

The independent cold evaluator uses all inverted holdouts in all four leg
classes and headings0/90degrees, with no prefix and the deterministic final
actor. It records every physical force/state/command/history frame and all
failures. A pass requires the native capture, no hard fault, and a further2s
continuous stationary-prone window inside the evaluation budget. Disabled
placement and enabled motion are reported separately, including whole-episode
20ms impulse. No16001 policy is run and its actual handoff time remains null.
1000Hz results do not establish transfer to200Hz or hardware.
