# Normally-up prone recovery

`contracts/wheel_leg_prone_support_v1.json` scopes recovery to a normally-up,
belly-grounded body. Thirty-two training leg configurations and eight held-out
configurations cover independent left/right knees and fore/aft wheel orbits.
Body roll/pitch, gimbal and all birth velocities are fixed. Reset parameters are
closed-chain solutions, screened against the original CAD surfaces with the
documented assembly exemptions. They are not static-equilibrium certificates.
Native reset admission must pass before PPO. The corrected chassis slope asset
is mandatory, including its original cooked-geometry admission threshold.

All six outputs belong to one fresh actor throughout the enabled interval.
The range-v2 encoder-relative common/relative decoder, 5ms feedback, physical
gas springs, leg friction and wheel delay remain active. No timed folding
trajectory, motor takeover, teacher or old policy is used. Actor inputs remain
the sensor-only 35D interface. Ground truth is used for rewards and scoring,
never added to the actor or used to switch an active episode's command.

The objective has three parts:

1. A bounded potential guides retraction and each wheel's fore/aft alignment
   beneath the whole-robot COM. Opposing left/right errors cannot cancel at a
   midpoint. Folding is soft guidance, fading with alignment and lift; it is
   not a completion condition or a sustained density bonus.
2. Dense support/lift and quiet-height rewards promote low supported balance
   and extension. Leg/chassis contact is allowed during the approach. Only the
   quiet wheel-supported goal bonus requires nonwheel support to become small.
3. Time, excursion, speed, effort, action changes and excess20ms impulse are
   penalized. Ordinary support by a leg is not itself penalized. Contact risk is
   a flat-ground resultant proxy, not structural stress or a hardware limit.

The existing height-command slot carries a goal fixed at episode reset.
Training starts at0.23m, then0.265m and finally the default0.305m stance.
Each leg class needs its latest16 ended episodes to achieve at least75%
actual3s captures before the next goal is assigned to future resets. Old-stage
episodes give no progression credit. PPO count and reward never advance goals.
Every episode starts prone; this is not nominal-standing rehearsal. Low-height
success is only a curriculum outcome, never reported as default standing.

Capture is checked after each completed native physics update: height error
at most20mm, tilt below10deg, COM speed below0.08m/s, body rate below0.5rad/s,
leg speed below3rad/s, each wheel within55mm longitudinally of the COM, both
wheels geometrically near the ground and measurably loaded, nonwheel load
below5% of weight, and displacement at most0.1m. Support interruptions up to2ms
are recorded and tolerated after capture begins; longer gaps reset the clock.
All other conditions must hold continuously for3s. Original mechanical and
impact guards remain in force. A simultaneous hard fault rejects completion.

`scripts/evaluate_prone_support.py` always uses0.305m and the eight held-out
configurations in two headings (16 cases), irrespective of training phase.
Completion must occur within20s of enable and hold for a further10s in the same
physical trajectory. Raw recording includes disabled placement and active
intervals. Positive world-Z force is integrated over20ms for each body before
part-group reduction; both complete-episode and enabled peaks are retained.
This entry point does not execute WheelBipe16001. Failed or absent handoffs
retain a null timestamp.

This is explicitly1000Hz physics,200Hz feedback/PD and100Hz policy. It does not
qualify the default200Hz plant, hardware or the13 long-term capabilities. A new
long budget requires independent actual behavior improvement; short budgets
are never extended. The legacy mixed locomotion/standing curriculum is absent;
this task has only prone births and the explicit height-outcome curriculum.
