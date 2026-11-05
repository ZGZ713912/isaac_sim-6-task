# Self-righting experiment contracts

The `wheel_leg_self_righting_*_v1.json` plans are explicit local simulation
candidates. They retain the full friction-calibrated v2.0.2 mechanism,40–110°
mechanical knee range,1ms physics,200Hz held feedback PD and100Hz recovery actor.
They require fresh actor, critic and Adam state. They do not import a locomotion
policy into the recovery actor.

`standard_capture` retains the existing fold/capture/standing objective on one
standard posture. `phase_staged` uses continuous orientation, two-wheel support
capture and gated lift guidance with the existing mastery-based pose sequence.
`phase_all` exposes all seven admitted fallen families from the first update,
while retaining the same independent mastery tests. `phase_all_low_impact`
doubles the existing contact-integral cost weights. These weights are search
choices, not identified structural limits.

All multi-pose candidates reserve30% of training lanes for recovery practice
before standing passes. Availability of practice never unlocks harder motion,
randomization or a passed capability. Family, contact and support geometry are
reward/evaluation inputs only; actor observations and motor ownership stay the
same. Completion and post-completion behavior require cold evaluation.

The phase organization is informed by [quadruped fall recovery](https://discovery.ucl.ac.uk/id/eprint/10175510/1/Zhibin_Learning_Complex_Motor_Skills_for_Legged_Robot_Fall_Recovery_RAL_Revision.pdf),
[humanoid getting up](https://arxiv.org/abs/2502.12152), and
[wheel-leg coordinated recovery](https://arxiv.org/abs/2506.05516).
The mechanical and actuator limits come from this repository's own contracts.

Post-stand WheelBipe16001 evaluation must use its own35D observation order,
0.5rad leg action scale,60/2 leg PD,50Hz policy and200Hz physics/PD. Its order is
LH,RH,LK,RK,LW,RW; it differs from the chassis recovery actor. A successful
recovery-actor cold test alone does not establish that handoff. No simulation
result authorizes hardware motion.
