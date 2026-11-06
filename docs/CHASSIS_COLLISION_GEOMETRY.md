# v2.0.2 chassis collision geometry

All fixed-gimbal, moving-gimbal and friction training exports use the same
`model/wheel_leg_v2.0.2/chassis_collision/geometry.json` geometry. It contains
the 42 source-support-v3 hulls derived from the original chassis STL. Every
original supporting extreme is retained, including the front, rear and side
lower slopes. Each hull has at most 64 vertices and uses a 0.01 mm minimum
convex cooking thickness.

The old chassis boxes and the eight conservative 26-direction chassis
envelopes are removed from USD, URDF and model specifications. Their OBJ files
and experimental overlay assets are removed as well. Existing geometry plans
select one canonical wrapper over the corrected moving-gimbal training asset.
`tools/full_robot_v202.py` generates the same corrected geometry, and the
native chassis entry point rejects an explicit asset override with different
chassis hulls.

Only chassis collision geometry changes. Visuals, other bodies, joints, mass,
inertia and actuator parameters retain their values. Component evidence can be
inherited only after checking the specification with collisions removed; it
does not qualify the new contact geometry. Asset hashes and current training
contracts bind the corrected exports. Old native receipts, run records and
checkpoints retain their original identities. An already running process
continues to use its loaded scene; disk migration does not change that scene
or authorize checkpoint continuation under the new geometry.

The supporting-envelope guarantee applies to an infinite flat ground plane.
It does not certify CAD cavities, concave terrain contact, learned recovery or
hardware behavior. Native geometry admission must inspect actual cooked hulls
and retain the 0.1 mm support-error limit. Reset-bank clearance labels measured
with the old geometry require new native validation.
