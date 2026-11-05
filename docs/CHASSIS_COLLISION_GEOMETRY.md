# Source chassis contact geometry

`wheel_leg_rollover_chassis_slopes_v1.json` selects an authenticated collision
overlay for the inverted-to-prone task. The parent is the original v2.0.2
moving-gimbal, friction_v1 training asset. Its eight base_link collision parts
were conservative 26-direction envelopes; those envelopes fill outside the
original armor slopes. The overlay disables those eight parts and replaces
their exterior supporting envelope from the original visual mesh. It does not
replace the visuals, other bodies, joints, mass, inertia, actuators or materials.

`scripts/build_chassis_collision.py` intersects every old convex part with the
convex hull of the original chassis surface. All original extreme points must
remain covered. The resulting volumes are partitioned using face-to-interior
tetrahedra, grouped into convex parts with at most 64 vertices each. The union
retains each clipped volume; this partition does not use a box or restrict face
normals to a fixed direction set. The generated metadata binds the original
USD layers, STL, model specification, source extremes and all output vertices.

The guarantee concerns the chassis supporting envelope on an infinite flat
ground plane. Original cavities and concavities have not been certified for
terrain contact. Flat-ground support equivalence is not CAD watertightness,
self-collision, a material calibration, or hardware qualification.

The rollover adapter authenticates the selected USD and metadata before native
construction, using the existing asset-selection hook only inside that
constructor. An external asset override is rejected. Original passive evidence
and the bound base environment source remain unchanged; their validation does
not qualify the new contact geometry. After construction, every clone must have
exactly the new active chassis parts and disabled old parts. PhysX's synchronous
collision-representation query reads each actual cooked hull. Support error
against source extremes must be below 0.1 mm over source face normals, axes and
a deterministic sphere. Native admission fails before PPO if this check fails.
The startup report retains actual cooked vertices and signed support errors.

Original bank joint/root states and independent initial leg angles are retained
to isolate the collision correction. Old bank clearance labels must not be
interpreted as new-asset certification. Disabled placement and active recovery
remain physical intervals, with complete contact records; there is no hidden
pose rewrite to compensate for a changed collision shape.

The flip task, rewards, prefix curriculum and completion thresholds are
unchanged. A new geometry experiment starts with fresh actor, critic and empty
Adam. Old geometry checkpoints are not donors or continuation points. Geometry
and native admission establish model selection and contact shape, not learned
recovery, eventual standing, or a continuous handoff to WheelBipe.
